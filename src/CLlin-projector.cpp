// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctype.h>
#include <iostream>
#include <regex>
#include <string>
#include <unistd.h>

// External libraries
#include "CLI/CLI.hpp" //Command line parser

// Internal libraries
#include "CArmArguments.hpp"
#include "CuttingVoxelProjector.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "DEN/DenProjectionMatrixReader.hpp"
#include "DEN/DenSupportedType.hpp"
#include "PROG/Program.hpp"
#include "SMA/BufferedSparseMatrixFloatWritter.hpp"

using namespace CTL;

/**Arguments parsed by the main function.
 */
struct Args
{
    uint32_t platformId = 0;
    bool debug = false;
    std::string frameSpecs = "";
    int eachkth = 1;
    std::vector<int> frames;
    int threads = 1;
    // It is evaluated from -0.5, pixels are centerred at integer coordinates
    uint32_t projectionSizeX = 616;
    uint32_t projectionSizeY = 480;
    double pixelSizeX = 0.616;
    double pixelSizeY = 0.616;
    double voxelSizeX = 1.0;
    double voxelSizeY = 1.0;
    double voxelSizeZ = 1.0;
    // Here (0,0,0) is in the center of the volume
    uint32_t volumeSizeX = 256;
    uint32_t volumeSizeY = 256;
    uint32_t volumeSizeZ = 199;
    uint32_t baseOffset = 0;
    bool noFrameOffset = false;
    bool centerVoxelProjector = false;
    bool useCosScaling = false;
    std::string inputVolume;
    std::string inputProjectionMatrices;
    std::string outputProjection;
    std::string rightHandSide = "";
    bool force = false;
    bool sidon = false;
    bool t3 = false;
    uint32_t probesPerEdge = 1;

    int parseArguments(int argc, char* argv[]);
};

/**Argument parsing
 *
 * @param argc
 * @param argv[]
 *
 * @return Returns 0 on success and nonzero for some error.
 */
int Args::parseArguments(int argc, char* argv[])
{

    CLI::App app{ "OpenCL implementation of the voxel cutting projector." };
    app.add_option("input_volume", inputVolume, "Volume to project")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("input_projection_matrices", inputProjectionMatrices,
                   "Projection matrices to be input of the computation."
                   "Files in a DEN format that contains projection matricess to process.")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("output_projection", outputProjection, "Output projection")->required();
    app.add_flag("--force", force, "Overwrite outputProjection if it exists.");

    app.add_option("-f,--frames", frameSpecs,
                   "Specify only particular projection matrices to process. You can input "
                   "range i.e. 0-20 or individual comma separated frames i.e. 1,8,9. Order "
                   "does matter. Accepts end literal that means total number of slices of the "
                   "input.");
    CLI::Option* sidon_opt = app.add_flag("-s,--sidon", sidon, "Use Sidon's projector");
    CLI::Option* t3_opt = app.add_flag("--tt", t3, "Use TT projector with A3 amplitude.");
    sidon_opt->excludes(t3_opt);
    t3_opt->excludes(sidon_opt);

    app.add_flag("--cos-scaling", useCosScaling, "Scaling scheme with f^2/cos^3.");
    CLI::Option* ppe
        = app.add_option("--probes-per-edge", probesPerEdge,
                         "Number of probes in each pixel edge, complexity scales with the "
                         "square of this number. Defaults to 1")
              ->check(CLI::Range(1, 1000));
    ppe->needs(sidon_opt);
    app.add_option("-k,--each-kth", eachkth,
                   "Process only each k-th frame intended for output. The frames to output "
                   "are then 1st specified, 1+kN, N=1...\\infty if such frame exists. Parameter k "
                   "must be positive integer.")
        ->check(CLI::Range(1, 65535));
    app.add_option("-b,--base_offset", baseOffset, "Base offset of projections indexing.");
    app.add_option("--right-hand-side", rightHandSide,
                   "If the parameter is specified, then we also compute the norm of the right hand "
                   "side from the projected vector.");
    app.add_flag("--center-voxel-projector", centerVoxelProjector,
                 "Use center voxel projector instead of cutting voxel projector.");
    CLI::Option* px = app.add_option("--projection-sizex", projectionSizeX,
                                     "Dimension of detector, defaults to 616.");
    CLI::Option* py = app.add_option("--projection-sizey", projectionSizeY,
                                     "Dimension of detector, defaults to 480.");
    CLI::Option* psx = app.add_option("--pixel-sizex", pixelSizeX,
                                      "Spacing of detector cells, defaults to 0.616.");
    CLI::Option* psy = app.add_option("--pixel-sizey", pixelSizeY,
                                      "Spacing of detector cells, defaults to 0.616.");
    CLI::Option* vsx
        = app.add_option("--voxel-sizex", voxelSizeX, "Spacing of voxels, defaults to 1.0.");
    CLI::Option* vsy
        = app.add_option("--voxel-sizey", voxelSizeY, "Spacing of voxels, defaults to 1.0.");
    CLI::Option* vsz
        = app.add_option("--voxel-sizez", voxelSizeZ, "Spacing of voxels, defaults to 1.0.");
    // Program flow parameters
    app.add_option("-j,--threads", threads, "Number of extra threads that application can use.")
        ->check(CLI::Range(0, 65535))
        ->group("Platform settings");
    app.add_option("-p,--platform_id", platformId, "OpenCL platform ID to use.")
        ->check(CLI::Range(0, 65535))
        ->group("Platform settings");
    app.add_flag("-d,--debug", debug, "OpenCL compilation including debugging information.")
        ->group("Platform settings");
    px->needs(py);
    py->needs(px);
    psx->needs(psy);
    psy->needs(psx);
    vsx->needs(vsy)->needs(vsz);
    vsy->needs(vsx)->needs(vsz);
    vsz->needs(vsx)->needs(vsy);
    try
    {
        app.parse(argc, argv);
        // If force is not set, then check if output file does not exist
        if(!force)
        {
            if(io::pathExists(outputProjection))
            {
                std::string msg
                    = "Error: output file already exists, use --force to force overwrite.";
                LOGE << msg;
                return 1;
            }
        }
        // How many projection matrices is there in total
        io::DenFileInfo di(inputProjectionMatrices);
        std::vector<int> f = util::processFramesSpecification(frameSpecs, di.getNumSlices());
        for(std::size_t i = 0; i != f.size(); i++)
        {
            if(i % eachkth == 0)
            {
                frames.push_back(f[i]);
            }
        }
    } catch(const CLI::CallForHelp e)
    {
        app.exit(e); // Prints help message
        return 1;
    } catch(const CLI::ParseError& e)
    {
        int exitcode = app.exit(e);
        LOGE << io::xprintf("There was perse error with exit code %d catched.\n %s", exitcode,
                            app.help().c_str());
        return -1;
    } catch(...)
    {
        LOGE << "Unknown exception catched";
        return -1;
    }
    return 0;
}

int main(int argc, char* argv[])
{
    using namespace CTL::util;
    Program PRG(argc, argv);
    PRG.startLog(true);
    // Argument parsing
    Args a;
    int parseResult = a.parseArguments(argc, argv);
    if(parseResult > 0)
    {
        return 0; // Exited sucesfully, help message printed
    } else if(parseResult != 0)
    {
        return -1; // Exited somehow wrong
    }
    std::string xpath = PRG.getRunTimeInfo().getExecutableDirectoryPath();
    std::shared_ptr<io::DenProjectionMatrixReader> dr
        = std::make_shared<io::DenProjectionMatrixReader>(a.inputProjectionMatrices);
    io::DenFileInfo pmi(a.inputProjectionMatrices);
    if(uint64_t(a.volumeSizeX) * uint64_t(a.volumeSizeY) * uint64_t(a.volumeSizeZ) > INT_MAX)
    {
        io::throwerr("Implement indexing by uint64_t matrix dimension overflow of voxels count.");
    }
    if(uint64_t(a.projectionSizeX) * uint64_t(a.projectionSizeY) * uint64_t(pmi.dimz()) > INT_MAX)
    {
        io::throwerr(
            "Implement indexing by uint64_t matrix dimension overflow of projection pixels count.");
    }

    // Write individual submatrices
    LOGD << io::xprintf("Number of projections to process is %d.", a.frames.size());
    // End parsing arguments
    io::DenFileInfo inf(a.inputVolume);
    io::DenSupportedType t = inf.getDataType();
    if(t != io::DenSupportedType::float_)
    {
        io::throwerr(
            "This program supports float volumes only but the supplied volume is of type %s!",
            io::DenSupportedTypeToString(t).c_str());
    }
    uint64_t totalVolumeSize = uint64_t(inf.dimx()) * uint64_t(inf.dimy()) * uint64_t(inf.dimz());
    float* volume = new float[totalVolumeSize];
    io::readBytesFrom(a.inputVolume, 6, (uint8_t*)volume, totalVolumeSize * 4);
    std::shared_ptr<CuttingVoxelProjector> cvp = std::make_shared<CuttingVoxelProjector>(
        volume, inf.dimx(), inf.dimy(), inf.dimz(), a.voxelSizeX, a.voxelSizeY, a.voxelSizeZ,
        a.pixelSizeX, a.pixelSizeY, xpath, a.debug, a.centerVoxelProjector, !a.useCosScaling);
    int res = cvp->initializeOpenCL(a.platformId);
    if(res < 0)
    {
        std::string ERR = io::xprintf("Could not initialize OpenCL platform %d.", a.platformId);
        LOGE << ERR;
        io::throwerr(ERR);
    }
    cvp->initializeVolumeImage();
    uint32_t projectionElementsCount = a.projectionSizeX * a.projectionSizeY;
    float* projection = new float[projectionElementsCount]();
    uint16_t buf[3];
    buf[0] = a.projectionSizeY;
    buf[1] = a.projectionSizeX;
    buf[2] = a.frames.size();
    io::createEmptyFile(a.outputProjection, 0, true); // Try if this is faster
    io::appendBytes(a.outputProjection, (uint8_t*)buf, (uint64_t)6);
    double normSquare = 0;
    double normSquareDifference = 0;
    std::shared_ptr<io::DenFrame2DReader<float>> dpr = nullptr;
    if(!a.rightHandSide.empty())
    {
        dpr = std::make_shared<io::DenFrame2DReader<float>>(a.rightHandSide);
    }
    double xoveryspacing = a.pixelSizeX / a.pixelSizeY;
    double yoverxspacing = a.pixelSizeY / a.pixelSizeX;
    for(int f : a.frames)
    {
        matrix::ProjectionMatrix pm = dr->readMatrix(f);

        double x1, x2, y1, y2;
        std::array<double, 3> sourcePosition = pm.sourcePosition();
        std::array<double, 3> normalToDetector = pm.normalToDetector();
        std::array<double, 3> tangentToDetector = pm.tangentToDetectorYDirection();
        pm.project(sourcePosition[0] - normalToDetector[0], sourcePosition[1] - normalToDetector[1],
                   sourcePosition[2] - normalToDetector[2], &x1, &y1);
        pm.project(sourcePosition[0] - normalToDetector[0] + tangentToDetector[0],
                   sourcePosition[1] - normalToDetector[1] + tangentToDetector[1],
                   sourcePosition[2] - normalToDetector[2] + tangentToDetector[2], &x2, &y2);
        double scalingFactor
            = (x1 - x2) * (x1 - x2) * xoveryspacing + (y1 - y2) * (y1 - y2) * yoverxspacing;
        if(a.sidon)
        {
            cvp->projectSiddon(projection, a.projectionSizeX, a.projectionSizeY, pm, scalingFactor,
                               a.probesPerEdge);
        } else if(a.t3)
        {
            double sourceToDetector
                = std::sqrt((x1 - x2) * (x1 - x2) * a.pixelSizeX * a.pixelSizeX
                            + (y1 - y2) * (y1 - y2) * a.pixelSizeY * a.pixelSizeY);
            cvp->projectTA3(projection, a.projectionSizeX, a.projectionSizeY, x1, y1,
                            sourceToDetector, pm);
        } else
        {
            if(a.useCosScaling)
            {
                cvp->projectCos(projection, a.projectionSizeX, a.projectionSizeY, pm,
                                scalingFactor);
            } else
            {
                double sourceToDetector
                    = std::sqrt((x1 - x2) * (x1 - x2) * a.pixelSizeX * a.pixelSizeX
                                + (y1 - y2) * (y1 - y2) * a.pixelSizeY * a.pixelSizeY);
                cvp->projectExact(projection, a.projectionSizeX, a.projectionSizeY, x1, y1,
                                  sourceToDetector, pm);
            }
        }
        if(dpr != nullptr)
        {
            std::shared_ptr<io::BufferedFrame2D<float>> fr = dpr->readBufferedFrame(f);
            normSquare += cvp->normSquare((float*)fr->getDataPointer(), a.projectionSizeX,
                                          a.projectionSizeY);
            normSquareDifference += cvp->normSquareDifference((float*)fr->getDataPointer(),
                                                              a.projectionSizeX, a.projectionSizeY);
        }
        io::appendBytes(a.outputProjection, (uint8_t*)projection,
                        projectionElementsCount * sizeof(float));
        std::fill_n(projection, projectionElementsCount, float(0.0));
    }
    delete[] volume;
    delete[] projection;
    if(dpr != nullptr)
    {
        LOGI << io::xprintf(
            "Initial norm is %f, norm of the difference  is %f that is %f%% of the initial norm.",
            std::sqrt(normSquare), std::sqrt(normSquareDifference),
            std::sqrt(normSquareDifference / normSquare) * 100);
    }
    PRG.endLog(true);
}
