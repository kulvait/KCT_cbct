// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <algorithm>
#include <cstdlib>
#include <ctype.h>
#include <iostream>
#include <regex>
#include <string>

#include <unistd.h>

// External libraries
#include "CLI/CLI.hpp" //Command line parser
#include "ctpl_stl.h" //Threadpool

// Internal libraries
#include "ARGPARSE/parseArgs.h"
#include "CuttingVoxelProjector.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenProjectionMatrixReader.hpp"
#include "DEN/DenSupportedType.hpp"
#include "SMA/BufferedSparseMatrixFloatWritter.hpp"

using namespace CTL;

/**Arguments parsed by the main function.
 */
struct Args
{
    int parseArguments(int argc, char* argv[]);
    uint32_t platformId = 0;
    bool debug = false;
    std::string frameSpecs = "";
    int eachkth = 1;
    std::vector<int> frames;
    int threads = 1;
    // It is evaluated from -0.5, pixels are centerred at integer coordinates
    uint32_t projectionSizeX = 616;
    uint32_t projectionSizeY = 480;
    double pixelSpacingX = 0.616;
    double pixelSpacingY = 0.616;
    // Here (0,0,0) is in the center of the volume
    uint32_t volumeSizeX = 256;
    uint32_t volumeSizeY = 256;
    uint32_t volumeSizeZ = 199;
    uint32_t baseOffset = 0;
    bool noFrameOffset = false;
    std::string inputVolume;
    std::string inputProjectionMatrices;
    std::string outputProjection;
    bool force = false;
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
    app.add_option("-k,--each-kth", eachkth,
                   "Process only each k-th frame intended for output. The frames to output "
                   "are then 1st specified, 1+kN, N=1...\\infty if such frame exists. Parameter k "
                   "must be positive integer.")
        ->check(CLI::Range(1, 65535));
    app.add_option("-b,--base_offset", baseOffset, "Base offset of projections indexing.");
    app.add_flag("-n,--no_frame_offset", noFrameOffset,
                 "When this flag is specified no offset of the projections will be used when "
                 "writing voxel pixel relationship. Normally the offset will be "
                 "framenum*projx*projy, where framenum is the zero based order of the projection "
                 "matrix in the source file, projx is the x dimension of projection area and y is "
                 "the y dimension of projection area.");
    CLI::Option* px
        = app.add_option("--projx", projectionSizeX, "Dimension of detector, defaults to 616.");
    CLI::Option* py
        = app.add_option("--projy", projectionSizeY, "Dimension of detector, defaults to 480.");
    CLI::Option* psx = app.add_option("--pixel_spacing_x", pixelSpacingX,
                                      "Spacing of detector cells, defaults to 0.616.");
    CLI::Option* psy = app.add_option("--pixel_spacing_y", pixelSpacingY,
                                      "Spacing of detector cells, defaults to 0.616.");
    CLI::Option* vx
        = app.add_option("--volumex", volumeSizeX, "Dimension of volume, defaults to 256.");
    CLI::Option* vy
        = app.add_option("--volumey", volumeSizeY, "Dimension of volume, defaults to 256.");
    CLI::Option* vz
        = app.add_option("--volumez", volumeSizeZ, "Dimension of volume, defaults to 199.");

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
    vx->needs(vy)->needs(vz);
    vy->needs(vx)->needs(vz);
    vz->needs(vx)->needs(vy);
    try
    {
        app.parse(argc, argv);
        // If force is not set, then check if output file does not exist
        if(!force)
        {
            if(io::fileExists(outputProjection))
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
    plog::Severity verbosityLevel = plog::debug; // debug, info, ...
    char exepath[PATH_MAX + 1] = { 0 };
    readlink("/proc/self/exe", exepath, sizeof(exepath));
    std::string argv0(exepath);
    std::string csvLogFile
        = io::xprintf("/tmp/%s.csv", io::getBasename(argv0.c_str())); // Set NULL to disable
    std::string xpath = io::getParent(argv0);
    bool logToConsole = true;
    plog::PlogSetup plogSetup(verbosityLevel, csvLogFile, logToConsole);
    plogSetup.initLogging();
    LOGI << io::xprintf("Xpath is %s", xpath.c_str());
    LOGI << io::xprintf("argv[0] is %s", argv[0]);
    LOGI << io::xprintf("START %s", argv[0]);
    // Argument parsing
    Args a;
    int parseResult = a.parseArguments(argc, argv);
    if(parseResult != 0)
    {
        if(parseResult > 0)
        {
            return 0; // Exited sucesfully, help message printed
        } else
        {
            return -1; // Exited somehow wrong
        }
    }
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
        volume, inf.dimx(), inf.dimy(), inf.dimz(), xpath, a.debug);
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
    io::appendBytes(a.outputProjection, (uint8_t*)buf, 6);
    for(int f : a.frames)
    {
        matrix::ProjectionMatrix pm = dr->readMatrix(f);

        std::array<double, 3> sourcePosition = pm.sourcePosition();
        std::array<double, 3> normalToDetector = pm.normalToDetector();

        double x1, x2, y1, y2;
        pm.project(sourcePosition[0] + normalToDetector[0], sourcePosition[1] + normalToDetector[1],
                   sourcePosition[2] + normalToDetector[2], &x1, &y1);
        pm.project(100.0, 100.0, 100.0, &x2, &y2);
        double xspacing2 = a.pixelSpacingX * a.pixelSpacingX;
        double yspacing2 = a.pixelSpacingY * a.pixelSpacingY;
        double distance
            = std::sqrt((x1 - x2) * (x1 - x2) * xspacing2 + (y1 - y2) * (y1 - y2) * yspacing2);
        double x = 100.0 - sourcePosition[0];
        double y = 100.0 - sourcePosition[1];
        double z = 100.0 - sourcePosition[2];
        double norma = std::sqrt(x * x + y * y + z * z);
        x /= norma;
        y /= norma;
        z /= norma;
        double cos = normalToDetector[0] * x + normalToDetector[1] * y + normalToDetector[2] * z;
        double theta = std::acos(cos);
        double distToDetector = std::abs(distance / std::tan(theta));
        double scalingFactor = distToDetector * distToDetector / a.pixelSpacingX / a.pixelSpacingY;
        //        LOGI << io::xprintf("Distance to the detector is %fmm therefore scaling factor is
        //        %f.",
        //                            distToDetector, scalingFactor);

        cvp->project(projection, a.projectionSizeX, a.projectionSizeY, pm, scalingFactor);
        io::appendBytes(a.outputProjection, (uint8_t*)projection,
                        projectionElementsCount * sizeof(float));
        std::fill_n(projection, projectionElementsCount, float(0.0));
    }
    delete[] volume;
    delete[] projection;

    LOGI << io::xprintf("END %s", argv[0]);
}
