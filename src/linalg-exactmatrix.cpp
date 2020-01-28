// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <algorithm>
#include <cstdlib>
#include <ctype.h>
#include <iostream>
#include <regex>
#include <string>

// External libraries
#include "CLI/CLI.hpp" //Command line parser
#include "ctpl_stl.h" //Threadpool

// Internal libraries
#include "PROG/parseArgs.h"
#include "DEN/DenProjectionMatrixReader.hpp"
#include "SMA/BufferedSparseMatrixFloatWritter.hpp"

#include "VolumeFootprintExecutor.hpp"

using namespace CTL;

struct Args
{
    int parseArguments(int argc, char* argv[]);
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
    std::string projectionMatrices;
    std::string outputSystemMatrix;
    bool force = false;
};

/**Argument parsing
 *
 */
int Args::parseArguments(int argc, char* argv[])
{

    CLI::App app{ "Using divide and conquer techniques to construct CT system matrix.." };
    app.add_option("input_matrices", projectionMatrices,
                   "Files in a DEN format that contains projection matricess to process.")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("output_system_matrix", outputSystemMatrix,
                   "File in a sparse matrix format to output or prefix of files.")
        ->required();
    app.add_flag("--force", force, "Overwrite outputSystemMatrix if it exists.");
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
    app.add_option("-j,--threads", threads, "Number of extra threads that application can use.")
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
            if(io::pathExists(outputSystemMatrix))
            {
                std::string msg
                    = "Error: output file already exists, use --force to force overwrite.";
                LOGE << msg;
                return 1;
            }
        }
        // How many projection matrices is there in total
        io::DenFileInfo di(projectionMatrices);
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
    std::string csvLogFile = io::xprintf(
        "/tmp/%s.csv", io::getBasename(std::string(argv[0])).c_str()); // Set NULL to disable
    bool logToConsole = true;
    plog::PlogSetup plogSetup(verbosityLevel, csvLogFile, logToConsole);
    plogSetup.initLogging();
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
        = std::make_shared<io::DenProjectionMatrixReader>(a.projectionMatrices);
    int count = dr->count();
    if(uint64_t(a.volumeSizeX) * uint64_t(a.volumeSizeY) * uint64_t(a.volumeSizeZ) > INT_MAX)
    {
        io::throwerr("Implement indexing by uint64_t matrix dimension overflow of voxels count.");
    }
    if(uint64_t(a.projectionSizeX) * uint64_t(a.projectionSizeY) * uint64_t(count) > INT_MAX)
    {
        io::throwerr(
            "Implement indexing by uint64_t matrix dimension overflow of projection pixels count.");
    }
    matrix::ProjectionMatrix pm = dr->readMatrix(0);

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
    LOGI << io::xprintf("Distance to the detector is %fmm therefore scaling factor is %f.",
                        distToDetector, scalingFactor);

    // Write individual submatrices
    LOGD << io::xprintf("Number of projections to process is %d.", a.frames.size());
    // End parsing arguments
    std::shared_ptr<matrix::BufferedSparseMatrixFloatWritter> matrixWritter
        = std::make_shared<matrix::BufferedSparseMatrixFloatWritter>(a.outputSystemMatrix, 8192,
                                                                     true);
    util::VolumeFootprintExecutor dfe(matrixWritter, a.projectionSizeX, a.projectionSizeY,
                                      a.volumeSizeX, a.volumeSizeY, a.volumeSizeZ, scalingFactor,
                                      a.threads);
    uint32_t projnum;
    uint32_t pixelIndexOffset;
    for(std::size_t i = 0; i != a.frames.size(); i++)
    {
        dfe.startThreadpool();
        projnum = a.frames[i];
        LOGD << io::xprintf("Processing projections from %dth position.", projnum);
        if(a.noFrameOffset)
        {
            pixelIndexOffset = a.baseOffset;
        } else
        {
            pixelIndexOffset = a.baseOffset + projnum * a.projectionSizeX * a.projectionSizeY;
        }
        matrix::ProjectionMatrix pm = dr->readMatrix(projnum);
        dfe.insertMatrixProjections(pm, pixelIndexOffset);
        dfe.stopThreadpool();
        dfe.reportNumberOfWrites();
    }
    LOGI << io::xprintf("END %s", argv[0]);
}
