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
#include "ctpl_stl.h" //Threadpool

// Internal libraries
#include "ARGPARSE/parseArgs.h"
#include "CGLSReconstructor.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenProjectionMatrixReader.hpp"
#include "DEN/DenSupportedType.hpp"
#include "GLSQRReconstructor.hpp"

using namespace CTL;

/**Arguments parsed by the main function.
 */
struct Args
{
    int parseArguments(int argc, char* argv[]);
    uint32_t platformId = 0;
    bool debug = false;
    bool reportIntermediate = false;
    int threads = 1;
    // It is evaluated from -0.5, pixels are centerred at integer coordinates
    uint32_t projectionSizeX = 616;
    uint32_t projectionSizeY = 480;
    uint32_t projectionSizeZ;
    uint64_t totalProjectionsSize;
    double pixelSizeX = 0.616;
    double pixelSizeY = 0.616;
    double voxelSizeX = 1.0;
    double voxelSizeY = 1.0;
    double voxelSizeZ = 1.0;
    // Here (0,0,0) is in the center of the volume
    uint32_t volumeSizeX = 256;
    uint32_t volumeSizeY = 256;
    uint32_t volumeSizeZ = 199;
    uint64_t totalVolumeSize;
    uint32_t baseOffset = 0;
    uint32_t maxIterations = 10;
    double stoppingError = 0.01;
    bool noFrameOffset = false;
    std::string outputVolume;
    std::string inputProjectionMatrices;
    std::string inputProjections;
    bool force = false;
    uint32_t itemsPerWorkgroup = 256;
    bool glsqr = false;
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

    CLI::App app{ "OpenCL implementation of CGLS." };
    app.add_option("input_projections", inputProjections, "Input projections")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("input_projection_matrices", inputProjectionMatrices,
                   "Projection matrices to be input of the computation."
                   "Files in a DEN format that contains projection matricess to process.")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("output_volume", outputVolume, "Volume to project")->required();
    app.add_flag("-f,--force", force, "Overwrite outputProjection if it exists.");
    app.add_flag("--glsqr", glsqr, "Perform GLSQR instead of CGLS.");
    CLI::Option* psx = app.add_option("--pixel-sizex", pixelSizeX,
                                      "Spacing of detector cells, defaults to 0.616.");
    CLI::Option* psy = app.add_option("--pixel-sizey", pixelSizeY,
                                      "Spacing of detector cells, defaults to 0.616.");
    CLI::Option* vx
        = app.add_option("--volume-sizex", volumeSizeX, "Dimension of volume, defaults to 256.");
    CLI::Option* vy
        = app.add_option("--volume-sizey", volumeSizeY, "Dimension of volume, defaults to 256.");
    CLI::Option* vz
        = app.add_option("--volume-sizez", volumeSizeZ, "Dimension of volume, defaults to 199.");
    app.add_option("--voxel-sizex", voxelSizeX, "Spacing of voxels, defaults to 1.0.");
    app.add_option("--voxel-sizey", voxelSizeY, "Spacing of voxels, defaults to 1.0.");
    app.add_option("--voxel-sizez", voxelSizeZ, "Spacing of voxels, defaults to 1.0.");

    // Program flow parameters
    app.add_option("-j,--threads", threads, "Number of extra threads that application can use.")
        ->check(CLI::Range(0, 65535))
        ->group("Platform settings");
    app.add_option("-p,--platform_id", platformId, "OpenCL platform ID to use.")
        ->check(CLI::Range(0, 65535))
        ->group("Platform settings");
    app.add_flag("-d,--debug", debug, "OpenCL compilation including debugging information.")
        ->group("Platform settings");
    app.add_option("--items-per-workgroup", itemsPerWorkgroup,
                   "OpenCL parameter that is important for norm computation, defaults to 256.")
        ->check(CLI::Range(1, 65535))
        ->group("Platform settings");
    app.add_flag("--report-intermediate", reportIntermediate,
                 "Report intermediate values of x, defaults to false.")
        ->group("Platform settings");
    app.add_option("-i,--max_iterations", maxIterations,
                   "Maximum number of CGLS iterations, defaults to 10.")
        ->check(CLI::Range(1, 65535))
        ->group("Platform settings");
    app.add_option("-e", stoppingError, "Stopping error, defaults to 0.01.")
        ->check(CLI::Range(0.0, 1.00))
        ->group("Platform settings");
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
            if(io::fileExists(outputVolume))
            {
                std::string msg
                    = "Error: output file already exists, use --force to force overwrite.";
                LOGE << msg;
                return 1;
            }
        }
        // How many projection matrices is there in total
        io::DenFileInfo pmi(inputProjectionMatrices);
        io::DenFileInfo inf(inputProjections);
        projectionSizeX = inf.dimx();
        projectionSizeY = inf.dimy();
        projectionSizeZ = inf.dimz();
        totalVolumeSize = uint64_t(volumeSizeX) * uint64_t(volumeSizeY) * uint64_t(volumeSizeZ);
        totalProjectionsSize
            = uint64_t(projectionSizeX) * uint64_t(projectionSizeY) * uint64_t(projectionSizeZ);
        if(inf.dimz() != pmi.dimz())
        {
            std::string ERR = io::xprintf(
                "Projection matrices z dimension %d is different from projections z dimension %d.",
                pmi.dimz(), inf.dimz());
            LOGE << ERR;
            io::throwerr(ERR);
        }
        if(totalVolumeSize > INT_MAX)
        {
            io::throwerr(
                "Implement indexing by uint64_t matrix dimension overflow of voxels count.");
        }
        // End parsing arguments
        if(totalProjectionsSize > INT_MAX)
        {
            io::throwerr("Implement indexing by uint64_t matrix dimension overflow of projection "
                         "pixels count.");
        }
        io::DenSupportedType t = inf.getDataType();
        if(t != io::DenSupportedType::float_)
        {
            std::string ERR
                = io::xprintf("This program supports float projections only but the supplied "
                              "projection file %s is "
                              "of type %s",
                              inputProjections.c_str(), io::DenSupportedTypeToString(t).c_str());
            LOGE << ERR;
            io::throwerr(ERR);
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
        = io::xprintf("/tmp/%s.csv", io::getBasename(argv0.c_str()).c_str()); // Set NULL to disable
    std::string xpath = io::getParent(argv0);
    bool logToConsole = true;
    plog::PlogSetup plogSetup(verbosityLevel, csvLogFile, logToConsole);
    plogSetup.initLogging();
    auto start = std::chrono::steady_clock::now();
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
    float* projection = new float[a.totalProjectionsSize];
    io::readBytesFrom(a.inputProjections, 6, (uint8_t*)projection, a.totalProjectionsSize * 4);
    std::string startPath = io::getParent(a.outputVolume);
    std::string bname = io::getBasename(a.outputVolume);
    bname = bname.substr(0, bname.find_last_of("."));
    startPath = io::xprintf("%s/%s_", startPath.c_str(), bname.c_str());
    LOGI << io::xprintf("startpath=%s", startPath.c_str());
    if(!a.glsqr)
    {
        std::shared_ptr<CGLSReconstructor> cgls = std::make_shared<CGLSReconstructor>(
            a.projectionSizeX, a.projectionSizeY, a.projectionSizeZ, a.pixelSizeX, a.pixelSizeY,
            a.volumeSizeX, a.volumeSizeY, a.volumeSizeZ, a.voxelSizeX, a.voxelSizeY, a.voxelSizeZ,
            xpath, a.debug, a.itemsPerWorkgroup, a.reportIntermediate, startPath);
        int res = cgls->initializeOpenCL(a.platformId);
        if(res < 0)
        {
            std::string ERR = io::xprintf("Could not initialize OpenCL platform %d.", a.platformId);
            LOGE << ERR;
            io::throwerr(ERR);
        }
        float* volume = new float[a.totalVolumeSize]();
        // testing
        //    io::readBytesFrom("/tmp/X.den", 6, (uint8_t*)volume, a.totalVolumeSize * 4);

        cgls->initializeVectors(projection, volume);
        uint16_t buf[3];
        buf[0] = a.volumeSizeY;
        buf[1] = a.volumeSizeX;
        buf[2] = a.volumeSizeZ;
        io::createEmptyFile(a.outputVolume, 0, true);
        io::appendBytes(a.outputVolume, (uint8_t*)buf, 6);
        cgls->reconstruct(dr, a.maxIterations, a.stoppingError);
        io::appendBytes(a.outputVolume, (uint8_t*)volume, a.totalVolumeSize * sizeof(float));
        delete[] volume;
        delete[] projection;
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start);
        LOGI << io::xprintf("END %s, duration %d ms.", argv[0], duration.count());
    } else
    {
        std::shared_ptr<GLSQRReconstructor> glsqr = std::make_shared<GLSQRReconstructor>(
            a.projectionSizeX, a.projectionSizeY, a.projectionSizeZ, a.pixelSizeX, a.pixelSizeY,
            a.volumeSizeX, a.volumeSizeY, a.volumeSizeZ, a.voxelSizeX, a.voxelSizeY, a.voxelSizeZ,
            xpath, a.debug, a.itemsPerWorkgroup, a.reportIntermediate, startPath);
        int res = glsqr->initializeOpenCL(a.platformId);
        if(res < 0)
        {
            std::string ERR = io::xprintf("Could not initialize OpenCL platform %d.", a.platformId);
            LOGE << ERR;
            io::throwerr(ERR);
        }
        float* volume = new float[a.totalVolumeSize]();
        // testing
        //    io::readBytesFrom("/tmp/X.den", 6, (uint8_t*)volume, a.totalVolumeSize * 4);

        glsqr->initializeVectors(projection, volume);
        uint16_t buf[3];
        buf[0] = a.volumeSizeY;
        buf[1] = a.volumeSizeX;
        buf[2] = a.volumeSizeZ;
        io::createEmptyFile(a.outputVolume, 0, true);
        io::appendBytes(a.outputVolume, (uint8_t*)buf, 6);
        glsqr->reconstruct(dr, a.maxIterations, a.stoppingError);
        io::appendBytes(a.outputVolume, (uint8_t*)volume, a.totalVolumeSize * sizeof(float));
        delete[] volume;
        delete[] projection;
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start);
        LOGI << io::xprintf("END %s, duration %d ms.", argv[0], duration.count());
    }
}
