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
#include "CGLSPerfusionReconstructor.hpp"
#include "FUN/FourierSeries.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenProjectionMatrixReader.hpp"
#include "DEN/DenSupportedType.hpp"

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
    uint32_t projectionsSizeX = 616;
    uint32_t projectionsSizeY = 480;
    uint32_t projectionsSizeZ = 248;
    uint64_t totalProjectionsSize;
    double pixelSpacingX = 0.616;
    double pixelSpacingY = 0.616;
    // Here (0,0,0) is in the center of the volume
    uint32_t volumeSizeX = 256;
    uint32_t volumeSizeY = 256;
    uint32_t volumeSizeZ = 199;
    uint64_t totalVolumeSize;
    uint32_t baseOffset = 0;
    uint32_t maxIterations = 10;
    bool noFrameOffset = false;
    std::string outputVolume;
    std::string inputProjectionMatrices;
    std::vector<std::string> inputProjections;
    bool force = false;
    uint32_t itemsPerWorkgroup = 256;
    float start_offset = 0.0, end_offset = 0.0;
    /**
     *Size of pause between sweeps [ms].
     *
     *Computed from DICOM files as 2088.88889ms. Based on experiment, it is 1171ms.
     */
    float pause_size = 1171;

    /** Frame Time. (0018, 1063) Nominal time (in msec) per individual frame.
     *
     *The model assumes that there is delay between two consecutive frames of the frame_time.
     *First frame is aquired directly after pause. From DICOM it is 16.6666667ms. From
     *experiment 16.8ms.
     */
    float frame_time = 16.8;
    /**Fourier functions degree*/
    uint32_t degree = 7;
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

    CLI::App app{ "OpenCL implementation of CGLS to perfusion reconstruction." };
    app.add_option("output_volume", outputVolume, "Volume to project")->required();
    app.add_option("input_projection_matrices", inputProjectionMatrices,
                   "Projection matrices to be input of the computation."
                   "Files in a DEN format that contains projection matricess to process.")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option(
           "projection_files", inputProjections,
           "Projection files in a DEN format to use for linear regression. All of them must "
           "correspond with its order to the projection_matrices file. Therefore they need to "
           "be reversed for backward sweep.")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_flag("--force", force, "Overwrite outputProjection if it exists.");
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

    app.add_option("--degree", degree, "Number of Fourier funtions to fit. Defaults to 7.")
        ->check(CLI::Range(1, 65535));
    app.add_option("--frame-time", frame_time,
                   "Frame Time. (0018, 1063) Nominal time (in msec) per individual frame (slice) "
                   "[ms]. Might be supplied for fine tuning of the algorithm. [default is "
                   "16.8]")
        ->check(CLI::Range(0.01, 10000.0));
    app.add_option("-s,--pause-size", pause_size,
                   "Size of pause [ms]. This might be supplied for fine tuning of the algorithm."
                   "[default is 1171] ")
        ->check(CLI::Range(0.01, 100000.0));
    app.add_option("-i,--start-offset", start_offset,
                   "From frame_time and pause_size is estimated the support interval of base "
                   "functions. First few milisecons might however be used to produce mask image "
                   "and therefore could be excluded from the time development. This parameter "
                   "controls the size of the time offset [ms] that should be identified with 0.0 "
                   "that is start of the support of the basis functions [defaults to 0.0].")
        ->check(CLI::Range(0.0, 100000.0));
    app.add_option(
           "-e,--end-offset", end_offset,
           "From frame_time and pause_size is estimated the support interval of base "
           "functions. There might be however extrapolation problems when fitting functions "
           "after last time stamp and therefore last miliseconds of the time development "
           "could be excluded from the support [ms] and should be identified with the end of the "
           "support of the basis functions [defaults to 0.0].")
        ->check(CLI::Range(0.0, 100000.0));
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
    app.add_option("--max_iterations", maxIterations,
                   "Maximum number of CGLS iterations, defaults to 10.")
        ->check(CLI::Range(1, 65535))
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
        io::DenFileInfo inf(inputProjections[0]);
        projectionsSizeX = inf.dimx();
        projectionsSizeY = inf.dimy();
        projectionsSizeZ = inf.dimz();
        totalVolumeSize = uint64_t(volumeSizeX) * uint64_t(volumeSizeY) * uint64_t(volumeSizeZ);
        totalProjectionsSize
            = uint64_t(projectionsSizeX) * uint64_t(projectionsSizeY) * uint64_t(projectionsSizeZ);
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
                              inputProjections[0].c_str(), io::DenSupportedTypeToString(t).c_str());
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
    float* projection;
    float* basisVals;
    float vals[a.degree];

    std::vector<float*> projections;
    std::vector<float*> basisFunctionsValues;
    double mean_sweep_time = (a.projectionsSizeZ - 1) * a.frame_time + a.pause_size;
    double startTime = a.start_offset;
    double endTime = (a.inputProjections.size() - 1) * mean_sweep_time
        + (a.projectionsSizeZ - 1) * a.frame_time - a.end_offset;
    std::shared_ptr<util::VectorFunctionI> baseFunctionsEvaluator
        = std::make_shared<util::FourierSeries>(a.degree, startTime, endTime);
    for(std::size_t j = 0; j != a.degree; j++)
    {
        basisVals = new float[a.projectionsSizeZ * a.inputProjections.size()];
        basisFunctionsValues.push_back(basisVals);
    }
    for(std::size_t i = 0; i != a.inputProjections.size(); i++)
    {
        projection = new float[a.totalProjectionsSize];
        io::readBytesFrom(a.inputProjections[i], 6, (uint8_t*)projection,
                          a.totalProjectionsSize * 4);
        projections.push_back(projection);
        for(std::size_t j = 0; j != a.projectionsSizeZ; j++)
        {
            baseFunctionsEvaluator->valuesAt(i * mean_sweep_time + j * a.frame_time, vals);
            for(std::size_t k = 0; k != a.degree; k++)
            {
                basisFunctionsValues[k][i * a.projectionsSizeZ + j] = vals[k];
            }
        }
    }

    std::string startPath = io::getParent(a.outputVolume);
    std::string bname = io::getBasename(a.outputVolume);
    bname = bname.substr(0, bname.find_last_of("."));
    startPath = io::xprintf("%s/%s_", startPath.c_str(), bname.c_str());
    LOGI << io::xprintf("startpath=%s", startPath.c_str());
    std::shared_ptr<CGLSPerfusionReconstructor> cgls = std::make_shared<CGLSPerfusionReconstructor>(
        a.projectionsSizeX, a.projectionsSizeY, a.projectionsSizeZ, a.pixelSpacingX,
        a.pixelSpacingY, a.volumeSizeX, a.volumeSizeY, a.volumeSizeZ, xpath, a.debug,
        a.itemsPerWorkgroup, a.reportIntermediate, startPath);
    int res = cgls->initializeOpenCL(a.platformId);
    if(res < 0)
    {
        std::string ERR = io::xprintf("Could not initialize OpenCL platform %d.", a.platformId);
        LOGE << ERR;
        io::throwerr(ERR);
    }
    std::vector<float*> volumes;
    float* volume;
    for(std::size_t i = 0; i != a.degree; i++)
    {
        volume = new float[a.totalVolumeSize]();
        volumes.push_back(volume);
    }
    // testing
    //    io::readBytesFrom("/tmp/X.den", 6, (uint8_t*)volume, a.totalVolumeSize * 4);

    cgls->initializeData(projections, basisFunctionsValues, volumes);
    uint16_t buf[3];
    buf[0] = a.volumeSizeY;
    buf[1] = a.volumeSizeX;
    buf[2] = a.volumeSizeZ;
    io::createEmptyFile(a.outputVolume, 0, true);
    io::appendBytes(a.outputVolume, (uint8_t*)buf, 6);
    cgls->reconstruct(dr, a.maxIterations);
    io::appendBytes(a.outputVolume, (uint8_t*)volume, a.totalVolumeSize * sizeof(float));
    delete[] volume;
    delete[] projection;
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
    LOGI << io::xprintf("END %s, duration %d ms.", argv[0], duration.count());
}
