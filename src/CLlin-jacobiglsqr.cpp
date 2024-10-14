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
#include "gitversion/version.h"

// Internal libraries
#include "CGLSReconstructor.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenProjectionMatrixReader.hpp"
#include "DEN/DenSupportedType.hpp"
#include "JacobiGLSQRReconstructor.hpp"
#include "PROG/ArgumentsForce.hpp"
#include "PROG/Program.hpp"
#include "PROG/parseArgs.h"

using namespace KCT;
using namespace KCT::util;

/**Arguments parsed by the main function.
 */
class Args : public ArgumentsForce
{
public:
    Args(int argc, char** argv, std::string programName)
        : Arguments(argc, argv, programName)
        , ArgumentsForce(argc, argv, programName){};
    int preParse() { return 0; };
    int postParse();
    void defineArguments();

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
    double stoppingResidualError = 0.01;
    bool noFrameOffset = false;
    std::string outputVolume;
    std::string inputProjectionMatrices;
    std::string inputProjections;
    uint32_t itemsPerWorkgroup = 256;
    bool glsqr = false;
};

int Args::postParse()
{
    std::string ERR;
    int e = handleFileExistence(outputVolume, force, force);
    if(e != 0)
    {
        return e;
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
        ERR = io::xprintf(
            "Projection matrices z dimension %d is different from projections z dimension %d.",
            pmi.dimz(), inf.dimz());
        LOGE << ERR;
        return -1;
    }
    if(totalVolumeSize > INT_MAX)
    {
        ERR = "Implement indexing by uint64_t matrix dimension overflow of voxels count.";
        LOGE << ERR;
        return -1;
    }
    // End parsing arguments
    if(totalProjectionsSize > INT_MAX)
    {
        ERR = "Implement indexing by uint64_t matrix dimension overflow of projection "
              "pixels count.";
        LOGE << ERR;
        return -1;
    }
    io::DenSupportedType t = inf.getElementType();
    if(t != io::DenSupportedType::FLOAT32)
    {
        std::string ERR
            = io::xprintf("This program supports FLOAT32 projections only but the supplied "
                          "projection file %s is "
                          "of type %s",
                          inputProjections.c_str(), io::DenSupportedTypeToString(t).c_str());
        LOGE << ERR;
        return -1;
    }
    return 0;
}

/**Argument parsing
 *
 * @param argc
 * @param argv[]
 *
 * @return Returns 0 on success and nonzero for some error.
 */
void Args::defineArguments()
{

    cliApp->add_option("input_projections", inputProjections, "Input projections")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp
        ->add_option("input_projection_matrices", inputProjectionMatrices,
                     "Projection matrices to be input of the computation."
                     "Files in a DEN format that contains projection matricess to process.")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp->add_option("output_volume", outputVolume, "Volume to project")->required();
    cliApp->add_flag("-f,--force", force, "Overwrite outputProjection if it exists.");
    CLI::Option* psx = cliApp->add_option("--pixel-sizex", pixelSizeX,
                                          "Spacing of detector cells, defaults to 0.616.");
    CLI::Option* psy = cliApp->add_option("--pixel-sizey", pixelSizeY,
                                          "Spacing of detector cells, defaults to 0.616.");
    CLI::Option* vx = cliApp->add_option("--volume-sizex", volumeSizeX,
                                         "Dimension of volume, defaults to 256.");
    CLI::Option* vy = cliApp->add_option("--volume-sizey", volumeSizeY,
                                         "Dimension of volume, defaults to 256.");
    CLI::Option* vz = cliApp->add_option("--volume-sizez", volumeSizeZ,
                                         "Dimension of volume, defaults to 199.");
    cliApp->add_option("--voxel-sizex", voxelSizeX, "Spacing of voxels, defaults to 1.0.");
    cliApp->add_option("--voxel-sizey", voxelSizeY, "Spacing of voxels, defaults to 1.0.");
    cliApp->add_option("--voxel-sizez", voxelSizeZ, "Spacing of voxels, defaults to 1.0.");

    // Program flow parameters
    cliApp->add_option("-j,--threads", threads, "Number of extra threads that application can use.")
        ->check(CLI::Range(0, 65535))
        ->group("Platform settings");
    cliApp->add_option("-p,--platform_id", platformId, "OpenCL platform ID to use.")
        ->check(CLI::Range(0, 65535))
        ->group("Platform settings");
    cliApp->add_flag("-d,--debug", debug, "OpenCL compilation including debugging information.")
        ->group("Platform settings");
    cliApp
        ->add_option("--items-per-workgroup", itemsPerWorkgroup,
                     "OpenCL parameter that is important for norm computation, defaults to 256.")
        ->check(CLI::Range(1, 65535))
        ->group("Platform settings");
    cliApp
        ->add_flag("--report-intermediate", reportIntermediate,
                   "Report intermediate values of x, defaults to false.")
        ->group("Platform settings");
    cliApp
        ->add_option("-i,--max_iterations", maxIterations,
                     "Maximum number of CGLS iterations, defaults to 10.")
        ->check(CLI::Range(1, 65535))
        ->group("Platform settings");
    cliApp->add_option("-e", stoppingResidualError, "Stopping error, defaults to 0.01.")
        ->check(CLI::Range(0.0, 1.00))
        ->group("Platform settings");
    psx->needs(psy);
    psy->needs(psx);
    vx->needs(vy)->needs(vz);
    vy->needs(vx)->needs(vz);
    vz->needs(vx)->needs(vy);
}

int main(int argc, char* argv[])
{
    CLI::App app{};
    Program PRG(argc, argv);
    std::string prgInfo
        = "OpenCL implementation of Jacobi preconditioned CGLSQR for C-Arm CT reconstruction.";
    if(version::MODIFIED_SINCE_COMMIT == true)
    {
        prgInfo = io::xprintf("%s Dirty commit %s", prgInfo.c_str(), version::GIT_COMMIT_ID);
    } else
    {
        prgInfo = io::xprintf("%s Git commit %s", prgInfo.c_str(), version::GIT_COMMIT_ID);
    }
    Args ARG(argc, argv, prgInfo);
    // Argument parsing
    int parseResult = ARG.parse();
    if(parseResult > 0)
    {
        return 0; // Exited sucesfully, help message printed
    } else if(parseResult != 0)
    {
        return -1; // Exited somehow wrong
    }
    LOGI << prgInfo;
    PRG.startLog(true);
    std::shared_ptr<io::DenProjectionMatrixReader> dr
        = std::make_shared<io::DenProjectionMatrixReader>(ARG.inputProjectionMatrices);
    float* projection = new float[ARG.totalProjectionsSize];
    io::DenFileInfo inputProjectionInfo(ARG.inputProjections);
    bool readxmajorprojection = false;
    inputProjectionInfo.readIntoArray<float>(projection, readxmajorprojection);
    std::string startPath = io::getParent(ARG.outputVolume);
    std::string bname = io::getBasename(ARG.outputVolume);
    bname = bname.substr(0, bname.find_last_of("."));
    startPath = io::xprintf("%s/%s_", startPath.c_str(), bname.c_str());
    LOGI << io::xprintf("startpath=%s", startPath.c_str());
    RunTimeInfo rti = PRG.getRunTimeInfo();
    std::string xpath = rti.getExecutablePath();
    std::shared_ptr<GLSQRReconstructor> glsqr = std::make_shared<GLSQRReconstructor>(
        ARG.projectionSizeX, ARG.projectionSizeY, ARG.projectionSizeZ, ARG.pixelSizeX,
        ARG.pixelSizeY, ARG.volumeSizeX, ARG.volumeSizeY, ARG.volumeSizeZ, ARG.voxelSizeX,
        ARG.voxelSizeY, ARG.voxelSizeZ, xpath, ARG.debug, ARG.itemsPerWorkgroup,
        ARG.reportIntermediate, startPath);
    int res = glsqr->initializeOpenCL(ARG.platformId);
    if(res < 0)
    {
        std::string ERR = io::xprintf("Could not initialize OpenCL platform %d.", ARG.platformId);
        KCTERR(ERR);
    }
    float* volume = new float[ARG.totalVolumeSize]();
    glsqr->initializeVectors(projection, volume);
    glsqr->reconstruct(dr, ARG.maxIterations, ARG.stoppingResidualError);
    bool volumexmajor = true;
    bool writexmajor = true;
    io::DenFileInfo::create3DDenFileFromArray(volume, volumexmajor, ARG.outputVolume,
                                              io::DenSupportedType::FLOAT32, ARG.volumeSizeX,
                                              ARG.volumeSizeY, ARG.volumeSizeZ, writexmajor);
    delete[] volume;
    delete[] projection;
    PRG.endLog(true);
}
