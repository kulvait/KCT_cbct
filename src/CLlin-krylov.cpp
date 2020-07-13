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
#include "CArmArguments.hpp"
#include "CGLSReconstructor.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenProjectionMatrixReader.hpp"
#include "DEN/DenSupportedType.hpp"
#include "GLSQRReconstructor.hpp"
#include "PROG/Program.hpp"

using namespace CTL;
using namespace CTL::util;

/**Arguments parsed by the main function.
 */
class Args : public CArmArguments
{
public:
    Args(int argc, char** argv, std::string programName)
        : CArmArguments(argc, argv, programName){};
    int preParse() { return 0; };
    int postParse()
    {
        if(!force)
        {
            if(io::pathExists(outputVolume))
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
        parsePlatformString();
        return 0;
    };
    void defineArguments();
    int threads = 1;
    // It is evaluated from -0.5, pixels are centerred at integer coordinates
    uint64_t totalProjectionsSize;
    uint64_t totalVolumeSize;
    uint32_t baseOffset = 0;
    double tikhonovLambda = -1.0;
    bool noFrameOffset = false;
    std::string outputVolume;
    std::string inputProjectionMatrices;
    std::string inputProjections;
    bool force = false;
    bool glsqr = false;
};

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
    CLI::Option* glsqr_cli = cliApp->add_flag("--glsqr", glsqr, "Perform GLSQR instead of CGLS.");

    CLI::Option* tl_cli = cliApp
                              ->add_option("--tikhonov-lambda", tikhonovLambda,
                                           "Tikhonov regularization parameter.")
                              ->check(CLI::Range(0.0, 5000.0));
    tl_cli->needs(glsqr_cli);

    // Reconstruction geometry
    addVolumeSizeArgs();
    addVoxelSizeArgs();
    addPixelSizeArgs();

    // Program flow parameters
    addSettingsArgs();
    addProjectorArgs();
}

int main(int argc, char* argv[])
{
    Program PRG(argc, argv);
    Args ARG(argc, argv,
             "OpenCL implementation of CGLS and GLSQR applied on the cone beam CT operator.");
    // Argument parsing
    int parseResult = ARG.parse();
    if(parseResult > 0)
    {
        return 0; // Exited sucesfully, help message printed
    } else if(parseResult != 0)
    {
        return -1; // Exited somehow wrong
    }
    PRG.startLog(true);
    std::string xpath = PRG.getRunTimeInfo().getExecutableDirectoryPath();
    std::shared_ptr<io::DenProjectionMatrixReader> dr
        = std::make_shared<io::DenProjectionMatrixReader>(ARG.inputProjectionMatrices);
    float* projection = new float[ARG.totalProjectionsSize];
    io::readBytesFrom(ARG.inputProjections, 6, (uint8_t*)projection, ARG.totalProjectionsSize * 4);
    std::string startPath;
    startPath = io::getParent(ARG.outputVolume);
    std::string bname = io::getBasename(ARG.outputVolume);
    bname = bname.substr(0, bname.find_last_of("."));
    startPath = io::xprintf("%s/%s_", startPath.c_str(), bname.c_str());
    LOGI << io::xprintf("startpath=%s", startPath.c_str());
    bool reportProgress = false;
    if(ARG.reportKthIteration != 0)
    {
        reportProgress = true;
    }
    float* volume;
    if(!ARG.glsqr)
    {
        std::shared_ptr<CGLSReconstructor> cgls = std::make_shared<CGLSReconstructor>(
            ARG.projectionSizeX, ARG.projectionSizeY, ARG.projectionSizeZ, ARG.pixelSizeX,
            ARG.pixelSizeY, ARG.volumeSizeX, ARG.volumeSizeY, ARG.volumeSizeZ, ARG.voxelSizeX,
            ARG.voxelSizeY, ARG.voxelSizeZ, ARG.CLitemsPerWorkgroup);
        cgls->setReportingParameters(reportProgress, startPath, ARG.reportKthIteration);
        int ecd = cgls->initializeOpenCL(ARG.CLplatformID, &ARG.CLdeviceIDs[0], ARG.CLdeviceIDs.size(), xpath, ARG.CLdebug);
        if(ecd < 0)
        {
            std::string ERR
                = io::xprintf("Could not initialize OpenCL platform %d.", ARG.CLplatformID);
            LOGE << ERR;
            throw std::runtime_error(ERR);
        }
        volume = new float[ARG.totalVolumeSize]();
        // testing
        //    io::readBytesFrom("/tmp/X.den", 6, (uint8_t*)volume, ARG.totalVolumeSize * 4);
        if(ARG.useSidonProjector)
        {
            cgls->initializeSidonProjector(ARG.probesPerEdge, ARG.probesPerEdge);
        } else if(ARG.useTTProjector)
        {

            cgls->initializeTTProjector();
        } else
        {
            cgls->initializeCVPProjector(ARG.useExactScaling);
        }

        ecd = cgls->initializeVectors(projection, volume);
        if(ecd != 0)
        {
            std::string ERR = io::xprintf("OpenCL buffers initialization failed.");
            LOGE << ERR;
            throw std::runtime_error(ERR);
        }
        uint16_t buf[3];
        buf[0] = ARG.volumeSizeY;
        buf[1] = ARG.volumeSizeX;
        buf[2] = ARG.volumeSizeZ;
        io::createEmptyFile(ARG.outputVolume, 0, true);
        io::appendBytes(ARG.outputVolume, (uint8_t*)buf, 6);
        cgls->reconstruct(dr, ARG.maxIterationCount, ARG.stoppingRelativeError);
        io::appendBytes(ARG.outputVolume, (uint8_t*)volume, ARG.totalVolumeSize * sizeof(float));
        delete[] volume;
        delete[] projection;
    } else
    {
        std::shared_ptr<GLSQRReconstructor> glsqr = std::make_shared<GLSQRReconstructor>(
            ARG.projectionSizeX, ARG.projectionSizeY, ARG.projectionSizeZ, ARG.pixelSizeX,
            ARG.pixelSizeY, ARG.volumeSizeX, ARG.volumeSizeY, ARG.volumeSizeZ, ARG.voxelSizeX,
            ARG.voxelSizeY, ARG.voxelSizeZ, ARG.CLitemsPerWorkgroup);
        glsqr->setReportingParameters(reportProgress, startPath, ARG.reportKthIteration);
        if(ARG.useSidonProjector)
        {
            glsqr->initializeSidonProjector(ARG.probesPerEdge, ARG.probesPerEdge);
        } else if(ARG.useTTProjector)
        {

            glsqr->initializeTTProjector();
        } else
        {
            glsqr->initializeCVPProjector(ARG.useExactScaling);
        }
        int ecd = glsqr->initializeOpenCL(ARG.CLplatformID, &ARG.CLdeviceIDs[0], ARG.CLdeviceIDs.size(), xpath, ARG.CLdebug);
        if(ecd < 0)
        {
            std::string ERR
                = io::xprintf("Could not initialize OpenCL platform %d.", ARG.CLplatformID);
            LOGE << ERR;
            throw std::runtime_error(ERR);
        }
        float* volume = new float[ARG.totalVolumeSize]();
        // testing
        //    io::readBytesFrom("/tmp/X.den", 6, (uint8_t*)volume, ARG.totalVolumeSize * 4);

        ecd = glsqr->initializeVectors(projection, volume);
        if(ecd != 0)
        {
            std::string ERR = io::xprintf("OpenCL buffers initialization failed.");
            LOGE << ERR;
            throw std::runtime_error(ERR);
        }
        uint16_t buf[3];
        buf[0] = ARG.volumeSizeY;
        buf[1] = ARG.volumeSizeX;
        buf[2] = ARG.volumeSizeZ;
        io::createEmptyFile(ARG.outputVolume, 0, true);
        io::appendBytes(ARG.outputVolume, (uint8_t*)buf, 6);
        if(ARG.tikhonovLambda <= 0.0)
        {
            glsqr->reconstruct(dr, ARG.maxIterationCount, ARG.stoppingRelativeError);
        } else
        {
            glsqr->reconstructTikhonov(dr, ARG.tikhonovLambda, ARG.maxIterationCount,
                                       ARG.stoppingRelativeError);
        }
        io::appendBytes(ARG.outputVolume, (uint8_t*)volume, ARG.totalVolumeSize * sizeof(float));
        delete[] volume;
        delete[] projection;
    }
    PRG.endLog(true);
}
