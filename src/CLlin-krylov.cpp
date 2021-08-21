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
#include "CArmArguments.hpp"
#include "CGLSReconstructor.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "DEN/DenProjectionMatrixReader.hpp"
#include "DEN/DenSupportedType.hpp"
#include "GLSQRReconstructor.hpp"
#include "MATRIX/CameraI.hpp"
#include "PROG/ArgumentsForce.hpp"
#include "PROG/Program.hpp"
#include "PSIRTReconstructor.hpp"

using namespace CTL;
using namespace CTL::util;
using namespace CTL::io;

void populateVoume(float* volume, std::string volumeFile);

/**Arguments parsed by the main function.
 */
class Args : public CArmArguments, public ArgumentsForce
{
public:
    Args(int argc, char** argv, std::string programName)
        : Arguments(argc, argv, programName)
        , CArmArguments(argc, argv, programName)
        , ArgumentsForce(argc, argv, programName){};
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
        std::string ERR;
        if(inf.dimz() != pmi.dimz())
        {
            ERR = io::xprintf(
                "Projection matrices z dimension %d is different from projections z dimension %d.",
                pmi.dimz(), inf.dimz());
            LOGE << ERR;
            return 1;
        }
        if(totalVolumeSize > INT_MAX)
        {
            ERR = "Implement indexing by uint64_t matrix dimension overflow of voxels count.";
            LOGE << ERR;
            return 1;
        }
        // End parsing arguments
        if(totalProjectionsSize > INT_MAX)
        {
            ERR = io::xprintf(
                "Implement indexing by uint64_t matrix dimension overflow of projection "
                "pixels count.");
            LOGE << ERR;
            return -1;
        }
        io::DenSupportedType t = inf.getDataType();
        if(t != io::DenSupportedType::float_)
        {
            ERR = io::xprintf("This program supports float projections only but the supplied "
                              "projection file %s is "
                              "of type %s",
                              inputProjections.c_str(), io::DenSupportedTypeToString(t).c_str());
            LOGE << ERR;
            return -1;
        }
        if(initialVectorX0 != "")
        {
            io::DenFileInfo x0inf(initialVectorX0);
            if(volumeSizeX != x0inf.dimx() || volumeSizeY != x0inf.dimy()
               || volumeSizeZ != x0inf.dimz())
            {

                ERR = io::xprintf("Declared dimensions of volume (%d, %d, %d) and the "
                                  "dimensions of x0 (%d, %d, %d) does not match!",
                                  volumeSizeX, volumeSizeY, volumeSizeZ, x0inf.dimx(), x0inf.dimy(),
                                  x0inf.dimz());
                LOGE << ERR;
                return -1;
            }
            DenSupportedType dataType = x0inf.getDataType();
            if(dataType != DenSupportedType::float_)
            {
                std::string ERR
                    = io::xprintf("The file %s has declared data type %s but this implementation "
                                  "only supports floats!",
                                  initialVectorX0.c_str(), DenSupportedTypeToString(dataType));
                LOGE << ERR;
                return -1;
            }
        }
        if(diagonalPreconditioner != "")
        {
            io::DenFileInfo x0inf(diagonalPreconditioner);
            if(volumeSizeX != x0inf.dimx() || volumeSizeY != x0inf.dimy()
               || volumeSizeZ != x0inf.dimz())
            {

                ERR = io::xprintf("Declared dimensions of volume (%d, %d, %d) and the "
                                  "dimensions of x0 (%d, %d, %d) does not match!",
                                  volumeSizeX, volumeSizeY, volumeSizeZ, x0inf.dimx(), x0inf.dimy(),
                                  x0inf.dimz());
                LOGE << ERR;
                return -1;
            }
            DenSupportedType dataType = x0inf.getDataType();
            if(dataType != DenSupportedType::float_)
            {
                ERR = io::xprintf("The file %s has declared data type %s but this implementation "
                                  "only supports floats!",
                                  diagonalPreconditioner.c_str(),
                                  DenSupportedTypeToString(dataType));
                LOGE << ERR;
                return -1;
            }
        }
        parsePlatformString();
        return 0;
    };
    void defineArguments();
    bool useJacobiPreconditioning = false;
    int threads = 1;
    // It is evaluated from -0.5, pixels are centerred at integer coordinates
    uint64_t totalProjectionsSize;
    uint64_t totalVolumeSize;
    uint32_t baseOffset = 0;
    double tikhonovLambda = -1.0;
    bool noFrameOffset = false;
    std::string initialVectorX0;
    std::string outputVolume;
    std::string inputProjectionMatrices;
    std::string inputProjections;
    std::string diagonalPreconditioner;
    bool glsqr = false;
    bool psirt = false;
    bool sirt = false;
    bool verbose = true;
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
    addForceArgs();
    // Reconstruction geometry
    addVolumeSizeArgs();
    addVolumeCenterArgs();
    addVoxelSizeArgs();
    // addPixelSizeArgs();

    // Program flow parameters
    addSettingsArgs();
    addCLSettingsArgs();
    addProjectorLocalNDRangeArgs();
    addBackprojectorLocalNDRangeArgs();
    addRelaxedArg();
    addProjectorArgs();

    cliApp->add_flag("--verbose,!--no-verbose", verbose, "Verbose print, defaults to true.");
    CLI::Option* glsqr_cli
        = og_settings->add_flag("--glsqr", glsqr, "Perform GLSQR instead of CGLS.");

    CLI::Option* tl_cli = og_settings
                              ->add_option("--tikhonov-lambda", tikhonovLambda,
                                           "Tikhonov regularization parameter.")
                              ->check(CLI::Range(0.0, 5000.0));
    tl_cli->needs(glsqr_cli);
    og_settings->add_flag("--psirt", psirt,
                          "Perform PSIRT instead of CGLS, note its not Krylov method.");
    og_settings->add_flag("--sirt", sirt,
                          "Perform SIRT instead of CGLS, note its not Krylov method.");

    og_settings->add_option("--x0", initialVectorX0, "Specify x0 vector, zero by default.");
    CLI::Option* dpc = og_settings->add_option(
        "--diagonal-preconditioner", diagonalPreconditioner,
        "Specify diagonal preconditioner vector to be used in preconditioned CGLS.");

    CLI::Option* jacobi_cli = og_settings->add_flag("--jacobi", useJacobiPreconditioning,
                                                    "Use Jacobi preconditioning.");
    jacobi_cli->excludes(glsqr_cli);
    dpc->excludes(jacobi_cli);
}

int main(int argc, char* argv[])
{
    Program PRG(argc, argv);
    std::string prgInfo
        = "OpenCL implementation of CGLS and GLSQR applied on the cone beam CT operator.";
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
    std::string xpath = PRG.getRunTimeInfo().getExecutableDirectoryPath();
    std::shared_ptr<io::DenProjectionMatrixReader> projectionMatrixReader
        = std::make_shared<io::DenProjectionMatrixReader>(ARG.inputProjectionMatrices);
    std::vector<std::shared_ptr<matrix::CameraI>> cameraVector;
    std::shared_ptr<matrix::CameraI> pm;
    for(std::size_t k = 0; k != projectionMatrixReader->count(); k++)
    {
        pm = std::make_shared<matrix::LightProjectionMatrix>(projectionMatrixReader->readMatrix(k));
        cameraVector.emplace_back(pm);
    }
    float* projection = new float[ARG.totalProjectionsSize];
    io::DenFrame2DReader<float> inputProjectionsReader(ARG.inputProjections);
    uint64_t projectionFrameSize = ARG.projectionSizeX * ARG.projectionSizeY;
    for(uint32_t z = 0; z != ARG.projectionSizeZ; z++)
    {
        inputProjectionsReader.readFrameIntoBuffer(z, projection + (z * projectionFrameSize),
                                                   false);
    }
    float* volume;
    if(ARG.initialVectorX0 != "")
    {
        volume = new float[ARG.totalVolumeSize];
        io::DenFileInfo iv(ARG.initialVectorX0);
        io::readBytesFrom(ARG.initialVectorX0, iv.getOffset(), (uint8_t*)volume,
                          ARG.totalVolumeSize * 4);
    } else
    {
        volume = new float[ARG.totalVolumeSize]();
    }
    std::string startPath;
    startPath = io::getParent(ARG.outputVolume);
    std::string bname = io::getBasename(ARG.outputVolume);
    bname = bname.substr(0, bname.find_last_of("."));
    startPath = io::xprintf("%s/%s_", startPath.c_str(), bname.c_str());
    LOGI << io::xprintf("startpath=%s", startPath.c_str());

    if(!ARG.glsqr && !ARG.psirt && !ARG.sirt)
    {
        cl::NDRange projectorLocalNDRange
            = cl::NDRange(ARG.projectorLocalNDRange[0], ARG.projectorLocalNDRange[1],
                          ARG.projectorLocalNDRange[2]);
        cl::NDRange backprojectorLocalNDRange
            = cl::NDRange(ARG.backprojectorLocalNDRange[0], ARG.backprojectorLocalNDRange[1],
                          ARG.backprojectorLocalNDRange[2]);
        std::shared_ptr<CGLSReconstructor> cgls = std::make_shared<CGLSReconstructor>(
            ARG.projectionSizeX, ARG.projectionSizeY, ARG.projectionSizeZ, ARG.volumeSizeX,
            ARG.volumeSizeY, ARG.volumeSizeZ, ARG.CLitemsPerWorkgroup, projectorLocalNDRange,
            backprojectorLocalNDRange);
        cgls->setReportingParameters(ARG.verbose, ARG.reportKthIteration, startPath);
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
            cgls->initializeCVPProjector(ARG.useExactScaling, ARG.useBarrierCalls,
                                         ARG.barrierArraySize);
        }
        if(ARG.useJacobiPreconditioning)
        {
            cgls->useJacobiVectorCLCode();
        }
        int ecd = cgls->initializeOpenCL(ARG.CLplatformID, &ARG.CLdeviceIDs[0],
                                         ARG.CLdeviceIDs.size(), xpath, ARG.CLdebug, ARG.CLrelaxed);
        if(ecd < 0)
        {
            std::string ERR
                = io::xprintf("Could not initialize OpenCL platform %d.", ARG.CLplatformID);
            LOGE << ERR;
            throw std::runtime_error(ERR);
        }
        bool X0initialized = ARG.initialVectorX0 != "";
        ecd = cgls->problemSetup(projection, volume, X0initialized, cameraVector, ARG.voxelSizeX,
                                 ARG.voxelSizeY, ARG.voxelSizeZ, ARG.volumeCenterX,
                                 ARG.volumeCenterY, ARG.volumeCenterZ);
        if(ecd != 0)
        {
            std::string ERR = io::xprintf("OpenCL buffers initialization failed.");
            LOGE << ERR;
            throw std::runtime_error(ERR);
        }
        if(ARG.useJacobiPreconditioning)
        {
            cgls->reconstructJacobi(ARG.maxIterationCount, ARG.stoppingRelativeError);
        } else
        {
            if(ARG.diagonalPreconditioner != "")
            {
                float* preconditionerVolume = new float[ARG.totalVolumeSize];
                io::DenFileInfo dpInfo(ARG.diagonalPreconditioner);
                io::readBytesFrom(ARG.diagonalPreconditioner, dpInfo.getOffset(),
                                  (uint8_t*)preconditionerVolume, ARG.totalVolumeSize * 4);
                cgls->reconstructDiagonalPreconditioner(preconditionerVolume, ARG.maxIterationCount,
                                                        ARG.stoppingRelativeError);
                delete[] preconditionerVolume;
            } else
            {
                cgls->reconstruct(ARG.maxIterationCount, ARG.stoppingRelativeError);
            }
        }
        DenFileInfo::createDenHeader(ARG.outputVolume, ARG.volumeSizeX, ARG.volumeSizeY,
                                     ARG.volumeSizeZ);
        io::appendBytes(ARG.outputVolume, (uint8_t*)volume, ARG.totalVolumeSize * sizeof(float));
        delete[] volume;
        delete[] projection;
    } else if(ARG.glsqr)
    {
        std::shared_ptr<GLSQRReconstructor> glsqr = std::make_shared<GLSQRReconstructor>(
            ARG.projectionSizeX, ARG.projectionSizeY, ARG.projectionSizeZ, ARG.volumeSizeX,
            ARG.volumeSizeY, ARG.volumeSizeZ, ARG.CLitemsPerWorkgroup);
        glsqr->setReportingParameters(ARG.verbose, ARG.reportKthIteration, startPath);
        if(ARG.useSidonProjector)
        {
            glsqr->initializeSidonProjector(ARG.probesPerEdge, ARG.probesPerEdge);
        } else if(ARG.useTTProjector)
        {

            glsqr->initializeTTProjector();
        } else
        {
            glsqr->initializeCVPProjector(ARG.useExactScaling, ARG.useBarrierCalls,
                                          ARG.barrierArraySize);
        }
        int ecd
            = glsqr->initializeOpenCL(ARG.CLplatformID, &ARG.CLdeviceIDs[0], ARG.CLdeviceIDs.size(),
                                      xpath, ARG.CLdebug, ARG.CLrelaxed);
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

        bool X0initialized = ARG.initialVectorX0 != "";
        ecd = glsqr->problemSetup(projection, volume, X0initialized, cameraVector, ARG.voxelSizeX,
                                  ARG.voxelSizeY, ARG.voxelSizeZ, ARG.volumeCenterX,
                                  ARG.volumeCenterY, ARG.volumeCenterZ);
        if(ecd != 0)
        {
            std::string ERR = io::xprintf("OpenCL buffers initialization failed.");
            LOGE << ERR;
            throw std::runtime_error(ERR);
        }
        if(ARG.tikhonovLambda <= 0.0)
        {
            glsqr->reconstruct(ARG.maxIterationCount, ARG.stoppingRelativeError);
        } else
        {
            glsqr->reconstructTikhonov(ARG.tikhonovLambda, ARG.maxIterationCount,
                                       ARG.stoppingRelativeError);
        }
        DenFileInfo::createDenHeader(ARG.outputVolume, ARG.volumeSizeX, ARG.volumeSizeY,
                                     ARG.volumeSizeZ);
        io::appendBytes(ARG.outputVolume, (uint8_t*)volume, ARG.totalVolumeSize * sizeof(float));
        delete[] volume;
        delete[] projection;
    } else if(ARG.psirt)
    {
        std::shared_ptr<PSIRTReconstructor> psirt = std::make_shared<PSIRTReconstructor>(
            ARG.projectionSizeX, ARG.projectionSizeY, ARG.projectionSizeZ, ARG.volumeSizeX,
            ARG.volumeSizeY, ARG.volumeSizeZ, ARG.CLitemsPerWorkgroup);
        psirt->setReportingParameters(ARG.verbose, ARG.reportKthIteration, startPath);
        if(ARG.useSidonProjector)
        {
            psirt->initializeSidonProjector(ARG.probesPerEdge, ARG.probesPerEdge);
        } else if(ARG.useTTProjector)
        {

            psirt->initializeTTProjector();
        } else
        {
            psirt->initializeCVPProjector(ARG.useExactScaling, ARG.useBarrierCalls,
                                          ARG.barrierArraySize);
        }
        int ecd
            = psirt->initializeOpenCL(ARG.CLplatformID, &ARG.CLdeviceIDs[0], ARG.CLdeviceIDs.size(),
                                      xpath, ARG.CLdebug, ARG.CLrelaxed);
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

        bool X0initialized = ARG.initialVectorX0 != "";
        ecd = psirt->problemSetup(projection, volume, X0initialized, cameraVector, ARG.voxelSizeX,
                                  ARG.voxelSizeY, ARG.voxelSizeZ, ARG.volumeCenterX,
                                  ARG.volumeCenterY, ARG.volumeCenterZ);
        if(ecd != 0)
        {
            std::string ERR = io::xprintf("OpenCL buffers initialization failed.");
            LOGE << ERR;
            throw std::runtime_error(ERR);
        }
        psirt->setup(1.99); // 10.1109/TMI.2008.923696
        if(ARG.tikhonovLambda <= 0.0)
        {
            psirt->reconstruct(ARG.maxIterationCount, ARG.stoppingRelativeError);
        } else
        {
            throw std::runtime_error("Tikhonov stabilization is not implemented for PSIRT!");
        }
        DenFileInfo::createDenHeader(ARG.outputVolume, ARG.volumeSizeX, ARG.volumeSizeY,
                                     ARG.volumeSizeZ);
        io::appendBytes(ARG.outputVolume, (uint8_t*)volume, ARG.totalVolumeSize * sizeof(float));
        delete[] volume;
        delete[] projection;
    } else if(ARG.sirt)
    {
        std::shared_ptr<PSIRTReconstructor> psirt = std::make_shared<PSIRTReconstructor>(
            ARG.projectionSizeX, ARG.projectionSizeY, ARG.projectionSizeZ, ARG.volumeSizeX,
            ARG.volumeSizeY, ARG.volumeSizeZ, ARG.CLitemsPerWorkgroup);
        psirt->setReportingParameters(ARG.verbose, ARG.reportKthIteration, startPath);
        if(ARG.useSidonProjector)
        {
            psirt->initializeSidonProjector(ARG.probesPerEdge, ARG.probesPerEdge);
        } else if(ARG.useTTProjector)
        {

            psirt->initializeTTProjector();
        } else
        {
            psirt->initializeCVPProjector(ARG.useExactScaling, ARG.useBarrierCalls,
                                          ARG.barrierArraySize);
        }
        int ecd
            = psirt->initializeOpenCL(ARG.CLplatformID, &ARG.CLdeviceIDs[0], ARG.CLdeviceIDs.size(),
                                      xpath, ARG.CLdebug, ARG.CLrelaxed);
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

        bool X0initialized = ARG.initialVectorX0 != "";
        ecd = psirt->problemSetup(projection, volume, X0initialized, cameraVector, ARG.voxelSizeX,
                                  ARG.voxelSizeY, ARG.voxelSizeZ, ARG.volumeCenterX,
                                  ARG.volumeCenterY, ARG.volumeCenterZ);
        if(ecd != 0)
        {
            std::string ERR = io::xprintf("OpenCL buffers initialization failed.");
            LOGE << ERR;
            throw std::runtime_error(ERR);
        }
        psirt->setup(1.00); // 10.1109/TMI.2008.923696
        if(ARG.tikhonovLambda <= 0.0)
        {
            psirt->reconstruct_sirt(ARG.maxIterationCount, ARG.stoppingRelativeError);
        } else
        {
            throw std::runtime_error("Tikhonov stabilization is not implemented for PSIRT!");
        }
        DenFileInfo::createDenHeader(ARG.outputVolume, ARG.volumeSizeX, ARG.volumeSizeY,
                                     ARG.volumeSizeZ);
        io::appendBytes(ARG.outputVolume, (uint8_t*)volume, ARG.totalVolumeSize * sizeof(float));
        delete[] volume;
        delete[] projection;
    }
    PRG.endLog(true);
}

void populateVoume(float* volume, std::string volumeFile)
{

    DenFileInfo fileInf(volumeFile);
    DenSupportedType dataType = fileInf.getDataType();
    uint64_t offset = fileInf.getOffset();
    uint64_t elementByteSize = fileInf.elementByteSize();
    uint64_t position;
    uint64_t frameSize = fileInf.dimx() * fileInf.dimy();
    uint8_t* buffer = new uint8_t[elementByteSize * frameSize];
    for(uint64_t frameID = 0; frameID != fileInf.dimz(); frameID++)
    {
        position = offset + uint64_t(frameID) * elementByteSize * frameSize;
        io::readBytesFrom(volumeFile, position, buffer, elementByteSize * frameSize);
        for(uint64_t pos = 0; pos != frameSize; pos++)
        {
            volume[frameID * frameSize + pos]
                = util::getNextElement<float>(buffer + pos * elementByteSize, dataType);
        }
    }
    delete[] buffer;
}
