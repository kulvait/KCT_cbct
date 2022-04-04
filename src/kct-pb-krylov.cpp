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

// Reconstructors
#include "CGLSReconstructor.hpp"
#include "GLSQRReconstructor.hpp"
#include "OSSARTReconstructor.hpp"
#include "PSIRTReconstructor.hpp"

// Internal libraries
#include "CArmArguments.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "DEN/DenGeometry3DParallelReader.hpp"
#include "DEN/DenSupportedType.hpp"
#include "GEOMETRY/Geometry3DParallel.hpp"
#include "GEOMETRY/Geometry3DParallelI.hpp"
#include "PROG/ArgumentsForce.hpp"
#include "PROG/KCTException.hpp"
#include "PROG/Program.hpp"

using namespace KCT;
using namespace KCT::util;
using namespace KCT::io;

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
        totalProjectionSize
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
        if(totalProjectionSize > INT_MAX)
        {
            ERR = io::xprintf(
                "Implement indexing by uint64_t matrix dimension overflow of projection "
                "pixels count.");
            LOGE << ERR;
            return -1;
        }
        io::DenSupportedType t = inf.getDataType();
        if(t != io::DenSupportedType::FLOAT32)
        {
            ERR = io::xprintf("This program supports FLOAT32 projections only but the supplied "
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
            if(dataType != DenSupportedType::FLOAT32)
            {
                std::string ERR
                    = io::xprintf("The file %s has declared data type %s but this implementation "
                                  "only supports FLOAT32 files.",
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
            if(dataType != DenSupportedType::FLOAT32)
            {
                ERR = io::xprintf("The file %s has declared data type %s but this implementation "
                                  "only supports FLOAT32.",
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
    uint64_t totalProjectionSize;
    uint64_t totalVolumeSize;
    uint32_t baseOffset = 0;
    double tikhonovLambdaL2 = std::numeric_limits<float>::quiet_NaN();
    double tikhonovLambdaV2 = std::numeric_limits<float>::quiet_NaN();
    double tikhonovLambdaLaplace2D = std::numeric_limits<float>::quiet_NaN();
    bool noFrameOffset = false;
    std::string initialVectorX0;
    std::string outputVolume;
    std::string inputProjectionMatrices;
    std::string inputDetectorTilts;
    std::string inputProjections;
    std::string diagonalPreconditioner;
    bool cgls = false;
    bool glsqr = false;
    bool psirt = false;
    bool sirt = false;
    bool ossart = false;
    bool gradient2D = false;
    bool laplace2D = false;
    uint32_t ossartSubsetCount = 1;
    float lowerBoxCondition = std::numeric_limits<float>::quiet_NaN();
    float upperBoxCondition = std::numeric_limits<float>::quiet_NaN();
    float relaxationParameter = 1.0f;
    bool verbose = true;
    bool disableExpensiveReporting = false;
    bool elevationCorrection = false;
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
    std::string optstr;
    cliApp
        ->add_option("input_projections", inputProjections,
                     "Input projections encoded in FLOAT32 file with the dimensions "
                     "[projection-size-x, projection-size-y, dimz]")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp
        ->add_option("input_projection_matrices", inputProjectionMatrices,
                     "Projection matrices of parallel ray geometry to be input of the computation."
                     "Files in FLOAT64 DEN format that contains projection matricess to process "
                     "with the dimensions [4,2,dimz].")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp->add_option("output_volume", outputVolume, "Output volume")->required();
    optstr = "Detector tilt encoded as positive cosine between normal to the detector and surface "
             "orthogonal to incomming rays. FLOAT64 file of the dimensions (1, 1, dimz). If no "
             "file is supplied, orthogonality of incomming rays to the detector is assumed, that "
             "effectivelly means all these parameters are 1.0.";
    cliApp->add_option("--detectorTilt", inputDetectorTilts, optstr);
    // START reconstruction algorithms
    CLI::Option_group* og_reconstructionAlgorithm = cliApp->add_option_group(
        "Reconstruction algorithm", "Algorithm that will be used for the reconstruction.");
    registerOptionGroup("reconstruction algorithm", og_reconstructionAlgorithm);
    og_reconstructionAlgorithm->require_option(1, 1);
    CLI::Option* cgls_opt = og_reconstructionAlgorithm->add_flag(
        "--cgls", cgls, "Perform CGLS reconstruction (Krylov method).");
    CLI::Option* glsqr_opt = og_reconstructionAlgorithm->add_flag(
        "--glsqr", glsqr, "Perform GLSQR instead reconstruction (Krylov method).");
    CLI::Option* psirt_opt = og_reconstructionAlgorithm->add_flag(
        "--psirt", psirt, "Perform PSIRT reconstruction (non Krylov method).");
    CLI::Option* sirt_opt = og_reconstructionAlgorithm->add_flag(
        "--sirt", sirt, "Perform SIRT reconstruction (non Krylov method).");
    CLI::Option* os_sart_opt = og_reconstructionAlgorithm->add_flag(
        "--os-sart", ossart, "OS SART reconstruction (non Krylov method).");
    addSettingsGroup();
    // STOP reconstruction algorithms
    std::string str = io::xprintf(
        "Tikhonov L2 regularization of volume, NAN to disable, [defaults to %f]", tikhonovLambdaL2);
    CLI::Option* tl2_opt = og_settings->add_option("--tikhonov-lambda-l2", tikhonovLambdaL2, str)
                               ->check(CLI::Range(0.0, 1000000.0));
    str = io::xprintf("Tikhonov V2 regularization of volume, NAN to disable, [defaults to %f]",
                      tikhonovLambdaV2);
    CLI::Option* tv2_opt = og_settings->add_option("--tikhonov-lambda-v2", tikhonovLambdaV2, str)
                               ->check(CLI::Range(0.0, 1000000.0));
    CLI::Option* g2d_opt = og_settings->add_flag("--gradient-2d", gradient2D, "Use 2D gradient.");
    g2d_opt->needs(tv2_opt);
    str = io::xprintf(
        "Tikhonov Laplace regularization of 2D slices of volume, NAN to disable, [defaults to %f]",
        tikhonovLambdaLaplace2D);
    CLI::Option* tlaplace2d_opt
        = og_settings->add_option("--tikhonov-lambda-laplace", tikhonovLambdaLaplace2D, str)
              ->check(CLI::Range(0.0, 1000000.0));
    tl2_opt->excludes(psirt_opt)->excludes(sirt_opt)->excludes(os_sart_opt);
    tv2_opt->excludes(psirt_opt)->excludes(sirt_opt)->excludes(os_sart_opt);
    tlaplace2d_opt->excludes(psirt_opt)->excludes(sirt_opt)->excludes(os_sart_opt);
    CLI::Option* l3d = cliApp->add_flag("--laplace-2d", laplace2D, "2D Laplace operator.");
    l3d->needs(tlaplace2d_opt);
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

    optstr = io::xprintf("Verbose print. [defaults to %s]", verbose ? "true" : "false");
    cliApp->add_flag("--verbose,!--no-verbose", verbose, optstr);
    optstr = io::xprintf("Disable reporting that might slow down the whole computation, e.g. "
                         "reporting of the norm of discrepancy in every itteration of OS method "
                         "that is nice to have for convergence graph but requires additional "
                         "projection per itteration. [defaults to %s]",
                         disableExpensiveReporting ? "true" : "false");
    cliApp->add_flag("--disable-expensive-reporting,!--no-disable-expensive-reporting",
                     disableExpensiveReporting, optstr);
    og_settings->add_option("--x0", initialVectorX0, "Specify x0 vector, zero by default.");
    CLI::Option* dpc = og_settings->add_option(
        "--diagonal-preconditioner", diagonalPreconditioner,
        "Specify diagonal preconditioner vector to be used in preconditioned CGLS.");

    CLI::Option* jacobi_cli = og_settings->add_flag("--jacobi", useJacobiPreconditioning,
                                                    "Use Jacobi preconditioning.");
    jacobi_cli->excludes(glsqr_opt);
    dpc->excludes(jacobi_cli);
    str = io::xprintf("Ordered subset level, number of subsets to be used, 1 for "
                      "classical SART. [defaults to %d]",
                      ossartSubsetCount);
    CLI::Option* ossubsets_opt
        = og_settings->add_option("--os-subset-count", ossartSubsetCount, str);
    str = io::xprintf(
        "Lower box condition, lower numbers will be substituted by this number after "
        "the end of each full iteration, NAN for no lower box condition [defaults to %f]",
        lowerBoxCondition);
    CLI::Option* lowbox_opt
        = og_settings->add_option("--lower-box-condition", lowerBoxCondition, str);
    str = io::xprintf(
        "Upper box condition, higher numbers will be substituted by this number after "
        "the end of each full iteration, NAN for no lower box condition [defaults to %f]",
        upperBoxCondition);
    CLI::Option* upbox_opt
        = og_settings->add_option("--upper-box-condition", upperBoxCondition, str);
    str = io::xprintf("Relaxation parameter for classical algorithms, update step will be "
                      "multiplied by this factor after end of each full iteration [defaults to %f]",
                      relaxationParameter);
    CLI::Option* relaxation_opt
        = og_settings->add_option("--relaxation-parameter", relaxationParameter, str);
    ossubsets_opt->excludes(sirt_opt)->excludes(psirt_opt);
    ossubsets_opt->excludes(cgls_opt)->excludes(glsqr_opt);
    lowbox_opt->excludes(cgls_opt)->excludes(glsqr_opt);
    upbox_opt->excludes(cgls_opt)->excludes(glsqr_opt);
    relaxation_opt->excludes(cgls_opt)->excludes(glsqr_opt)->excludes(psirt_opt);
}

int main(int argc, char* argv[])
{
    try
    {
        Program PRG(argc, argv);
        std::string prgInfo = "OpenCL implementation 3D parallel beam CT reconstruction operator.";
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
        std::shared_ptr<io::DenGeometry3DParallelReader> geometryReader
            = std::make_shared<io::DenGeometry3DParallelReader>(ARG.inputProjectionMatrices,
                                                                ARG.inputDetectorTilts);
        std::vector<std::shared_ptr<geometry::Geometry3DParallelI>> geometryVector;
        std::shared_ptr<geometry::Geometry3DParallelI> geometry;
        for(std::size_t k = 0; k != geometryReader->count(); k++)
        {
            geometry
                = std::make_shared<geometry::Geometry3DParallel>(geometryReader->readGeometry(k));
            geometryVector.emplace_back(geometry);
        }
        float* projection = new float[ARG.totalProjectionSize];
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
        /*TODO
        cl::NDRange projectorLocalNDRange
            = cl::NDRange(ARG.projectorLocalNDRange[0], ARG.projectorLocalNDRange[1],
                          ARG.projectorLocalNDRange[2]);
        cl::NDRange backprojectorLocalNDRange
            = cl::NDRange(ARG.backprojectorLocalNDRange[0], ARG.backprojectorLocalNDRange[1],
                          ARG.backprojectorLocalNDRange[2]);
                if(ARG.cgls)
                {
                    std::shared_ptr<CGLSReconstructor> cgls = std::make_shared<CGLSReconstructor>(
                        ARG.projectionSizeX, ARG.projectionSizeY, ARG.projectionSizeZ,
           ARG.volumeSizeX, ARG.volumeSizeY, ARG.volumeSizeZ, ARG.CLitemsPerWorkgroup,
           projectorLocalNDRange, backprojectorLocalNDRange);
                    cgls->setReportingParameters(ARG.verbose, ARG.reportKthIteration, startPath);
                    // testing
                    //    io::readBytesFrom("/tmp/X.den", 6, (uint8_t*)volume, ARG.totalVolumeSize *
                   4);
                   if(ARG.useSidonProjector)
                   {
                       cgls->initializeSidonProjector(ARG.probesPerEdge, ARG.probesPerEdge);
                   } else if(ARG.useTTProjector)
                   {

                       cgls->initializeTTProjector();
                   } else
                   {
                       cgls->initializeCVPProjector(ARG.useExactScaling, ARG.useElevationCorrection,
                                                    ARG.useBarrierCalls, ARG.barrierArraySize);
                   }
                   if(!std::isnan(ARG.tikhonovLambdaL2) || !std::isnan(ARG.tikhonovLambdaV2)
                      || !std::isnan(ARG.tikhonovLambdaLaplace2D))
                   {
                       cgls->initializeVolumeConvolution();
                       cgls->addTikhonovRegularization(ARG.tikhonovLambdaL2, ARG.tikhonovLambdaV2,
                                                       ARG.tikhonovLambdaLaplace2D);
                       cgls->useGradient3D(!ARG.gradient2D);
                       cgls->useLaplace3D(!ARG.laplace2D);
                   }
                   if(ARG.useJacobiPreconditioning)
                   {
                       cgls->useJacobiVectorCLCode();
                   }
                   int ecd
                       = cgls->initializeOpenCL(ARG.CLplatformID, &ARG.CLdeviceIDs[0],
                                                ARG.CLdeviceIDs.size(), xpath, ARG.CLdebug,
           ARG.CLrelaxed); if(ecd < 0)
                   {
                       std::string ERR
                           = io::xprintf("Could not initialize OpenCL platform %d.",
           ARG.CLplatformID); LOGE << ERR; throw std::runtime_error(ERR);
                   }
                   bool X0initialized = ARG.initialVectorX0 != "";
                   ecd = cgls->problemSetup(projection, volume, X0initialized, cameraVector,
           ARG.voxelSizeX, ARG.voxelSizeY, ARG.voxelSizeZ, ARG.volumeCenterX, ARG.volumeCenterY,
           ARG.volumeCenterZ); if(ecd != 0)
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
                                             (uint8_t*)preconditionerVolume, ARG.totalVolumeSize *
           4); cgls->reconstructDiagonalPreconditioner( preconditionerVolume, ARG.maxIterationCount,
           ARG.stoppingRelativeError); delete[] preconditionerVolume; } else
                       {
                           cgls->reconstruct(ARG.maxIterationCount, ARG.stoppingRelativeError);
                       }
                   }
                   DenFileInfo::createDenHeader(ARG.outputVolume, ARG.volumeSizeX, ARG.volumeSizeY,
                                                ARG.volumeSizeZ);
                   io::appendBytes(ARG.outputVolume, (uint8_t*)volume, ARG.totalVolumeSize *
           sizeof(float)); delete[] volume; delete[] projection; } else if(ARG.glsqr)
                {
                    std::shared_ptr<GLSQRReconstructor> glsqr =
           std::make_shared<GLSQRReconstructor>( ARG.projectionSizeX, ARG.projectionSizeY,
           ARG.projectionSizeZ, ARG.volumeSizeX, ARG.volumeSizeY, ARG.volumeSizeZ,
           ARG.CLitemsPerWorkgroup); glsqr->setReportingParameters(ARG.verbose,
           ARG.reportKthIteration, startPath); if(ARG.useSidonProjector)
                    {
                        glsqr->initializeSidonProjector(ARG.probesPerEdge, ARG.probesPerEdge);
                    } else if(ARG.useTTProjector)
                    {

                        glsqr->initializeTTProjector();
                    } else
                    {
                        glsqr->initializeCVPProjector(ARG.useExactScaling,
           ARG.useElevationCorrection, ARG.useBarrierCalls, ARG.barrierArraySize);
                    }
                    int ecd = glsqr->initializeOpenCL(ARG.CLplatformID, &ARG.CLdeviceIDs[0],
                                                      ARG.CLdeviceIDs.size(), xpath, ARG.CLdebug,
                                                      ARG.CLrelaxed);
                    if(ecd < 0)
                    {
                        std::string ERR
                            = io::xprintf("Could not initialize OpenCL platform %d.",
           ARG.CLplatformID); LOGE << ERR; throw std::runtime_error(ERR);
                    }
                    float* volume = new float[ARG.totalVolumeSize]();
                    // testing
                    //    io::readBytesFrom("/tmp/X.den", 6, (uint8_t*)volume, ARG.totalVolumeSize *
                   4);

                   bool X0initialized = ARG.initialVectorX0 != "";
                   ecd = glsqr->problemSetup(projection, volume, X0initialized, cameraVector,
                                             ARG.voxelSizeX, ARG.voxelSizeY, ARG.voxelSizeZ,
                                             ARG.volumeCenterX, ARG.volumeCenterY,
           ARG.volumeCenterZ); if(ecd != 0)
                   {
                       std::string ERR = io::xprintf("OpenCL buffers initialization failed.");
                       LOGE << ERR;
                       throw std::runtime_error(ERR);
                   }
                   if(!std::isnan(ARG.tikhonovLambdaL2))
                   {
                       glsqr->reconstruct(ARG.maxIterationCount, ARG.stoppingRelativeError);
                   } else
                   {
                       glsqr->reconstructTikhonov(ARG.tikhonovLambdaL2, ARG.maxIterationCount,
                                                  ARG.stoppingRelativeError);
                   }
                   DenFileInfo::createDenHeader(ARG.outputVolume, ARG.volumeSizeX, ARG.volumeSizeY,
                                                ARG.volumeSizeZ);
                   io::appendBytes(ARG.outputVolume, (uint8_t*)volume, ARG.totalVolumeSize *
           sizeof(float)); delete[] volume; delete[] projection; } else if(ARG.psirt)
                {
                    std::shared_ptr<PSIRTReconstructor> psirt =
           std::make_shared<PSIRTReconstructor>( ARG.projectionSizeX, ARG.projectionSizeY,
           ARG.projectionSizeZ, ARG.volumeSizeX, ARG.volumeSizeY, ARG.volumeSizeZ,
           ARG.CLitemsPerWorkgroup); psirt->setReportingParameters(ARG.verbose,
           ARG.reportKthIteration, startPath); if(ARG.useSidonProjector)
                    {
                        psirt->initializeSidonProjector(ARG.probesPerEdge, ARG.probesPerEdge);
                    } else if(ARG.useTTProjector)
                    {

                        psirt->initializeTTProjector();
                    } else
                    {
                        psirt->initializeCVPProjector(ARG.useExactScaling,
           ARG.useElevationCorrection, ARG.useBarrierCalls, ARG.barrierArraySize);
                    }
                    int ecd = psirt->initializeOpenCL(ARG.CLplatformID, &ARG.CLdeviceIDs[0],
                                                      ARG.CLdeviceIDs.size(), xpath, ARG.CLdebug,
                                                      ARG.CLrelaxed);
                    if(ecd < 0)
                    {
                        std::string ERR
                            = io::xprintf("Could not initialize OpenCL platform %d.",
           ARG.CLplatformID); LOGE << ERR; throw std::runtime_error(ERR);
                    }
                    float* volume = new float[ARG.totalVolumeSize]();
                    // testing
                    //    io::readBytesFrom("/tmp/X.den", 6, (uint8_t*)volume, ARG.totalVolumeSize *
                   4);

                   bool X0initialized = ARG.initialVectorX0 != "";
                   ecd = psirt->problemSetup(projection, volume, X0initialized, cameraVector,
                                             ARG.voxelSizeX, ARG.voxelSizeY, ARG.voxelSizeZ,
                                             ARG.volumeCenterX, ARG.volumeCenterY,
           ARG.volumeCenterZ); if(ecd != 0)
                   {
                       std::string ERR = io::xprintf("OpenCL buffers initialization failed.");
                       LOGE << ERR;
                       throw std::runtime_error(ERR);
                   }
                   psirt->setup(1.99); // 10.1109/TMI.2008.923696
                   psirt->reconstruct(ARG.maxIterationCount, ARG.stoppingRelativeError);
                   DenFileInfo::createDenHeader(ARG.outputVolume, ARG.volumeSizeX, ARG.volumeSizeY,
                                                ARG.volumeSizeZ);
                   io::appendBytes(ARG.outputVolume, (uint8_t*)volume, ARG.totalVolumeSize *
           sizeof(float)); delete[] volume; delete[] projection; } else if(ARG.sirt)
                {
                    std::shared_ptr<PSIRTReconstructor> psirt =
           std::make_shared<PSIRTReconstructor>( ARG.projectionSizeX, ARG.projectionSizeY,
           ARG.projectionSizeZ, ARG.volumeSizeX, ARG.volumeSizeY, ARG.volumeSizeZ,
           ARG.CLitemsPerWorkgroup); psirt->setReportingParameters(ARG.verbose,
           ARG.reportKthIteration, startPath); if(ARG.useSidonProjector)
                    {
                        psirt->initializeSidonProjector(ARG.probesPerEdge, ARG.probesPerEdge);
                    } else if(ARG.useTTProjector)
                    {

                        psirt->initializeTTProjector();
                    } else
                    {
                        psirt->initializeCVPProjector(ARG.useExactScaling,
           ARG.useElevationCorrection, ARG.useBarrierCalls, ARG.barrierArraySize);
                    }
                    int ecd = psirt->initializeOpenCL(ARG.CLplatformID, &ARG.CLdeviceIDs[0],
                                                      ARG.CLdeviceIDs.size(), xpath, ARG.CLdebug,
                                                      ARG.CLrelaxed);
                    if(ecd < 0)
                    {
                        std::string ERR
                            = io::xprintf("Could not initialize OpenCL platform %d.",
           ARG.CLplatformID); LOGE << ERR; throw std::runtime_error(ERR);
                    }
                    float* volume = new float[ARG.totalVolumeSize]();
                    // testing
                    //    io::readBytesFrom("/tmp/X.den", 6, (uint8_t*)volume, ARG.totalVolumeSize *
                   4);

                   bool X0initialized = ARG.initialVectorX0 != "";
                   ecd = psirt->problemSetup(projection, volume, X0initialized, cameraVector,
                                             ARG.voxelSizeX, ARG.voxelSizeY, ARG.voxelSizeZ,
                                             ARG.volumeCenterX, ARG.volumeCenterY,
           ARG.volumeCenterZ); if(ecd != 0)
                   {
                       std::string ERR = io::xprintf("OpenCL buffers initialization failed.");
                       LOGE << ERR;
                       throw std::runtime_error(ERR);
                   }
                   psirt->setup(1.00); // 10.1109/TMI.2008.923696
                   psirt->reconstruct_sirt(ARG.maxIterationCount, ARG.stoppingRelativeError);
                   DenFileInfo::createDenHeader(ARG.outputVolume, ARG.volumeSizeX, ARG.volumeSizeY,
                                                ARG.volumeSizeZ);
                   io::appendBytes(ARG.outputVolume, (uint8_t*)volume, ARG.totalVolumeSize *
           sizeof(float)); delete[] volume; delete[] projection; } else if(ARG.ossart)
                {
                    std::shared_ptr<OSSARTReconstructor> ossart =
           std::make_shared<OSSARTReconstructor>( ARG.projectionSizeX, ARG.projectionSizeY,
           ARG.projectionSizeZ, ARG.volumeSizeX, ARG.volumeSizeY, ARG.volumeSizeZ,
           ARG.CLitemsPerWorkgroup); ossart->setup(ARG.relaxationParameter, ARG.ossartSubsetCount);
                    if(!std::isnan(ARG.lowerBoxCondition))
                    {
                        ossart->addLowerBoxCondition(ARG.lowerBoxCondition, ARG.lowerBoxCondition);
                    }
                    if(!std::isnan(ARG.upperBoxCondition))
                    {
                        ossart->addUpperBoxCondition(ARG.upperBoxCondition, ARG.upperBoxCondition);
                    }
                    ossart->setReportingParameters(ARG.verbose, ARG.reportKthIteration, startPath);
                    if(ARG.useSidonProjector)
                    {
                        ossart->initializeSidonProjector(ARG.probesPerEdge, ARG.probesPerEdge);
                    } else if(ARG.useTTProjector)
                    {

                        ossart->initializeTTProjector();
                    } else
                    {
                        ossart->initializeCVPProjector(ARG.useExactScaling,
           ARG.useElevationCorrection, ARG.useBarrierCalls, ARG.barrierArraySize);
                    }
                    int ecd = ossart->initializeOpenCL(ARG.CLplatformID, &ARG.CLdeviceIDs[0],
                                                       ARG.CLdeviceIDs.size(), xpath, ARG.CLdebug,
                                                       ARG.CLrelaxed);

                    if(ecd < 0)
                    {
                        std::string ERR
                            = io::xprintf("Could not initialize OpenCL platform %d.",
           ARG.CLplatformID); LOGE << ERR; throw std::runtime_error(ERR);
                    }
                    float* volume = new float[ARG.totalVolumeSize]();
                    // testing
                    //    io::readBytesFrom("/tmp/X.den", 6, (uint8_t*)volume, ARG.totalVolumeSize *
                   4);

                   bool X0initialized = ARG.initialVectorX0 != "";
                   ecd = ossart->problemSetup(projection, volume, X0initialized, cameraVector,
                                              ARG.voxelSizeX, ARG.voxelSizeY, ARG.voxelSizeZ,
                                              ARG.volumeCenterX, ARG.volumeCenterY,
           ARG.volumeCenterZ); if(ecd != 0)
                   {
                       std::string ERR = io::xprintf("OpenCL buffers initialization failed.");
                       LOGE << ERR;
                       throw std::runtime_error(ERR);
                   }
                   ossart->reconstruct(ARG.maxIterationCount, ARG.stoppingRelativeError);
                   DenFileInfo::createDenHeader(ARG.outputVolume, ARG.volumeSizeX, ARG.volumeSizeY,
                                                ARG.volumeSizeZ);
                   io::appendBytes(ARG.outputVolume, (uint8_t*)volume, ARG.totalVolumeSize *
           sizeof(float)); delete[] volume; delete[] projection;
                }
        */
        PRG.endLog(true);
    } catch(KCTException& ex)
    {
        std::cerr << io::xprintf_red("%s", ex.what()) << std::endl;
        return 1;
    }
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
