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
#include "CGLSPBCT2DReconstructor.hpp"

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
    int postParse();
    void defineArguments();
    bool useJacobiPreconditioning = false;
    bool useSumPreconditioning = false;
    int threads = 1;
    // It is evaluated from -0.5, pixels are centerred at integer coordinates
    uint64_t totalProjectionSize;
    uint64_t projectionFrameSize;
    uint64_t totalVolumeSize;
    uint32_t slabFrom = 0;
    uint32_t slabSize = 0;
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
    addSettingsGroup();
    // STOP reconstruction algorithms
    addForceArgs();
    // Reconstruction geometry
    bool includeVolumeSizez = false;
    addVolumeSizeArgs(includeVolumeSizez);
    og_geometry->add_option(
        "--slab-from", slabFrom,
        "Use for slab reconstruction, reconstruct only part of the projection data.");
    og_geometry->add_option("--slab-size", slabSize,
                            "Use for slab reconstruction, reconstruct only part of the projection "
                            "data, 0 means reconstruct up to dimy.");
    addVolumeCenterArgs();
    addVoxelSizeArgs();

    // Program flow parameters
    addSettingsArgs();
    addCLSettingsArgs();
    addProjectorLocalNDRangeArgs();
    addBackprojectorLocalNDRangeArgs();
    addRelaxedArg();
    addProjectorArgs();
    addBackprojectorScalingArgs();

    optstr = io::xprintf("Verbose print. [defaults to %s]", verbose ? "true" : "false");
    cliApp->add_flag("--verbose,!--no-verbose", verbose, optstr);
    optstr = io::xprintf("Disable reporting that might slow down the whole computation, e.g. "
                         "reporting of the norm of discrepancy in every itteration of OS method "
                         "that is nice to have for convergence graph but requires additional "
                         "projection per itteration. [defaults to %s]",
                         disableExpensiveReporting ? "true" : "false");
    cliApp->add_flag("--disable-expensive-reporting,!--no-disable-expensive-reporting",
                     disableExpensiveReporting, optstr);
}

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
    if(pmi.dimz() != inf.dimz())
    {
        ERR = io::xprintf("Incompatible number of %d projections with %d projection matrices",
                          inf.dimz(), pmi.dimz());
        LOGE << ERR;
        return -1;
    }
    projectionSizeX = inf.dimx();
    projectionSizeY = inf.dimy();
    projectionSizeZ = inf.dimz();
    if(slabFrom >= projectionSizeY)
    {
        ERR = io::xprintf("Inconsistent slab spec!");
        LOGE << ERR;
        return -1;
    }
    if(slabSize == 0)
    {
        slabSize = projectionSizeY - slabFrom;
    }
    uint32_t slabTo = slabFrom + slabSize;
    if(slabTo > projectionSizeY)
    {
        ERR = io::xprintf("Inconsistent slab spec!");
        LOGE << ERR;
        return -1;
    }
    totalVolumeSize = uint64_t(volumeSizeX) * uint64_t(volumeSizeY) * uint64_t(slabSize);
    projectionFrameSize = static_cast<uint64_t>(projectionSizeX) * static_cast<uint64_t>(slabSize);
    totalProjectionSize
        = uint64_t(projectionSizeX) * uint64_t(slabSize) * uint64_t(projectionSizeZ);
    LOGD << io::xprintf("Projection (x,y,z) = (%d, %d, %d), totalProjectionSize=%lu",
                        projectionSizeX, projectionSizeY, projectionSizeZ, totalProjectionSize);
    if(inf.dimz() != pmi.dimz())
    {
        ERR = io::xprintf(
            "Projection matrices count %d is different from number of projections %d.", pmi.dimz(),
            inf.dimz());
        LOGE << ERR;
        return -1;
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
        ERR = io::xprintf("Implement indexing by uint64_t matrix dimension overflow of projection "
                          "pixels count.");
        LOGE << ERR;
        return -1;
    }
    io::DenSupportedType t = inf.getElementType();
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
        DenSupportedType dataType = x0inf.getElementType();
        if(dataType != DenSupportedType::FLOAT32)
        {
            std::string ERR
                = io::xprintf("The file %s has declared data type %s but this implementation "
                              "only supports FLOAT32 files.",
                              initialVectorX0.c_str(), DenSupportedTypeToString(dataType).c_str());
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
        DenSupportedType dataType = x0inf.getElementType();
        if(dataType != DenSupportedType::FLOAT32)
        {
            ERR = io::xprintf("The file %s has declared data type %s but this implementation "
                              "only supports FLOAT32.",
                              diagonalPreconditioner.c_str(),
                              DenSupportedTypeToString(dataType).c_str());
            LOGE << ERR;
            return -1;
        }
    }
    bool barrierAdjustSize = false;
    if(useBarrierCalls && barrierArraySize == -1)
    {
        barrierAdjustSize = true;
    }
    uint64_t localMemSize = parsePlatformString();
    if(barrierAdjustSize)
    {
        barrierArraySize = localMemSize / 4 - 16; // 9 shall be sufficient but for memory alignment
        LOGI << io::xprintf("Setting LOCALARRAYSIZE=%d for optimal performance.", barrierArraySize);
    }
    if(useBarrierCalls && barrierArraySize > localMemSize / 4 - 9)
    {
        ERR = io::xprintf("Array of size %d can not be allocated on given device, maximum is %d!",
                          barrierArraySize, localMemSize / 4 - 9);
    }
    return 0;
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
        bool helpOnError = false;
        int parseResult = ARG.parse(helpOnError);
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
        cl::NDRange projectorLocalNDRange
            = cl::NDRange(ARG.projectorLocalNDRange[0], ARG.projectorLocalNDRange[1],
                          ARG.projectorLocalNDRange[2]);
        cl::NDRange backprojectorLocalNDRange
            = cl::NDRange(ARG.backprojectorLocalNDRange[0], ARG.backprojectorLocalNDRange[1],
                          ARG.backprojectorLocalNDRange[2]);
        float* projection = new float[ARG.totalProjectionSize];
        io::DenFileInfo inputProjectionInfo(ARG.inputProjections);
        bool readxmajor = false;
        inputProjectionInfo.readIntoArray<float>(projection, readxmajor, 0, 0, ARG.slabFrom,
                                                 ARG.slabSize);
        float* volume = new float[ARG.totalVolumeSize];
        std::string startPath;
        startPath = io::getParent(ARG.outputVolume);
        std::string bname = io::getBasename(ARG.outputVolume);
        bname = bname.substr(0, bname.find_last_of("."));
        startPath = io::xprintf("%s/%s_", startPath.c_str(), bname.c_str());
        LOGI << io::xprintf("startpath=%s", startPath.c_str());
        std::shared_ptr<CGLSPBCT2DReconstructor> cgls = std::make_shared<CGLSPBCT2DReconstructor>(
            ARG.projectionSizeX, ARG.slabSize, ARG.projectionSizeZ, ARG.volumeSizeX,
            ARG.volumeSizeY, ARG.slabSize, ARG.CLitemsPerWorkgroup);
        cgls->setReportingParameters(ARG.verbose, ARG.reportKthIteration, startPath);
        // testing
        //    io::readBytesFrom("/tmp/X.den", 6, (uint8_t*)volume, ARG.totalVolumeSize *
        //       4);
        if(ARG.useSidonProjector)
        {
            cgls->initializeSidonProjector(ARG.probesPerEdge, ARG.probesPerEdge);
        } else if(ARG.useTTProjector)
        {

            cgls->initializeTTProjector();
        } else
        {
            cgls->initializeCVPProjector(ARG.useBarrierCalls, ARG.barrierArraySize);
        }
        int ecd = cgls->initializeOpenCL(ARG.CLplatformID, &ARG.CLdeviceIDs[0],
                                         ARG.CLdeviceIDs.size(), xpath, ARG.CLdebug, ARG.CLrelaxed,
                                         projectorLocalNDRange, backprojectorLocalNDRange);
        if(ecd < 0)
        {
            std::string ERR
                = io::xprintf("Could not initialize OpenCL platform %d.", ARG.CLplatformID);
            LOGE << ERR;
            KCTERR(ERR);
        }
        bool X0initialized = false;
        float geometryAtY = ARG.voxelSizeY
            * (static_cast<float>(ARG.slabFrom) + 0.5f * static_cast<float>(ARG.slabSize));
        cgls->problemSetup(geometryVector, geometryAtY, ARG.voxelSizeX, ARG.voxelSizeY,
                           ARG.voxelSizeZ, ARG.volumeCenterX, ARG.volumeCenterY, ARG.volumeCenterZ);
        ecd = cgls->initializeVectors(projection, volume, X0initialized);
        if(ecd != 0)
        {
            std::string ERR = io::xprintf("OpenCL buffers initialization failed.");
            LOGE << ERR;
            KCTERR(ERR);
        }
        cgls->simpleBackprojection(ARG.backprojectorNaturalScaling);
        bool volumexmajor = true;
        bool writexmajor = true;
        io::DenFileInfo::create3DDenFileFromArray(volume, volumexmajor, ARG.outputVolume,
                                                  io::DenSupportedType::FLOAT32, ARG.volumeSizeX,
                                                  ARG.volumeSizeY, ARG.slabSize, writexmajor);
        delete[] volume;
        delete[] projection;
        PRG.endLog(true);
    } catch(KCTException& ex)
    {
        std::cerr << io::xprintf_red("%s", ex.what()) << std::endl;
        return 1;
    }
}
