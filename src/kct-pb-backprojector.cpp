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
#include "gitversion/version.h"

// Internal libraries
#include "CArmArguments.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "DEN/DenGeometry3DParallelReader.hpp"
#include "DEN/DenProjectionMatrixReader.hpp"
#include "DEN/DenSupportedType.hpp"
#include "MATRIX/LightProjectionMatrix.hpp"
#include "PROG/ArgumentsForce.hpp"
#include "PROG/ArgumentsFramespec.hpp"
#include "PROG/Program.hpp"
#include "ParallelBeamProjector.hpp"
#include "SMA/BufferedSparseMatrixFloatWritter.hpp"

using namespace KCT;
using namespace KCT::util;

/**Arguments parsed by the main function.
 */
class Args : public CArmArguments, public ArgumentsFramespec, public ArgumentsForce
{
public:
    Args(int argc, char** argv, std::string programName)
        : Arguments(argc, argv, programName)
        , CArmArguments(argc, argv, programName)
        , ArgumentsFramespec(argc, argv, programName)
        , ArgumentsForce(argc, argv, programName){};
    int preParse() { return 0; };
    int postParse()
    {
        std::string ERR;
        int e = handleFileExistence(outputVolume, force, force);
        if(e != 0)
        {
            return e;
        }
        // How many projection matrices is there in total
        io::DenFileInfo pmi(inputProjectionMatrices);
        io::DenFileInfo inf(inputProjection);
        // Number of projections equals the size of the frames vector
        fillFramesVector(pmi.dimz());
        projectionSizeX = inf.dimx();
        projectionSizeY = inf.dimy();
        projectionSizeZ = frames.size();
        if(pmi.dimz() != inf.dimz())
        {
            ERR = io::xprintf("Incompatible number of %d projections with %d projection matrices",
                              inf.dimz(), pmi.dimz());
        }
        // How many projection matrices is there in total
        projectionFrameSize = projectionSizeX * projectionSizeY;
        totalVolumeSize = uint64_t(volumeSizeX) * uint64_t(volumeSizeY) * uint64_t(volumeSizeZ);
        totalProjectionSize
            = uint64_t(projectionSizeX) * uint64_t(projectionSizeY) * uint64_t(projectionSizeZ);
        if(totalVolumeSize > INT_MAX)
        {
            ERR = "Implement indexing by uint64_t matrix dimension overflow of voxels count.";
            LOGE << ERR;
            return 1;
        }
        // End parsing arguments
        if(totalProjectionSize > INT_MAX)
        {
            ERR = "Implement indexing by uint64_t matrix dimension overflow of projection "
                  "pixels count.";
            LOGE << ERR;
            return 1;
        }
        io::DenSupportedType t = inf.getElementType();
        if(t != io::DenSupportedType::FLOAT32)
        {
            std::string ERR
                = io::xprintf("This program supports FLOAT32 volumes only but the supplied "
                              "projection file %s is "
                              "of type %s",
                              inputProjection.c_str(), io::DenSupportedTypeToString(t).c_str());
            LOGE << ERR;
            return -1;
        }
        parsePlatformString();
        fillFramesVector(pmi.dimz());
        return 0;
    }
    void defineArguments();
    // It is evaluated from -0.5, pixels are centerred at integer coordinates
    // Here (0,0,0) is in the center of the volume
    uint64_t totalVolumeSize;
    uint64_t projectionFrameSize;
    uint64_t totalProjectionSize;
    uint32_t baseOffset = 0;
    bool noFrameOffset = false;
    bool useMinMaxBackprojector = false;
    std::string inputProjection;
    std::string outputVolume;
    std::string inputProjectionMatrices;
    std::string inputDetectorTilts;
    std::string rightHandSide = "";
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
        ->add_option("input_projection", inputProjection,
                     "Input projection FLOAT32[projx, projy, dimz]")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp
        ->add_option("input_projection_matrices", inputProjectionMatrices,
                     "Projection matrices of parallel ray geometry to be input of the computation."
                     "Files in FLOAT64 DEN format that contains projection matricess to process "
                     "with the dimensions [4,2,dimz].")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp
        ->add_option("output_volume", outputVolume,
                     "Volume to project FLOAT32 [sizex, sizey, sizez]")
        ->required();
    optstr = "Detector tilt encoded as positive cosine between normal to the detector and surface "
             "orthogonal to incomming rays. FLOAT64 file of the dimensions (1, 1, dimz). If no "
             "file is supplied, orthogonality of incomming rays to the detector is assumed, that "
             "effectivelly means all these parameters are 1.0.";
    cliApp->add_option("--detectorTilt", inputDetectorTilts, optstr);
    addForceArgs();
    addFramespecArgs();
    addCuttingVoxelProjectorArgs(true);
    addTTProjectorArgs();
    addSidonProjectorArgs();
    addCenterVoxelProjectorArgs();
    cliApp->add_option("-b,--base_offset", baseOffset, "Base offset of projections indexing.");
    cliApp->add_option(
        "--right-hand-side", rightHandSide,
        "If the parameter is specified, then we also compute the norm of the right hand "
        "side from the projected vector.");
    addVolumeSizeArgs();
    addVolumeCenterArgs();
    addVoxelSizeArgs();
    addCLSettingsArgs();
    addRelaxedArg();
}

int main(int argc, char* argv[])
{
    using namespace KCT::util;
    Program PRG(argc, argv);
    std::string prgInfo
        = "OpenCL implementation of the PBCVP and other backprojectors for parallel rays geometry.";
    if(version::MODIFIED_SINCE_COMMIT == true)
    {
        prgInfo = io::xprintf("%s Dirty commit %s", prgInfo.c_str(), version::GIT_COMMIT_ID);
    } else
    {
        prgInfo = io::xprintf("%s Git commit %s", prgInfo.c_str(), version::GIT_COMMIT_ID);
    }
    Args ARG(argc, argv, prgInfo);
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
    std::shared_ptr<io::DenGeometry3DParallelReader> geometryReader
        = std::make_shared<io::DenGeometry3DParallelReader>(ARG.inputProjectionMatrices,
                                                            ARG.inputDetectorTilts);
    std::vector<std::shared_ptr<geometry::Geometry3DParallelI>> geometryVector;
    std::shared_ptr<geometry::Geometry3DParallelI> geometry;
    for(uint32_t i = 0; i != ARG.frames.size(); i++)
    {
        uint32_t k = ARG.frames[i];
        geometry = std::make_shared<geometry::Geometry3DParallel>(geometryReader->readGeometry(k));
        geometryVector.emplace_back(geometry);
    }

    cl::NDRange projectorLocalNDRange = cl::NDRange(
        ARG.projectorLocalNDRange[0], ARG.projectorLocalNDRange[1], ARG.projectorLocalNDRange[2]);
    cl::NDRange backprojectorLocalNDRange
        = cl::NDRange(ARG.backprojectorLocalNDRange[0], ARG.backprojectorLocalNDRange[1],
                      ARG.backprojectorLocalNDRange[2]);

    // Construct projector and initialize OpenCL
    ParallelBeamProjector PBCVP(ARG.projectionSizeX, ARG.projectionSizeY, ARG.projectionSizeZ,
                                ARG.volumeSizeX, ARG.volumeSizeY, ARG.volumeSizeZ,
                                ARG.CLitemsPerWorkgroup);
    // PBCVP.initializeAllAlgorithms();
    if(ARG.useSidonProjector)
    {
        PBCVP.initializeSidonProjector(ARG.probesPerEdge, ARG.probesPerEdge);
    } else if(ARG.useTTProjector)
    {

        PBCVP.initializeTTProjector();
    } else
    {
        PBCVP.initializeCVPProjector(ARG.useBarrierCalls, ARG.barrierArraySize);
    }
    int ecd = PBCVP.initializeOpenCL(ARG.CLplatformID, &ARG.CLdeviceIDs[0], ARG.CLdeviceIDs.size(),
                                     xpath, ARG.CLdebug, ARG.CLrelaxed, projectorLocalNDRange, backprojectorLocalNDRange);
    if(ecd < 0)
    {
        std::string ERR = io::xprintf("Could not initialize OpenCL platform %d.", ARG.CLplatformID);
        KCTERR(ERR);
    }
    PBCVP.problemSetup(geometryVector, ARG.voxelSizeX, ARG.voxelSizeY, ARG.voxelSizeZ,
                       ARG.volumeCenterX, ARG.volumeCenterY, ARG.volumeCenterZ);
    float* volume = new float[ARG.totalVolumeSize];
    float* projection = new float[ARG.totalProjectionSize];
    float* projection_rhs = nullptr;
    io::DenFileInfo inputProjectionInfo(ARG.inputProjection);
    bool readxmajor = false;
    inputProjectionInfo.readIntoArray<float>(projection, readxmajor);
    PBCVP.backproject(projection, volume);
    bool volumexmajor = true;
    bool writexmajor = true;
    io::DenFileInfo::create3DDenFileFromArray(volume, volumexmajor, ARG.outputVolume,
                                              io::DenSupportedType::FLOAT32, ARG.volumeSizeX,
                                              ARG.volumeSizeY, ARG.volumeSizeZ, writexmajor);
    delete[] volume;
    delete[] projection;
    if(projection_rhs != nullptr)
    {
        delete[] projection_rhs;
    }
    PRG.endLog(true);
}
