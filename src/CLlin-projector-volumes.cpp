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
#include "CuttingVoxelProjector.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "DEN/DenProjectionMatrixReader.hpp"
#include "DEN/DenSupportedType.hpp"
#include "MATRIX/LightProjectionMatrix.hpp"
#include "PROG/ArgumentsForce.hpp"
#include "PROG/ArgumentsFramespec.hpp"
#include "PROG/Program.hpp"
#include "PROG/parseArgs.h"
#include "SMA/BufferedSparseMatrixFloatWritter.hpp"

using namespace KCT;
using namespace KCT::util;
using namespace KCT::io;

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
    void defineArguments();
    int preParse() { return 0; };
    int postParse();
    int parseArguments(int argc, char* argv[]);
    // It is evaluated from -0.5, pixels are centerred at integer coordinates
    // Here (0,0,0) is in the center of the volume
    uint64_t voxelNumX;
    uint64_t voxelNumY;
    uint64_t voxelNumZ;
    uint64_t totalVoxelNum;

    uint32_t baseOffset = 0;
    bool noFrameOffset = false;
    std::vector<std::string> inputVolumes;
    std::string inputProjectionMatrices;
    std::string outputProjection;
    std::string rightHandSide = "";
};

int Args::postParse()
{
    int e = handleFileExistence(outputProjection, force, force);
    if(e != 0)
    {
        return e;
    }
    // How many projection matrices is there in total
    io::DenFileInfo di(inputProjectionMatrices);
    fillFramesVector(di.dimz());
    if(inputVolumes.size() != frames.size())
    {
        std::string msg = io::xprintf(
            "Number of volume files %d should match number of frames to process %d.\n",
            inputVolumes.size(), frames.size());
        LOGE << msg;
        return -1;
    }
    if(inputVolumes.size() < 1)
    {
        std::string msg = io::xprintf("Number of volume files %d should be more than one.\n",
                                      inputVolumes.size());
        LOGE << msg;
        return -1;
    }
    io::DenFileInfo inf(inputVolumes[0]);
    voxelNumX = inf.dimx();
    voxelNumY = inf.dimy();
    voxelNumZ = inf.dimz();
    totalVoxelNum = voxelNumX * voxelNumY * voxelNumZ;
    for(uint32_t i = 0; i != inputVolumes.size(); i++)
    {
        inf = io::DenFileInfo(inputVolumes[i]);
        io::DenSupportedType t = inf.getElementType();
        if(t != io::DenSupportedType::FLOAT32)
        {
            std::string msg = io::xprintf("This program supports float volumes only but the "
                                          "supplied volume is of type %s!",
                                          io::DenSupportedTypeToString(t).c_str());
            LOGE << msg;
            return -1;
        }
        if(inf.dimx() != voxelNumX || inf.dimy() != voxelNumY || inf.dimz() != voxelNumZ)
        {
            std::string msg = io::xprintf("Dimensions of file %s of (%d, %d, %d) are "
                                          "incompatible with the dimensions (%d, %d, %d).",
                                          inputVolumes[i].c_str(), inf.dimx(), inf.dimy(),
                                          inf.dimz(), voxelNumX, voxelNumY, voxelNumZ);
            LOGE << msg;
            return -1;
        }
    }
    parsePlatformString();
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
    cliApp
        ->add_option("input_projection_matrices", inputProjectionMatrices,
                     "Projection matrices to be input of the computation."
                     "Files in a DEN format that contains projection matricess to process.")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp->add_option("output_projection", outputProjection, "Output projection")->required();
    cliApp->add_option("input_volumes", inputVolumes, "Volumes to project")
        ->required()
        ->check(CLI::ExistingFile);
    addForceArgs();
    addFramespecArgs();
    addCuttingVoxelProjectorArgs(true);
    addTTProjectorArgs();
    addSiddonProjectorArgs();
    addCenterVoxelProjectorArgs();
    addProjectionSizeArgs();
    addVolumeCenterArgs();
    addVoxelSizeArgs();
    addCLSettingsArgs();
    addRelaxedArg();

    cliApp->add_option("-b,--base_offset", baseOffset, "Base offset of projections indexing.");
    cliApp->add_option(
        "--right-hand-side", rightHandSide,
        "If the parameter is specified, then we also compute the norm of the right hand "
        "side from the projected vector.");
}

int main(int argc, char* argv[])
{
    using namespace KCT::util;
    Program PRG(argc, argv);
    std::string prgInfo = "OpenCL implementation of the cone beam projector for multiple volumes.";
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
    std::shared_ptr<io::DenProjectionMatrixReader> projectionMatrixReader
        = std::make_shared<io::DenProjectionMatrixReader>(ARG.inputProjectionMatrices);
    io::DenFileInfo pmi(ARG.inputProjectionMatrices);
    std::vector<std::shared_ptr<matrix::CameraI>> cameraVector;
    std::shared_ptr<matrix::CameraI> pm;
    for(std::size_t i = 0; i != ARG.frames.size(); i++)
    {
        pm = std::make_shared<matrix::LightProjectionMatrix>(
            projectionMatrixReader->readMatrix(ARG.frames[i]));
        cameraVector.emplace_back(pm);
    }
    if(uint64_t(ARG.voxelNumX) * uint64_t(ARG.voxelNumY) * uint64_t(ARG.voxelNumZ) > INT_MAX)
    {
        io::throwerr("Implement indexing by uint64_t matrix dimension overflow of voxels count.");
    }
    if(uint64_t(ARG.projectionSizeX) * uint64_t(ARG.projectionSizeY) * uint64_t(pmi.dimz())
       > INT_MAX)
    {
        KCTERR(
            "Implement indexing by uint64_t matrix dimension overflow of projection pixels count.");
    }
    // Write individual submatrices
    LOGD << io::xprintf("Number of projections to process is %d and total volume size is %d.",
                        ARG.frames.size(), ARG.totalVoxelNum);
    // End parsing arguments
    float* volume = new float[ARG.totalVoxelNum];
    CuttingVoxelProjector CVP(ARG.projectionSizeX, ARG.projectionSizeY, ARG.voxelNumX,
                              ARG.voxelNumY, ARG.voxelNumZ);
    // CVP.initializeAllAlgorithms();
    if(ARG.useSiddonProjector)
    {
        CVP.initializeSiddonProjector(ARG.probesPerEdge, ARG.probesPerEdge);
    } else if(ARG.useTTProjector)
    {

        CVP.initializeTTProjector();
    } else
    {
        CVP.initializeCVPProjector(ARG.useExactScaling, ARG.useElevationCorrection,
                                   ARG.useBarrierCalls, ARG.barrierArraySize);
    }
    int ecd = CVP.initializeOpenCL(ARG.CLplatformID, &ARG.CLdeviceIDs[0], ARG.CLdeviceIDs.size(),
                                   xpath, ARG.CLdebug, ARG.CLrelaxed);
    if(ecd < 0)
    {
        std::string ERR = io::xprintf("Could not initialize OpenCL platform %d.", ARG.CLplatformID);
        KCTERR(ERR);
    }
    uint32_t frameSize = ARG.projectionSizeX * ARG.projectionSizeY;
    float* projection = new float[frameSize]; //();
    double normSquare = 0;
    double normSquareDifference = 0;
    std::shared_ptr<io::DenFrame2DReader<float>> dpr = nullptr;
    if(!ARG.rightHandSide.empty())
    {
        dpr = std::make_shared<io::DenFrame2DReader<float>>(ARG.rightHandSide);
    }
    CVP.problemSetup(ARG.voxelSizeX, ARG.voxelSizeY, ARG.voxelSizeZ, ARG.volumeCenterX,
                     ARG.volumeCenterY, ARG.volumeCenterZ);
    io::DenAsyncFrame2DWritter<float> projectionWritter(ARG.outputProjection, ARG.projectionSizeX,
                                                        ARG.projectionSizeY, ARG.frames.size());
    bool readxmajorvolume = true;
    for(uint32_t i = 0; i != ARG.frames.size(); i++)
    {
        io::DenFileInfo volumeInfo(ARG.inputVolumes[i]);
        volumeInfo.readIntoArray<float>(volume, readxmajorvolume);
        CVP.initializeOrUpdateVolumeBuffer(volume);
        std::shared_ptr<matrix::CameraI> pm = cameraVector[i];
        int success = 0;
        success = CVP.project(projection, pm);
        if(success != 0)
        {
            std::string msg
                = io::xprintf("Some problem occurred during the projection of %s th volume",
                              ARG.inputVolumes[i].c_str());
            LOGE << msg;
            throw std::runtime_error(msg);
        }
        if(dpr != nullptr)
        {
            std::shared_ptr<io::BufferedFrame2D<float>> fr = dpr->readBufferedFrame(ARG.frames[i]);
            normSquare += CVP.normSquare((float*)fr->getDataPointer(), ARG.projectionSizeX,
                                         ARG.projectionSizeY);
            normSquareDifference += CVP.normSquareDifference(
                (float*)fr->getDataPointer(), ARG.projectionSizeX, ARG.projectionSizeY);
        }
        // io::appendBytes(ARG.outputProjection, (uint8_t*)projection, frameSize * sizeof(float));
        //        std::fill_n(projection, projectionElementsCount, float(0.0));
        io::BufferedFrame2D<float> transposedFrame(projection, ARG.projectionSizeY,
                                                   ARG.projectionSizeX);
        std::shared_ptr<io::Frame2DI<float>> frame = transposedFrame.transposed();
        projectionWritter.writeFrame(*frame, i);
        std::fill_n(projection, frameSize, float(0.0));
    }
    delete[] volume;
    delete[] projection;
    if(dpr != nullptr)
    {
        LOGI << io::xprintf(
            "Initial norm is %f, norm of the difference  is %f that is %f%% of the initial norm.",
            std::sqrt(normSquare), std::sqrt(normSquareDifference),
            std::sqrt(normSquareDifference / normSquare) * 100);
    }
    PRG.endLog(true);
}
