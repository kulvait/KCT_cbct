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
        if(!force)
        {
            if(io::pathExists(outputProjection))
            {
                std::string msg
                    = "Error: output file already exists, use --force to force overwrite.";
                LOGE << msg;
                return 1;
            }
        }
        // How many projection matrices is there in total
        io::DenFileInfo pmi(inputProjectionMatrices);
        io::DenFileInfo inf(inputVolume);
        volumeSizeX = inf.dimx();
        volumeSizeY = inf.dimy();
        volumeSizeZ = inf.dimz();
        totalVolumeSize = uint64_t(volumeSizeX) * uint64_t(volumeSizeY) * uint64_t(volumeSizeZ);
        totalProjectionSize
            = uint64_t(projectionSizeX) * uint64_t(projectionSizeY) * uint64_t(projectionSizeZ);
        if(totalVolumeSize > INT_MAX)
        {
            io::throwerr(
                "Implement indexing by uint64_t matrix dimension overflow of voxels count.");
        }
        // End parsing arguments
        if(totalProjectionSize > INT_MAX)
        {
            io::throwerr("Implement indexing by uint64_t matrix dimension overflow of projection "
                         "pixels count.");
        }
        io::DenSupportedType t = inf.getDataType();
        if(t != io::DenSupportedType::float_)
        {
            std::string ERR
                = io::xprintf("This program supports float volumes only but the supplied "
                              "projection file %s is "
                              "of type %s",
                              inputVolume.c_str(), io::DenSupportedTypeToString(t).c_str());
            LOGE << ERR;
            io::throwerr(ERR);
        }
        parsePlatformString();
        fillFramesVector(pmi.dimz());
        return 0;
    }
    void defineArguments();
    // It is evaluated from -0.5, pixels are centerred at integer coordinates
    // Here (0,0,0) is in the center of the volume
    uint64_t totalVolumeSize;
    uint64_t totalProjectionSize;
    uint32_t baseOffset = 0;
    bool noFrameOffset = false;
    std::string inputVolume;
    std::string inputProjectionMatrices;
    std::string outputProjection;
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

    cliApp->add_option("input_volume", inputVolume, "Volume to project")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp
        ->add_option("input_projection_matrices", inputProjectionMatrices,
                     "Projection matrices to be input of the computation."
                     "Files in a DEN format that contains projection matricess to process.")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp->add_option("output_projection", outputProjection, "Output projection")->required();
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
    addProjectionSizeArgs();
    addProjectorLocalNDRangeArgs();
    addVolumeCenterArgs();
    addVoxelSizeArgs();
    addCLSettingsArgs();
    addRelaxedArg();
}

int main(int argc, char* argv[])
{
    using namespace KCT::util;
    Program PRG(argc, argv);
    std::string prgInfo = "OpenCL implementation of the cutting voxel projector.";
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
    std::shared_ptr<io::DenProjectionMatrixReader> dr
        = std::make_shared<io::DenProjectionMatrixReader>(ARG.inputProjectionMatrices);
    cl::NDRange projectorLocalNDRange = cl::NDRange(
        ARG.projectorLocalNDRange[0], ARG.projectorLocalNDRange[1], ARG.projectorLocalNDRange[2]);

    // Construct projector and initialize OpenCL
    CuttingVoxelProjector CVP(ARG.projectionSizeX, ARG.projectionSizeY, ARG.volumeSizeX,
                              ARG.volumeSizeY, ARG.voxelSizeZ, projectorLocalNDRange);
    // CVP.initializeAllAlgorithms();
    if(ARG.useSidonProjector)
    {
        CVP.initializeSidonProjector(ARG.probesPerEdge, ARG.probesPerEdge);
    } else if(ARG.useTTProjector)
    {

        CVP.initializeTTProjector();
    } else
    {
        CVP.initializeCVPProjector(ARG.useExactScaling, ARG.useBarrierCalls, ARG.barrierArraySize);
    }
    int ecd = CVP.initializeOpenCL(ARG.CLplatformID, &ARG.CLdeviceIDs[0], ARG.CLdeviceIDs.size(),
                                   xpath, ARG.CLdebug, ARG.CLrelaxed);
    if(ecd < 0)
    {
        std::string ERR = io::xprintf("Could not initialize OpenCL platform %d.", ARG.CLplatformID);
        LOGE << ERR;
        throw std::runtime_error(ERR);
    }
    CVP.problemSetup(ARG.voxelSizeX, ARG.voxelSizeY, ARG.voxelSizeZ, ARG.volumeCenterX,
                     ARG.volumeCenterY, ARG.volumeCenterZ);
    // Write individual submatrices
    LOGD << io::xprintf("Number of projections to process is %d.", ARG.frames.size());
    // End parsing arguments
    float* volume = new float[ARG.totalVolumeSize];
    io::readBytesFrom(ARG.inputVolume, 6, (uint8_t*)volume, ARG.totalVolumeSize * 4);
    CVP.initializeOrUpdateVolumeBuffer(ARG.volumeSizeX, ARG.volumeSizeY, ARG.volumeSizeZ, volume);
    CVP.initializeOrUpdateProjectionBuffer(ARG.projectionSizeX, ARG.projectionSizeY, 1);
    uint32_t projectionElementsCount = ARG.projectionSizeX * ARG.projectionSizeY;
    float* projection = new float[projectionElementsCount]();
    uint16_t buf[3];
    buf[0] = ARG.projectionSizeY;
    buf[1] = ARG.projectionSizeX;
    buf[2] = ARG.frames.size();
    io::createEmptyFile(ARG.outputProjection, 0, true); // Try if this is faster
    io::appendBytes(ARG.outputProjection, (uint8_t*)buf, (uint64_t)6);
    double normSquare = 0;
    double normSquareDifference = 0;
    std::shared_ptr<io::DenFrame2DReader<float>> dpr = nullptr;
    if(!ARG.rightHandSide.empty())
    {
        LOGD << io::xprintf("Initialize RHS");
        dpr = std::make_shared<io::DenFrame2DReader<float>>(ARG.rightHandSide);
    }
    io::DenAsyncFrame2DWritter<float> projectionWritter(ARG.outputProjection, ARG.projectionSizeX,
                                                        ARG.projectionSizeY, ARG.frames.size());
    for(uint32_t i = 0; i != ARG.frames.size(); i++)
    {
        uint32_t f = ARG.frames[i];
        using namespace KCT::matrix;
        std::shared_ptr<CameraI> P = std::make_shared<LightProjectionMatrix>(dr->readMatrix(f));
        if(ARG.useSidonProjector)
        {
            CVP.projectSidon(projection, P);
        } else if(ARG.useTTProjector)
        {
            CVP.projectTA3(projection, P);
        } else
        {
            if(ARG.useCosScaling)
            {
                CVP.projectCos(projection, P);
            } else if(ARG.useNoScaling)
            {
                CVP.projectorWithoutScaling(projection, P);
            } else
            {
                CVP.projectExact(projection, P);
            }
        }
        if(dpr != nullptr)
        {
            std::shared_ptr<io::BufferedFrame2D<float>> fr = dpr->readBufferedFrame(f);
            normSquare += CVP.normSquare((float*)fr->getDataPointer(), ARG.projectionSizeX,
                                         ARG.projectionSizeY);
            normSquareDifference += CVP.normSquareDifference(
                (float*)fr->getDataPointer(), ARG.projectionSizeX, ARG.projectionSizeY);
        }
        io::BufferedFrame2D<float> transposedFrame(projection, ARG.projectionSizeY,
                                                   ARG.projectionSizeX);
        std::shared_ptr<io::Frame2DI<float>> frame = transposedFrame.transposed();
        projectionWritter.writeFrame(*frame, i);
        std::fill_n(projection, projectionElementsCount, float(0.0));
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
