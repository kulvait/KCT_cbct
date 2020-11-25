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
#include "FrameMemoryViewer2D.hpp"
#include "MATRIX/LightProjectionMatrix.hpp"
#include "PROG/ArgumentsForce.hpp"
#include "PROG/ArgumentsFramespec.hpp"
#include "PROG/Program.hpp"
#include "SMA/BufferedSparseMatrixFloatWritter.hpp"

using namespace CTL;
using namespace CTL::util;

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
        std::string msg;
        if(!force)
        {
            if(io::pathExists(outputVolume))
            {
                msg = "Error: output file already exists, use --force to force overwrite.";
                LOGE << msg;
                return 1;
            }
        }
        // How many projection matrices is there in total
        io::DenFileInfo pmi(inputProjectionMatrices);
        io::DenFileInfo inf(inputProjection);
        projectionSizeX = inf.dimx();
        projectionSizeY = inf.dimy();
        projectionSizeZ = inf.dimz();
        projectionFrameSize = projectionSizeX * projectionSizeY;
        totalVolumeSize = uint64_t(volumeSizeX) * uint64_t(volumeSizeY) * uint64_t(volumeSizeZ);
        totalProjectionSize
            = uint64_t(projectionSizeX) * uint64_t(projectionSizeY) * uint64_t(projectionSizeZ);
        if(totalVolumeSize > INT_MAX)
        {
            msg = "Implement indexing by uint64_t matrix dimension overflow of voxels count.";
            LOGE << msg;
            return 1;
        }
        // End parsing arguments
        if(totalProjectionSize > INT_MAX)
        {
            msg = "Implement indexing by uint64_t matrix dimension overflow of projection "
                  "pixels count.";
            LOGE << msg;
            return 1;
        }
        if(projectionSizeZ != pmi.dimz())
        {
            msg = io::xprintf(
                "Number of the projection matrices %d do not match with the dimension of "
                "projections %d!",
                pmi.dimz(), projectionSizeZ);
            LOGE << msg;
            return 1;
        }
        io::DenSupportedType t = inf.getDataType();
        if(t != io::DenSupportedType::float_)
        {
            std::string ERR
                = io::xprintf("This program supports float projections only but the supplied "
                              "projection file %s is "
                              "of type %s",
                              inputProjection.c_str(), io::DenSupportedTypeToString(t).c_str());
            LOGE << ERR;
            throw std::runtime_error(ERR);
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

    cliApp->add_option("input_projection", inputProjection, "Input projection")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp
        ->add_option("input_projection_matrices", inputProjectionMatrices,
                     "Projection matrices to be input of the computation."
                     "Files in a DEN format that contains projection matricess to process.")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp->add_option("output_volume", outputVolume, "Volume to output")->required();
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
    og_projectortypesettings->add_flag("--minmax", useMinMaxBackprojector,
                                       io::xprintf("Use minmax backprojector."));
    addVolumeSizeArgs();
    addPixelSizeArgs();
    addVoxelSizeArgs();
    addCLSettingsArgs();
}

int main(int argc, char* argv[])
{
    using namespace CTL::util;
    Program PRG(argc, argv);
    std::string prgInfo = "OpenCL implementation of the cone beam backprojection.";
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

    // Construct projector and initialize OpenCL
    CuttingVoxelProjector CVP(ARG.projectionSizeX, ARG.projectionSizeY, ARG.volumeSizeX,
                              ARG.volumeSizeY, ARG.volumeSizeZ);
    CVP.initializeAllAlgorithms();
    if(ARG.useSidonProjector)
    {
        CVP.initializeSidonProjector(ARG.probesPerEdge, ARG.probesPerEdge);
    } else if(ARG.useTTProjector)
    {

        CVP.initializeTTProjector();
    } else
    {
        CVP.initializeCVPProjector(ARG.useExactScaling);
    }
    int ecd = CVP.initializeOpenCL(ARG.CLplatformID, &ARG.CLdeviceIDs[0], ARG.CLdeviceIDs.size(),
                                   xpath, ARG.CLdebug);
    if(ecd < 0)
    {
        std::string ERR = io::xprintf("Could not initialize OpenCL platform %d.", ARG.CLplatformID);
        LOGE << ERR;
        throw std::runtime_error(ERR);
    }
    // Write individual submatrices
    LOGD << io::xprintf("Number of projections to process is %d.", ARG.frames.size());
    // End parsing arguments
    std::vector<std::shared_ptr<matrix::CameraI>> cameraVector;
    std::shared_ptr<matrix::CameraI> pm;
    float* projection = new float[ARG.projectionFrameSize * ARG.frames.size()];
    {
        io::DenFrame2DReader<float> dpr(ARG.inputProjection);
        for(uint32_t i = 0; i != ARG.frames.size(); i++)
        {
            dpr.readFrameIntoBuffer(ARG.frames[i], projection + i * ARG.projectionFrameSize);
            pm = std::make_shared<matrix::LightProjectionMatrix>(dr->readMatrix(ARG.frames[i]));
            cameraVector.emplace_back(pm);
        }
    }
    float* volume = new float[ARG.totalVolumeSize];
    CVP.initializeOrUpdateVolumeBuffer(ARG.volumeSizeX, ARG.volumeSizeY, ARG.volumeSizeZ);
    CVP.initializeOrUpdateProjectionBuffer(ARG.projectionSizeX, ARG.projectionSizeY,
                                           ARG.frames.size(), projection);
    double normSquare = 0;
    double normSquareDifference = 0;
    std::shared_ptr<io::DenFrame2DReader<float>> dpr = nullptr;
    if(!ARG.rightHandSide.empty())
    {
        dpr = std::make_shared<io::DenFrame2DReader<float>>(ARG.rightHandSide);
    }
    if(ARG.useMinMaxBackprojector)
    {
        CVP.backproject_minmax(volume, cameraVector);
    } else
    {
        CVP.backproject(volume, cameraVector);
    }
    io::DenAsyncFrame2DWritter<float> ofw(ARG.outputVolume, ARG.volumeSizeX, ARG.volumeSizeY,
                                          ARG.volumeSizeZ);
    uint64_t volumeFrameSize = ARG.volumeSizeX * ARG.volumeSizeY;
    for(uint32_t k = 0; k != ARG.volumeSizeZ; k++)
    {
        io::FrameMemoryViewer2D<float> frame(volume + k * volumeFrameSize, ARG.volumeSizeX,
                                             ARG.volumeSizeY);
        ofw.writeFrame(frame, k);
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
