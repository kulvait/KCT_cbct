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

// Internal libraries
#include "CArmArguments.hpp"
#include "CuttingVoxelProjector.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "DEN/DenProjectionMatrixReader.hpp"
#include "DEN/DenSupportedType.hpp"
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
    addPixelSizeArgs();
    addVoxelSizeArgs();
    addCLSettingsArgs();
}

int main(int argc, char* argv[])
{
    using namespace CTL::util;
    Program PRG(argc, argv);
    Args ARG(argc, argv, "OpenCL implementation of the voxel cutting projector.");
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

    // Write individual submatrices
    LOGD << io::xprintf("Number of projections to process is %d.", ARG.frames.size());
    // End parsing arguments
    float* volume = new float[ARG.totalVolumeSize];
    io::readBytesFrom(ARG.inputVolume, 6, (uint8_t*)volume, ARG.totalVolumeSize * 4);
    std::shared_ptr<CuttingVoxelProjector> cvp = std::make_shared<CuttingVoxelProjector>(
        volume, ARG.volumeSizeX, ARG.volumeSizeY, ARG.volumeSizeZ, ARG.voxelSizeX, ARG.voxelSizeY,
        ARG.voxelSizeZ, ARG.pixelSizeX, ARG.pixelSizeY, xpath, ARG.CLdebug, ARG.useCenterVoxelProjector,
        ARG.useExactScaling);
    int res = cvp->initializeOpenCL(ARG.CLplatformID);
    if(res < 0)
    {
        std::string ERR = io::xprintf("Could not initialize OpenCL platform %d.", ARG.CLplatformID);
        LOGE << ERR;
        io::throwerr(ERR);
    }
    cvp->initializeVolumeImage();
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
        dpr = std::make_shared<io::DenFrame2DReader<float>>(ARG.rightHandSide);
    }
    double xoveryspacing = ARG.pixelSizeX / ARG.pixelSizeY;
    double yoverxspacing = ARG.pixelSizeY / ARG.pixelSizeX;
    for(int f : ARG.frames)
    {
        matrix::ProjectionMatrix pm = dr->readMatrix(f);

        double x1, x2, y1, y2;
        std::array<double, 3> sourcePosition = pm.sourcePosition();
        std::array<double, 3> normalToDetector = pm.normalToDetector();
        std::array<double, 3> tangentToDetector = pm.tangentToDetectorYDirection();
        pm.project(sourcePosition[0] - normalToDetector[0], sourcePosition[1] - normalToDetector[1],
                   sourcePosition[2] - normalToDetector[2], &x1, &y1);
        pm.project(sourcePosition[0] - normalToDetector[0] + tangentToDetector[0],
                   sourcePosition[1] - normalToDetector[1] + tangentToDetector[1],
                   sourcePosition[2] - normalToDetector[2] + tangentToDetector[2], &x2, &y2);
        double scalingFactor
            = (x1 - x2) * (x1 - x2) * xoveryspacing + (y1 - y2) * (y1 - y2) * yoverxspacing;
        if(ARG.useSidonProjector)
        {
            cvp->projectSiddon(projection, ARG.projectionSizeX, ARG.projectionSizeY, pm,
                               scalingFactor, ARG.probesPerEdge);
        } else if(ARG.useTTProjector)
        {
            double sourceToDetector
                = std::sqrt((x1 - x2) * (x1 - x2) * ARG.pixelSizeX * ARG.pixelSizeX
                            + (y1 - y2) * (y1 - y2) * ARG.pixelSizeY * ARG.pixelSizeY);
            cvp->projectTA3(projection, ARG.projectionSizeX, ARG.projectionSizeY, x1, y1,
                            sourceToDetector, pm);
        } else
        {
            if(ARG.useCosScaling)
            {
                cvp->projectCos(projection, ARG.projectionSizeX, ARG.projectionSizeY, pm,
                                scalingFactor);
            } else if(ARG.useNoScaling)
            {

                double sourceToDetector
                    = std::sqrt((x1 - x2) * (x1 - x2) * ARG.pixelSizeX * ARG.pixelSizeX
                                + (y1 - y2) * (y1 - y2) * ARG.pixelSizeY * ARG.pixelSizeY);
                cvp->projectorWithoutScaling(projection, ARG.projectionSizeX, ARG.projectionSizeY,
                                             x1, y1, sourceToDetector, pm);
            }else
            {
                double sourceToDetector
                    = std::sqrt((x1 - x2) * (x1 - x2) * ARG.pixelSizeX * ARG.pixelSizeX
                                + (y1 - y2) * (y1 - y2) * ARG.pixelSizeY * ARG.pixelSizeY);
                cvp->projectExact(projection, ARG.projectionSizeX, ARG.projectionSizeY, x1, y1,
                                  sourceToDetector, pm);
            }
        }
        if(dpr != nullptr)
        {
            std::shared_ptr<io::BufferedFrame2D<float>> fr = dpr->readBufferedFrame(f);
            normSquare += cvp->normSquare((float*)fr->getDataPointer(), ARG.projectionSizeX,
                                          ARG.projectionSizeY);
            normSquareDifference += cvp->normSquareDifference(
                (float*)fr->getDataPointer(), ARG.projectionSizeX, ARG.projectionSizeY);
        }
        io::appendBytes(ARG.outputProjection, (uint8_t*)projection,
                        projectionElementsCount * sizeof(float));
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
