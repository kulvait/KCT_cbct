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

// Essential internal libraries
#include "CArmArguments.hpp"
#include "DEN/DenFileInfo.hpp"
#include "PROG/ArgumentsForce.hpp"
#include "PROG/Program.hpp"
#include "VolumeConvolutionOperator.hpp"
#include "rawop.h"

// Internal includes
//#include "DEN/DenAsyncFrame2DWritter.hpp"
//#include "DEN/DenFrame2DReader.hpp"
//#include "DEN/DenProjectionMatrixReader.hpp"
//#include "DEN/DenSupportedType.hpp"
//#include "MATRIX/LightProjectionMatrix.hpp"
//#include "SMA/BufferedSparseMatrixFloatWritter.hpp"

using namespace CTL;
using namespace CTL::util;

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
        std::string ERR;
        if(!force)
        {
            if(sobelGradient3D)
            {
                std::string gx = io::xprintf("%s_x", outputVolume.c_str());
                std::string gy = io::xprintf("%s_y", outputVolume.c_str());
                std::string gz = io::xprintf("%s_z", outputVolume.c_str());
                if(io::pathExists(gx) || io::pathExists(gy) || io::pathExists(gz))
                {
                    ERR = "Error: output file already exists, use --force to force overwrite.";
                    LOGE << ERR;
                    return -1;
                }
            } else
            {
                if(io::pathExists(outputVolume))
                {
                    ERR = "Error: output file already exists, use --force to force overwrite.";
                    LOGE << ERR;
                    return -1;
                }
            }
        }
        // How many projection matrices is there in total
        io::DenFileInfo inf(inputVolume);
        volumeSizeX = inf.dimx();
        volumeSizeY = inf.dimy();
        volumeSizeZ = inf.dimz();
        totalVolumeSize = uint64_t(volumeSizeX) * uint64_t(volumeSizeY) * uint64_t(volumeSizeZ);
        if(totalVolumeSize > INT_MAX)
        {
            ERR = io::xprintf(
                "Implement indexing by uint64_t matrix dimension overflow of voxels count.");
            LOGE << ERR;
            return -1;
        }
        io::DenSupportedType t = inf.getDataType();
        if(t != io::DenSupportedType::float_)
        {
            ERR = io::xprintf("This program supports float volumes only but the supplied "
                              "projection file %s is "
                              "of type %s",
                              inputVolume.c_str(), io::DenSupportedTypeToString(t).c_str());
            LOGE << ERR;
            return -1;
        }
        parsePlatformString();
        return 0;
    }
    void defineArguments();
    // It is evaluated from -0.5, pixels are centerred at integer coordinates
    // Here (0,0,0) is in the center of the volume
    uint64_t totalVolumeSize;
    bool sobelGradient3D = false;
    std::string inputVolume;
    std::string outputVolume;
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
    cliApp->add_option("output_volume", outputVolume, "Output projection")->required();
    cliApp->add_flag("--sobel-gradient-3d", sobelGradient3D, "Output projection")->required();
    addForceArgs();
    addVoxelSizeArgs();
    addCLSettingsArgs();
    addRelaxedArg();
}

int main(int argc, char* argv[])
{
    using namespace CTL::util;
    Program PRG(argc, argv);
    std::string prgInfo = "OpenCL implementation of the convolution of the volumes.";
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
    cl::NDRange projectorLocalNDRange = cl::NDRange(
        ARG.projectorLocalNDRange[0], ARG.projectorLocalNDRange[1], ARG.projectorLocalNDRange[2]);

    // Construct projector and initialize OpenCL
    VolumeConvolutionOperator VCO(ARG.volumeSizeX, ARG.volumeSizeY, ARG.voxelSizeZ,
                                  projectorLocalNDRange);
    VCO.initializeConvolution();
    int ecd = VCO.initializeOpenCL(ARG.CLplatformID, &ARG.CLdeviceIDs[0], ARG.CLdeviceIDs.size(),
                                   xpath, ARG.CLdebug, ARG.CLrelaxed);
    if(ecd < 0)
    {
        std::string ERR = io::xprintf("Could not initialize OpenCL platform %d.", ARG.CLplatformID);
        LOGE << ERR;
        throw std::runtime_error(ERR);
    }
    VCO.problemSetup(ARG.voxelSizeX, ARG.voxelSizeY, ARG.voxelSizeZ);
    // End parsing arguments
    LOGI << io::xprintf("Before if ARG.sobelGradient3D=%d", ARG.sobelGradient3D);
    if(ARG.sobelGradient3D)
    {
        float* vx = new float[ARG.totalVolumeSize];
        float* vy = new float[ARG.totalVolumeSize];
        float* vz = new float[ARG.totalVolumeSize];
        io::DenFileInfo volumeInfo(ARG.inputVolume);
        // Use vx also as input buffer
        io::readBytesFrom(ARG.inputVolume, volumeInfo.getOffset(), (uint8_t*)vx,
                          ARG.totalVolumeSize * 4);
        VCO.initializeOrUpdateVolumeBuffer(ARG.volumeSizeX, ARG.volumeSizeY, ARG.volumeSizeZ, vx);
        cl_float3 voxelSizes
            = { (float)ARG.voxelSizeX, (float)ARG.voxelSizeY, (float)ARG.voxelSizeZ };
        LOGI << io::xprintf("Call VCO.sobelGradient3D.");
        VCO.sobelGradient3D(voxelSizes, vx, vy, vz);
        LOGI << io::xprintf("After VCO.sobelGradient3D.");
        uint64_t totalArraySize = ARG.totalVolumeSize * sizeof(float);
        std::string gx = io::xprintf("%s_x", ARG.outputVolume.c_str());
        std::string gy = io::xprintf("%s_y", ARG.outputVolume.c_str());
        std::string gz = io::xprintf("%s_z", ARG.outputVolume.c_str());
        LOGI << io::xprintf("Printing x to file %s.", gx.c_str());
        io::DenFileInfo::createDenHeader(gx, ARG.volumeSizeX, ARG.volumeSizeY, ARG.volumeSizeZ);
        io::appendBytes(gx, (uint8_t*)vx, totalArraySize);
        io::DenFileInfo::createDenHeader(gy, ARG.volumeSizeX, ARG.volumeSizeY, ARG.volumeSizeZ);
        io::appendBytes(gy, (uint8_t*)vy, totalArraySize);
        io::DenFileInfo::createDenHeader(gz, ARG.volumeSizeX, ARG.volumeSizeY, ARG.volumeSizeZ);
        io::appendBytes(gz, (uint8_t*)vz, totalArraySize);
        delete[] vx;
        delete[] vy;
        delete[] vz;
    } else
    {
        float* volume = new float[ARG.totalVolumeSize];
        io::DenFileInfo volumeInfo(ARG.inputVolume);
        io::readBytesFrom(ARG.inputVolume, volumeInfo.getOffset(), (uint8_t*)volume,
                          ARG.totalVolumeSize * 4);
        VCO.initializeOrUpdateVolumeBuffer(ARG.volumeSizeX, ARG.volumeSizeY, ARG.volumeSizeZ,
                                           volume);
        std::string kernelName = "Laplace";
        VCO.convolve(kernelName, volume);

        io::DenFileInfo::createDenHeader(ARG.outputVolume, ARG.volumeSizeX, ARG.volumeSizeY,
                                         ARG.volumeSizeZ);
        uint64_t totalArraySize = ARG.totalVolumeSize * sizeof(float);
        io::appendBytes(ARG.outputVolume, (uint8_t*)volume, totalArraySize);
        delete[] volume;
    }
    PRG.endLog(true);
}
