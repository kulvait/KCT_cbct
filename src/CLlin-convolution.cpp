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
    int postParse()
    {
        int e = 0;
        std::string ERR;
        if(sobelGradient3DZeroBoundary || sobelGradient3DReflectionBoundary)
        {
            std::string gx = io::xprintf("%s_x", outputVolume.c_str());
            std::string gy = io::xprintf("%s_y", outputVolume.c_str());
            std::string gz = io::xprintf("%s_z", outputVolume.c_str());
            e = handleFileExistence(gx, force, force);
            if(e != 0)
            {
                return e;
            }
            e = handleFileExistence(gy, force, force);
            if(e != 0)
            {
                return e;
            }
            e = handleFileExistence(gz, force, force);
            if(e != 0)
            {
                return e;
            }
        } else
        {
            e = handleFileExistence(outputVolume, force, force);
            if(e != 0)
            {
                return e;
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
        io::DenSupportedType t = inf.getElementType();
        if(t != io::DenSupportedType::FLOAT32)
        {
            ERR = io::xprintf("This program supports FLOAT32 volumes only but the supplied "
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
    bool sobelGradient3DZeroBoundary = false;
    bool farid5Gradient3DZeroBoundary = false;
    bool farid5Gradient3DReflectionBoundary = false;
    bool sobelGradient3DReflectionBoundary = false;
    bool isotropicGradient3D = false;
    bool laplace3D = false;
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
    CLI::Option* srb = cliApp->add_flag("--sobel-gradient-3d-reflection-boundary",
                                        sobelGradient3DReflectionBoundary, "3D gradient.");
    CLI::Option* szb = cliApp->add_flag("--sobel-gradient-3d-zero-boundary",
                                        sobelGradient3DZeroBoundary, "3D gradient.");
    CLI::Option* f53zb = cliApp->add_flag("--farid5-gradient-3d-zero-boundary",
                                          farid5Gradient3DZeroBoundary, "3D gradient.");
    CLI::Option* f53rb = cliApp->add_flag("--farid5-gradient-3d-reflection-boundary",
                                          farid5Gradient3DReflectionBoundary, "3D gradient.");
    CLI::Option* ig3
        = cliApp->add_flag("--isotropic-gradient-3d", isotropicGradient3D, "3D gradient.");
    CLI::Option* l3d = cliApp->add_flag("--laplace-3d", laplace3D, "3D Laplace operator.");
    srb->excludes(szb)->excludes(l3d)->excludes(f53zb)->excludes(ig3)->excludes(f53rb);
    szb->excludes(l3d)->excludes(srb)->excludes(f53zb)->excludes(ig3)->excludes(f53rb);
    l3d->excludes(srb)->excludes(szb)->excludes(f53zb)->excludes(ig3)->excludes(f53rb);
    f53zb->excludes(srb)->excludes(szb)->excludes(l3d)->excludes(ig3)->excludes(f53rb);
    ig3->excludes(srb)->excludes(szb)->excludes(l3d)->excludes(f53zb)->excludes(f53rb);
    f53rb->excludes(srb)->excludes(szb)->excludes(l3d)->excludes(ig3)->excludes(f53zb);

    addForceArgs();
    addVoxelSizeArgs();
    addCLSettingsArgs();
    addRelaxedArg();
}

int main(int argc, char* argv[])
{
    using namespace KCT::util;
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
    if(ARG.sobelGradient3DReflectionBoundary || ARG.sobelGradient3DZeroBoundary
       || ARG.farid5Gradient3DZeroBoundary || ARG.isotropicGradient3D
       || ARG.farid5Gradient3DReflectionBoundary)
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
        if(ARG.sobelGradient3DReflectionBoundary || ARG.sobelGradient3DZeroBoundary)
        {
            VCO.sobelGradient3D(
                voxelSizes, vx, vy, vz,
                ARG.sobelGradient3DReflectionBoundary); // When false it will lead to
                                                        // zero boundary conditions
        } else if(ARG.farid5Gradient3DZeroBoundary || ARG.farid5Gradient3DReflectionBoundary)
        {

            VCO.faridGradient3D(voxelSizes, vx, vy, vz, ARG.farid5Gradient3DReflectionBoundary);
        } else
        {
            VCO.isotropicGradient3D(voxelSizes, vx, vy, vz);
        }
        uint64_t totalArraySize = ARG.totalVolumeSize * sizeof(float);
        std::string gx = io::xprintf("%s_x", ARG.outputVolume.c_str());
        std::string gy = io::xprintf("%s_y", ARG.outputVolume.c_str());
        std::string gz = io::xprintf("%s_z", ARG.outputVolume.c_str());
        io::DenFileInfo::create3DDenHeader(gx, DenSupportedType::FLOAT32, ARG.volumeSizeX,
                                           ARG.volumeSizeY, ARG.volumeSizeZ);
        io::appendBytes(gx, (uint8_t*)vx, totalArraySize);
        io::DenFileInfo::create3DDenHeader(gy, DenSupportedType::FLOAT32, ARG.volumeSizeX,
                                           ARG.volumeSizeY, ARG.volumeSizeZ);
        io::appendBytes(gy, (uint8_t*)vy, totalArraySize);
        io::DenFileInfo::create3DDenHeader(gz, DenSupportedType::FLOAT32, ARG.volumeSizeX,
                                           ARG.volumeSizeY, ARG.volumeSizeZ);
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
        if(ARG.laplace3D)
        {
            cl_float3 voxelSizes
                = { (float)ARG.voxelSizeX, (float)ARG.voxelSizeY, (float)ARG.voxelSizeZ };
            VCO.laplace3D(voxelSizes, volume);
        } else
        {
            std::string kernelName = "Laplace";
            VCO.convolve(kernelName, volume);
        }
        io::DenFileInfo::create3DDenHeader(ARG.outputVolume, DenSupportedType::FLOAT32,
                                           ARG.volumeSizeX, ARG.volumeSizeY, ARG.volumeSizeZ);
        uint64_t totalArraySize = ARG.totalVolumeSize * sizeof(float);
        io::appendBytes(ARG.outputVolume, (uint8_t*)volume, totalArraySize);
        delete[] volume;
    }
    PRG.endLog(true);
}
