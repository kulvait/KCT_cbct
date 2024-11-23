// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

// External libraries
#include "CLI/CLI.hpp"
#include "gitversion/version.h"

// Internal libraries
#include "CArmArguments.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "GradientType.hpp"
#include "PDHGROFExecutor.hpp"
#include "PROG/ArgumentsForce.hpp"
#include "PROG/KCTException.hpp"
#include "PROG/Program.hpp"

// Internal includes
using namespace KCT;
using namespace KCT::util;
using namespace KCT::io;

/** Arguments parsed by the main function. */
class Args : public CArmArguments, public ArgumentsForce
{
public:
    Args(int argc, char** argv, std::string programName)
        : Arguments(argc, argv, programName)
        , CArmArguments(argc, argv, programName)
        , ArgumentsForce(argc, argv, programName){};

    int preParse() { return 0; }
    int postParse();

    void defineArguments();

    uint64_t totalVolumeSize;
    uint32_t slabFrom = 0;
    uint32_t slabSize = 0;
    std::string inputVolume;
    std::string outputVolume;
    std::string initialVectorX0;
    float pdhg_mu = -0.1; // Lambda parameter in ||Ax-b|| + lambda TV(X)
    // float pdhg_tau = -0.7*0.125; // Primal variable update
    // float pdhg_sigma = -0.7*0.125; // Dual variable update
    float pdhg_tau = -0.7; // Primal variable update
    float pdhg_sigma = -0.7; // Dual variable update
                             // 1/sqrt(2)=0.7071 > 0.7
    float pdhg_theta = 1.0; // Relaxation
    GradientType useGradientType = GradientType::ForwardDifference2Point;

private:
    std::string str;
};

void Args::defineArguments()
{
    cliApp->add_option("input_volume", inputVolume, "Volume to denoise")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp->add_option("output_volume", outputVolume, "Output denoised volume")->required();
    addSettingsGroup();
    og_settings
        ->add_option(
            "--max-iterations-pdhg", maxIterationPDHG,
            io::xprintf("Maximum number of PDHG iterations, defaults to %d.", maxIterationPDHG))
        ->check(CLI::Range(1, 65535));
    og_settings
        ->add_option(
            "--stopping-relative-error-pdhg", stoppingRelativePDHG,
            io::xprintf("Stopping relative error of PDHG, defaults to %f.", stoppingRelativePDHG))
        ->check(CLI::Range(0.0, 1.0));
    // Start PDHG options
    CLI::Option_group* og_pdhg = og_settings->add_option_group(
        "PDHG Options", "Primal Dual Hybrid Gradient method options.");
    registerOptionGroup("PDHG options", og_pdhg);
    str = io::xprintf("PDHG lambda parameter in ||Ax-b|| + mu TV(X), [defaults to %f]", pdhg_mu);
    og_pdhg->add_option("--pdhg-mu", pdhg_mu, str);
    str = io::xprintf(
        "Primal variable update parameter, negative values are multiplied by -1, [defaults to %f]",
        pdhg_tau);
    og_pdhg->add_option("--pdhg-tau", pdhg_tau, str);
    str = io::xprintf("Dual variable update parameter, negative values are multiplied by "
                      "voxel_size*voxel_size, [defaults to %f]",
                      pdhg_sigma);
    og_pdhg->add_option("--pdhg-sigma", pdhg_sigma, str);
    str = io::xprintf("Relaxation parameter, [defaults to %f]", pdhg_theta);
    og_pdhg->add_option("--pdhg-theta", pdhg_theta, str);
    CLI::Option_group* og_pdhg_gradient
        = og_pdhg->add_option_group("Gradient Type", "Gradient type used in PDHG method.");
    og_pdhg_gradient->add_flag_function(
        "--gradient-forward-2point",
        [this](std::int64_t count) { useGradientType = GradientType::ForwardDifference2Point; },
        "Use forward difference 2 point gradient, [default].");
    og_pdhg_gradient->add_flag_function(
        "--gradient-forward-3point",
        [this](std::int64_t count) { useGradientType = GradientType::ForwardDifference3Point; },
        "Use forward difference 3 point gradient.");
    og_pdhg_gradient->add_flag_function(
        "--gradient-forward-4point",
        [this](std::int64_t count) { useGradientType = GradientType::ForwardDifference4Point; },
        "Use forward difference 4 point gradient.");
    og_pdhg_gradient->add_flag_function(
        "--gradient-forward-5point",
        [this](std::int64_t count) { useGradientType = GradientType::ForwardDifference5Point; },
        "Use forward difference 5 point gradient.");
    og_pdhg_gradient->add_flag_function(
        "--gradient-forward-6point",
        [this](std::int64_t count) { useGradientType = GradientType::ForwardDifference6Point; },
        "Use forward difference 6 point gradient.");
    og_pdhg_gradient->add_flag_function(
        "--gradient-forward-7point",
        [this](std::int64_t count) { useGradientType = GradientType::ForwardDifference7Point; },
        "Use forward difference 7 point gradient.");
    og_pdhg_gradient->add_flag_function(
        "--gradient-central-3point",
        [this](std::int64_t count) { useGradientType = GradientType::CentralDifference3Point; },
        "Use central difference 3 point gradient.");
    og_pdhg_gradient->add_flag_function(
        "--gradient-central-5point",
        [this](std::int64_t count) { useGradientType = GradientType::CentralDifference5Point; },
        "Use central difference 5 point gradient.");
    og_pdhg_gradient->require_option(0, 1);

    addVoxelSizeArgs();
    og_geometry->add_option(
        "--slab-from", slabFrom,
        "Use for slab reconstruction, reconstruct only part of the projection data.");
    og_geometry->add_option("--slab-size", slabSize,
                            "Use for slab reconstruction, reconstruct only part of the projection "
                            "data, 0 means reconstruct up to dimy.");

    // Program flow parameters
    addForceArgs();
    addCLSettingsArgs();
    addRelaxedArg();
}

int Args::postParse()
{
    std::string ERR;
    int e = handleFileExistence(outputVolume, force, force);
    if(e != 0)
    {
        return e;
    }

    // Get volume size
    io::DenFileInfo inf(inputVolume);
    volumeSizeX = inf.dimx();
    volumeSizeY = inf.dimy();
    volumeSizeZ = inf.dimz();

    if(slabSize == 0)
    {
        slabSize = volumeSizeZ - slabFrom;
    }
    volumeSizeZ = slabSize;
    totalVolumeSize = uint64_t(volumeSizeX) * uint64_t(volumeSizeY) * uint64_t(slabSize);
    if(totalVolumeSize > INT_MAX)
    {
        ERR = "Implement indexing by uint64_t matrix dimension overflow of voxels count.";
        LOGW << ERR;
        // return 1;
    }
    // Ensure 32-bit float type for input
    io::DenSupportedType t = inf.getElementType();
    if(t != io::DenSupportedType::FLOAT32)
    {
        ERR = io::xprintf("This program supports FLOAT32 volumes only but the supplied "
                          "volume file %s is of type %s",
                          inputVolume.c_str(), io::DenSupportedTypeToString(t).c_str());
        LOGE << ERR;
        return -1;
    }
    parsePlatformString();
    double voxelSizeAvg = (voxelSizeX + voxelSizeY + voxelSizeZ) / 3.0;
    if(pdhg_tau < 0.0)
    {
        pdhg_tau = -pdhg_tau;
    }
    if(pdhg_sigma < 0.0)
    {
        pdhg_sigma = -pdhg_sigma * voxelSizeAvg * voxelSizeAvg;
    }
    if(pdhg_mu < 0.0)
    {
        pdhg_mu = -pdhg_mu * voxelSizeAvg * voxelSizeAvg;
    }
    return 0;
}

int main(int argc, char* argv[])
{
    using namespace KCT::util;
    Program PRG(argc, argv);
    std::string prgInfo = "OpenCL implementation of the ROF denoising algorithm using PDHG.";

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
        return 0; // Exited successfully, help message printed
    } else if(parseResult != 0)
    {
        return -1; // Exited with an error
    }

    PRG.startLog(true);
    std::string xpath = PRG.getRunTimeInfo().getExecutableDirectoryPath();

    // Initialize ROF operator
    float* volume = new float[ARG.totalVolumeSize];
    bool readxmajorvolume = true;
    io::DenFileInfo iv(ARG.inputVolume);
    iv.readIntoArray(volume, readxmajorvolume, 0, 0, 0, 0, ARG.slabFrom, ARG.slabSize);
    std::string startPath;
    startPath = io::getParent(ARG.outputVolume);
    std::string bname = io::getBasename(ARG.outputVolume);
    bname = bname.substr(0, bname.find_last_of("."));
    startPath = io::xprintf("%s/%s_", startPath.c_str(), bname.c_str());
    LOGI << io::xprintf("startpath=%s", startPath.c_str());

    std::shared_ptr<PDHGROFExecutor> pdhg = std::make_shared<PDHGROFExecutor>(
        ARG.volumeSizeX, ARG.volumeSizeY, ARG.slabSize, ARG.CLitemsPerWorkgroup);
    // pdhg > setReportingParameters(ARG.verbose, ARG.reportKthIteration, startPath);
    pdhg->initializeVolumeConvolution();
    pdhg->initializeProximal();
    pdhg->initializeGradient();
    pdhg->setGradientType(ARG.useGradientType);
    int ecd = pdhg->initializeOpenCL(ARG.CLplatformID, &ARG.CLdeviceIDs[0], ARG.CLdeviceIDs.size(),
                                     xpath, ARG.CLdebug, ARG.CLrelaxed);
    if(ecd < 0)
    {
        std::string ERR = io::xprintf("Could not initialize OpenCL platform %d.", ARG.CLplatformID);
        LOGE << ERR;
        KCTERR(ERR);
    }
    pdhg->problemSetup(ARG.voxelSizeX, ARG.voxelSizeY, ARG.voxelSizeZ);
    ecd = pdhg->initializeVolume(volume);
    if(ecd != 0)
    {
        std::string ERR = io::xprintf("OpenCL buffers initialization failed.");
        LOGE << ERR;
        KCTERR(ERR);
    }
    pdhg->reconstruct(ARG.pdhg_mu, ARG.pdhg_tau, ARG.pdhg_sigma, ARG.pdhg_theta,
                      ARG.maxIterationPDHG, ARG.stoppingRelativePDHG);
    bool volumexmajor = true;
    bool writexmajor = true;
    io::DenFileInfo::create3DDenFileFromArray(volume, volumexmajor, ARG.outputVolume,
                                              io::DenSupportedType::FLOAT32, ARG.volumeSizeX,
                                              ARG.volumeSizeY, ARG.slabSize, writexmajor);

    delete[] volume;

    return 0;
}
