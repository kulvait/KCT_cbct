#include "CArmArguments.hpp"

namespace KCT::util {

CArmArguments::CArmArguments(int argc, char** argv, std::string appName)
    : Arguments(argc, argv, appName)
{
}

inline void CArmArguments::insertDeviceID(uint32_t deviceID, uint32_t devicesOnPlatform)
{
    std::string ERR;
    if(deviceID < devicesOnPlatform)
    {
        if(std::find(CLdeviceIDs.begin(), CLdeviceIDs.end(), deviceID) == CLdeviceIDs.end())
        {
            CLdeviceIDs.push_back(deviceID);
        } else
        {
            ERR = io::xprintf("The device %d was already inserted to the CLdeviceIDs list.",
                              deviceID);
            LOGW << ERR;
        }
    } else
    {
        ERR = io::xprintf("The device %d not exist can not be in the CLdeviceIDs list.", deviceID);
        KCTERR(ERR);
    }
}

inline void CArmArguments::fillDevicesList(std::string commaSeparatedEntries, uint32_t CLplatformID)
{
    const std::regex range("(\\d+)-(\\d+)");
    const std::regex digitsRegex("(\\d+)");
    std::string ERR;
    uint32_t devicesOnPlatform = util::OpenCLManager::deviceCount(CLplatformID);
    std::stringstream entries(commaSeparatedEntries);
    std::string segment;
    while(std::getline(entries, segment, ','))
    {
        if(std::regex_match(segment, digitsRegex))
        {
            uint32_t deviceID = std::stoul(segment);
            insertDeviceID(deviceID, devicesOnPlatform);
        } else
        {
            std::smatch pieces_match;
            std::regex_match(segment, pieces_match, range);
            if(pieces_match.size() != 3)
            {
                ERR = io::xprintf("Error!");
                KCTERR(ERR);
            }
            uint32_t deviceIDstart, deviceIDend;
            deviceIDstart = std::stoul(pieces_match[1].str());
            deviceIDend = std::stoul(pieces_match[2].str());
            for(uint32_t deviceID = deviceIDstart; deviceID != deviceIDend + 1; deviceID++)
            {
                insertDeviceID(deviceID, devicesOnPlatform);
            }
        }
    }
}

uint64_t CArmArguments::parsePlatformString(bool verbose)
{
    const std::regex platformDeviceRegex("(\\d+|\\d+:((\\d+-\\d+|\\d+),)*(\\d+-\\d+|\\d+))");
    const std::regex platformOnlyRegex("\\d+");
    const std::regex platformExtendedRegex("(\\d+):([\\d,-]+)");
    // Handle empty string
    std::string ERR;
    uint32_t platformsOpenCL = util::OpenCLManager::platformCount();
    if(platformsOpenCL == 0)
    {
        ERR = io::xprintf("No OpenCL platform available to this program.");
        KCTERR(ERR);
    }
    CLdeviceIDs.clear();
    if(CLplatformString.empty())
    {
        for(uint32_t platformID = 0; platformID != platformsOpenCL; platformID++)
        {
            uint32_t devicesOnPlatform = util::OpenCLManager::deviceCount(platformID);
            if(devicesOnPlatform > 0)
            {
                CLplatformID = platformID;
                CLdeviceIDs.push_back(0);
                if(verbose)
                {
                    LOGD << io::xprintf("Selected device %d on platform %d.", 0, CLplatformID);
                }
                break;
            }
        }
    } else
    {

        std::string platformString;
        std::string deviceString;
        // Remove spaces
        CLplatformString.erase(
            std::remove_if(CLplatformString.begin(), CLplatformString.end(), ::isspace),
            CLplatformString.end());
        if(!std::regex_match(CLplatformString, platformDeviceRegex))
        {
            ERR = io::xprintf("The platform string does not match the required regexp");
            KCTERR(ERR);
        }
        if(std::regex_match(CLplatformString, platformOnlyRegex))
        {
            CLplatformID = std::stoul(CLplatformString);
            uint32_t devicesOnPlatform = util::OpenCLManager::deviceCount(CLplatformID);
            if(devicesOnPlatform > 0)
            {
                CLdeviceIDs.push_back(0);

            } else
            {
                ERR = io::xprintf("The platform %d does not contain any device!", CLplatformID);
                KCTERR(ERR);
            }
        } else
        {
            std::smatch pieces_match;
            std::regex_match(CLplatformString, pieces_match, platformExtendedRegex);
            if(pieces_match.size() != 3)
            {
                ERR = io::xprintf("Error!");
                KCTERR(ERR);
            }
            CLplatformID = std::stoul(pieces_match[1].str());
            fillDevicesList(pieces_match[2].str(), CLplatformID);
        }
    }
    uint32_t deviceCount = CLdeviceIDs.size();
    if(deviceCount == 0)
    {
        return 0;
    }
    if(verbose)
    {
        std::string str = io::xprintf("%d", CLdeviceIDs[0]);
        for(uint32_t i = 1; i < deviceCount; i++)
        {
            str = io::xprintf("%s, %d", str.c_str(), CLdeviceIDs[i]);
        }
        str = io::xprintf("%s.", str.c_str());
        LOGD << io::xprintf("Selected device%s %s on platformID %d.", deviceCount > 1 ? "s" : "",
                            str.c_str(), CLplatformID);
    }
    uint64_t localMemMaxByteSize, devMaxLoc;
    localMemMaxByteSize = util::OpenCLManager::localMemSize(CLplatformID, CLdeviceIDs[0]);
    for(uint32_t i = 0; i < deviceCount; i++)
    {
        devMaxLoc = util::OpenCLManager::localMemSize(CLplatformID, CLdeviceIDs[0]);
        if(devMaxLoc < localMemMaxByteSize)
        {
            localMemMaxByteSize = devMaxLoc;
        }
    }
    if(barrierArraySize < 0)
    {
        uint64_t localFloatSize = localMemMaxByteSize / 4;
        if(localFloatSize > 256)
        {
            barrierArraySize = localFloatSize - 256; // Space for 32 int like local variables
        } else
        {
            barrierArraySize = 1; // Space for 32 int like local variables
        }
    }
    return localMemMaxByteSize;
} // namespace KCT::util

void CArmArguments::addGeometryGroup()
{
    if(og_geometry == nullptr)
    {
        og_geometry = cliApp->add_option_group(
            "Geometry specification", "Specification of the dimensions of the CT geometry.");
    }
}

void CArmArguments::addVolumeSizeArgs(bool includeVolumeSizez)
{
    using namespace CLI;
    addGeometryGroup();
    Option* vx = og_geometry->add_option(
        "--volume-sizex", volumeSizeX,
        io::xprintf("X dimension of volume as voxel count, defaults to %d.", volumeSizeX));
    Option* vy = og_geometry->add_option(
        "--volume-sizey", volumeSizeY,
        io::xprintf("Y dimension of volume as voxel count, defaults to %d.", volumeSizeY));
    if(includeVolumeSizez)
    {
        Option* vz = og_geometry->add_option(
            "--volume-sizez", volumeSizeZ,
            io::xprintf("Z dimension of volume as voxel count, defaults to %d.", volumeSizeZ));
        vx->needs(vy)->needs(vz);
        vy->needs(vx)->needs(vz);
        vz->needs(vx)->needs(vy);
    } else
    {
        vx->needs(vy);
        vy->needs(vx);
    }
}

void CArmArguments::addProjectionSizeArgs()
{
    using namespace CLI;
    addGeometryGroup();
    Option* px = og_geometry->add_option(
        "--projection-sizex", projectionSizeX,
        io::xprintf("X dimension of detector in pixel count, defaults to %d.", projectionSizeX));
    Option* py = og_geometry->add_option(
        "--projection-sizey", projectionSizeY,
        io::xprintf("Y dimension of detector in pixel count, defaults to %d.", projectionSizeY));
    px->needs(py);
    py->needs(px);
}

void CArmArguments::addVoxelSizeArgs()
{
    using namespace CLI;
    addGeometryGroup();
    Option* vox
        = og_geometry
              ->add_option("--voxel-sizex", voxelSizeX,
                           io::xprintf("X spacing of voxels in mm, defaults to %0.2f.", voxelSizeX))
              ->check(CLI::Range(0.0, 10000.00));
    Option* voy
        = og_geometry
              ->add_option("--voxel-sizey", voxelSizeY,
                           io::xprintf("Y spacing of voxels in mm, defaults to %0.2f.", voxelSizeY))
              ->check(CLI::Range(0.0, 10000.00));
    Option* voz
        = og_geometry
              ->add_option("--voxel-sizez", voxelSizeZ,
                           io::xprintf("Z spacing of voxels in mm, defaults to %0.2f.", voxelSizeZ))
              ->check(CLI::Range(0.0, 10000.00));
    vox->needs(voy)->needs(voz);
    voy->needs(vox)->needs(voz);
    voz->needs(vox)->needs(voy);
}

void CArmArguments::addVolumeCenterArgs()
{
    using namespace CLI;
    addGeometryGroup();
    Option* vox = og_geometry
                      ->add_option(
                          "--volume-centerx", volumeCenterX,
                          io::xprintf("X coordinate of the volume center in mm, defaults to %0.2f.",
                                      volumeCenterX))
                      ->check(CLI::Range(-10000.0, 10000.0));
    Option* voy = og_geometry
                      ->add_option(
                          "--volume-centery", volumeCenterY,
                          io::xprintf("Y coordinate of the volume center in mm, defaults to %0.2f.",
                                      volumeCenterY))
                      ->check(CLI::Range(-10000.0, 10000.0));
    Option* voz = og_geometry
                      ->add_option(
                          "--volume-centerz", volumeCenterZ,
                          io::xprintf("Z coordinate of the volume center in mm, defaults to %0.2f.",
                                      volumeCenterZ))
                      ->check(CLI::Range(-10000.0, 10000.0));
    vox->needs(voy)->needs(voz);
    voy->needs(vox)->needs(voz);
    voz->needs(vox)->needs(voy);
}

void CArmArguments::addPixelSizeArgs()
{
    using namespace CLI;
    addGeometryGroup();
    Option* psx
        = og_geometry
              ->add_option(
                  "--pixel-sizex", pixelSizeX,
                  io::xprintf("X spacing of detector cells in mm, defaults to %0.3f.", pixelSizeX))
              ->check(CLI::Range(0.0, 10000.00));
    Option* psy
        = og_geometry
              ->add_option(
                  "--pixel-sizey", pixelSizeY,
                  io::xprintf("Y spacing of detector cells in mm, defaults to %0.3f.", pixelSizeY))
              ->check(CLI::Range(0.0, 10000.00));
    psx->needs(psy);
    psy->needs(psx);
}

void CArmArguments::addBasisGroup()
{
    if(og_basis == nullptr)
    {
        og_basis = cliApp->add_option_group(
            "Basis functions specification and timings.",
            "Specification of the basis functions that include definitions of the timings.");
    }
}

void CArmArguments::addBasisSpecificationArgs(bool includeBasisSize)
{
    addBasisGroup();
    CLI::Option_group* og_basis_type
        = og_basis->add_option_group("Basis type.", "Specification of the basis type.");

    og_basis_type->add_flag("--legendre", useLegendrePolynomials, "Use Legendre polynomials.");
    og_basis_type->add_flag("--chebyshev", useChebyshevPolynomials, "Use Fourier basis.");
    og_basis_type->add_flag("--fourier", useFourierBasis, "Use Fourier basis.");
    og_basis_type->add_option("--engineer", engineerBasis,
                              "Use basis that is stored in a DEN file.");
    og_basis_type->require_option(1);
    if(includeBasisSize)
    {
        og_basis->add_option("--basis-size", basisSize, "Size of the basis. Defaults to 7.")
            ->check(CLI::Range(1, 65535));
    }
    og_basis
        ->add_option("--frame-time", frame_time,
                     "Frame Time. (0018, 1063) Nominal time (in msec) per individual frame (slice) "
                     "[ms]. Might be supplied for fine tuning of the algorithm. [default is "
                     "16.8]")
        ->check(CLI::Range(0.01, 10000.0));
    og_basis
        ->add_option("-s,--pause-size", pause_size,
                     "Size of pause [ms]. This might be supplied for fine tuning of the algorithm."
                     "[default is 1171] ")
        ->check(CLI::Range(0.01, 100000.0));
    og_basis
        ->add_option(
            "-i,--start-offset", start_offset,
            "From frame_time and pause_size is computed the scan time and time of "
            "acquisition of particular frames. In reality time dynamics might apply after "
            "some delay from the acquisition of the first frame due to the mask image or "
            "contrast delay. This parameter controls the lenght of the time interval [ms] "
            "from the start of the acquisition to the time when the basis functions are "
            "used to estimate dynamics. Before this time basis functions are considered "
            "having the same value as at the beggining of their support [defaults to 0.0].")
        ->check(CLI::Range(0.0, 100000.0));
    og_basis
        ->add_option(
            "-e,--end-offset", end_offset,
            "From frame_time and pause_size is computed the scan time and time of the "
            "acquisition of particular frames. In reality we can assume that time dynamic "
            "does not affect the beginning and the end of the acquisition. This parameter "
            "controls the length of the time interval [ms] before the end of the "
            "acquisition in which all basis functions are considered having the same value "
            "as at the end of their support [defaults to 0.0].")
        ->check(CLI::Range(0.0, 100000.0));
}

void CArmArguments::addSettingsGroup()
{
    if(getRegisteredOptionGroup("settings") == nullptr)
    {
        registerOptionGroup(
            "settings",
            cliApp->add_option_group("Settings", "Settings of the program and used algorithms."));
    }
    og_settings = getRegisteredOptionGroup("settings");
}

void CArmArguments::addCLSettingsGroup()
{
    addSettingsGroup();
    if(og_cl_settings == nullptr)
    {
        og_cl_settings
            = og_settings->add_option_group("CL settings", "Setting of the OpenCL computing.");
    }
}

void CArmArguments::addProjectorSettingsGroups()
{
    addSettingsGroup();
    if(og_projectorsettings == nullptr)
    {
        og_projectorsettings = og_settings->add_option_group(
            "Projector settings", "Configuration of projectors and backprojectors.");
    }
    if(og_projectortypesettings == nullptr)
    {
        og_projectortypesettings
            = og_projectorsettings->add_option_group("Projector type", "Select projector type.");
        og_projectortypesettings->require_option(1);
    }
}

void CArmArguments::addSettingsArgs()
{
    addSettingsGroup();
    og_settings
        ->add_option("--report-kth-intermediate", reportKthIteration,
                     io::xprintf("Report each k-th iteration 0 for no reports, defaults to %d.",
                                 reportKthIteration))
        ->check(CLI::Range(0, 100));
    og_settings
        ->add_option(
            "--max-iterations", maxIterationCount,
            io::xprintf("Maximum number of iterations, defaults to %d.", maxIterationCount))
        ->check(CLI::Range(1, 65535));
    og_settings
        ->add_option("--stopping-relative-error", stoppingRelativeError,
                     io::xprintf("Stopping relative error of ||Ax-b||/||b||, defaults to %f.",
                                 stoppingRelativeError))
        ->check(CLI::Range(0.0, 1.0));
    og_settings
        ->add_option(
            "--max-iterations-pdhg", maxIterationPDHG,
            io::xprintf("Maximum number of PDHG iterations, defaults to %d.", maxIterationPDHG))
        ->check(CLI::Range(1, 65535));
    og_settings
        ->add_option("--stopping-relative-error-pdhg", stoppingRelativePDHG,
                     io::xprintf("Stopping relative error of PDHG, defaults to %f.",
                                 stoppingRelativePDHG))
        ->check(CLI::Range(0.0, 1.0));
}

void CArmArguments::addCLSettingsArgs()
{
    addCLSettingsGroup();
    og_cl_settings->add_option(
        "-p,--platform_id", CLplatformString,
        io::xprintf("OpenCL platform and device IDs to use, can be 0:1 or 0:0-5, defaults to %s.",
                    CLplatformString.c_str()));
    std::string debugValue = (CLdebug ? "true" : "false");
    og_cl_settings->add_flag(
        "-d,--debug", CLdebug,
        io::xprintf("OpenCL compilation including debugging information, defaults to %s.",
                    debugValue.c_str()));
    og_cl_settings
        ->add_option(
            "--items-per-workgroup", CLitemsPerWorkgroup,
            io::xprintf("OpenCL parameter that is important for norm computation, defaults to %d.",
                        CLitemsPerWorkgroup))
        ->check(CLI::Range(1, 65535));
}

void CArmArguments::addCuttingVoxelProjectorArgs(bool includeNoScaling)
{
    addProjectorSettingsGroups();
    std::string optValue;
    CLI::Option* optCVP;
    CLI::Option* optExactScaling;
    CLI::Option* optCosScaling;
    CLI::Option* optWithoutScaling;
    CLI::Option* optBarier;
    optValue = (useCVPProjector ? "true" : "false");
    optCVP = og_projectortypesettings->add_flag(
        "--cvp", useCVPProjector,
        io::xprintf("Use Cutting voxel projector, defaults to %s.", optValue.c_str()));
    if(includeNoScaling)
    {
        optValue = (useExactScaling ? "true" : "false");
        optExactScaling = og_projectorsettings->add_flag(
            "--exact-scaling", useExactScaling,
            io::xprintf("Use exact scaling as an oposite to cos scaling, defaults to %s.",
                        optValue.c_str()));
        optValue = (useExactScaling ? "true" : "false");
        optCosScaling = og_projectorsettings->add_flag(
            "--cos-scaling", useCosScaling,
            io::xprintf("Use exact scaling as an oposite to cos scaling, defaults to %s.",
                        optValue.c_str()));
        optValue = (useNoScaling ? "true" : "false");
        optWithoutScaling = og_projectorsettings->add_flag(
            "--without-scaling", useNoScaling,
            io::xprintf("Use no scaling in CVP, defaults to %s. FOR DEBUG ONLY!",
                        optValue.c_str()));
        optWithoutScaling->needs(optCVP);
        optWithoutScaling->excludes(optExactScaling);
        optWithoutScaling->excludes(optCosScaling);
        optCosScaling->excludes(optWithoutScaling);
        optExactScaling->excludes(optWithoutScaling);
        optCosScaling->needs(optCVP);
        optCosScaling->excludes(optExactScaling);
        optExactScaling->excludes(optCosScaling);
    } else
    {
        optValue = (useExactScaling ? "true" : "false");
        optExactScaling = og_projectorsettings->add_flag(
            "--exact-scaling,!--cos-scaling", useExactScaling,
            io::xprintf("Use exact scaling as an oposite to cos scaling, defaults to %s.",
                        optValue.c_str()));
    }
    optExactScaling->needs(optCVP);
    optValue = (useBarrierCalls ? "true" : "false");
    optBarier = og_projectorsettings->add_flag(
        "--barrier,!--no-barrier", useBarrierCalls,
        io::xprintf("Use barrier calls for CVP, defaults to %s.", optValue.c_str()));
    optBarier->needs(optCVP);
    opt_cl_localarraysize = og_projectorsettings->add_option(
        "--local-array-size", barrierArraySize,
        io::xprintf("Size of LOCALARRAYSIZE for barrier calls, defaults to %d.", barrierArraySize));
    opt_cl_localarraysize->needs(optCVP);
    std::string optstr
        = io::xprintf("Elevation correction for CVP projector, increase excactness for high "
                      "elevation angles. [defaults to %s]",
                      useElevationCorrection ? "true" : "false");
    CLI::Option* elevationCorrection_cli
        = og_projectorsettings->add_flag("--elevation-correction", useElevationCorrection, optstr);
    elevationCorrection_cli->needs(optCVP);
}

void CArmArguments::addTTProjectorArgs()
{
    addProjectorSettingsGroups();
    std::string optValue;
    optValue = (useTTProjector ? "true" : "false");
    og_projectortypesettings->add_flag(
        "--tt", useTTProjector,
        io::xprintf("Use TT projector with A3 amplitude and adjoint backprojector pair instead of "
                    "cuting voxel projector, defaults to %s.",
                    optValue.c_str()));
}

void CArmArguments::addSidonProjectorArgs()
{
    addProjectorSettingsGroups();
    std::string optValue;
    CLI::Option* optSid;
    CLI::Option* optPPE;
    optValue = (useSidonProjector ? "true" : "false");
    optSid = og_projectortypesettings->add_flag(
        "--siddon", useSidonProjector,
        io::xprintf("Use Siddon projector and backprojector pair instead of "
                    "cuting voxel projector, defaults to %s.",
                    optValue.c_str()));
    optPPE = og_projectorsettings
                 ->add_option("--probes-per-edge", probesPerEdge,
                              io::xprintf("Number of probes in each pixel edge in Sidon raycaster, "
                                          "complexity scales with the "
                                          "square of this number. Defaults to %d.",
                                          probesPerEdge))
                 ->check(CLI::Range(1, 1000));
    optPPE->needs(optSid);
}

void CArmArguments::addCenterVoxelProjectorArgs()
{
    addProjectorSettingsGroups();
    std::string optValue;
    optValue = (useCenterVoxelProjector ? "true" : "false");
    og_projectortypesettings->add_flag(
        "--center-voxel-projector", useCenterVoxelProjector,
        io::xprintf(
            "Use center voxel projector to approximate voxel by its center, defaults to %s.",
            optValue.c_str()));
}

void CArmArguments::addProjectorArgs()
{
    addCuttingVoxelProjectorArgs(false);
    addTTProjectorArgs();
    addSidonProjectorArgs();
}

void CArmArguments::addBackprojectorScalingArgs()
{
    using namespace CLI;
    addProjectorSettingsGroups();

    CLI::Option_group* op_bpx = og_projectorsettings->add_option_group(
        "Backprojector scaling",
        "Scaling method for backprojection output, which can modify A^T output.");
    op_bpx->add_flag(
        "--backprojector-no-scaling", backprojectorNoScaling,
        io::xprintf("No scaling applied to A^T, defaults scaling."));
    op_bpx->add_flag("--backprojector-fbp-scaling", backprojectorFBPScaling,
                     io::xprintf("Perform fbp scaling of backprojected values, which is normally "
                                 "used when performing FBP. If true scaling "
                                 "factor of pi/pdimz is used"));
    op_bpx->add_flag(
        "--backprojector-natural-scaling", backprojectorNaturalScaling,
        io::xprintf("Perform natural scaling of backprojected values, which is close to "
                    "backsmearing process. Vector b is divided by 1/A(1) and this vector is "
                    "backprojected with scaling factor 1/pdimz."));
    op_bpx->add_flag(
        "--backprojector-kaczmarz-scaling", backprojectorKaczmarzScaling,
        io::xprintf("Perform kaczmarz scaling of backprojected values, which means divide b by "
                    "diag(A A^T) and then apply A^T operator scaled by 1/pdimz. "));
    op_bpx->require_option(0, 1);
}

void CArmArguments::addProjectorLocalNDRangeArgs()
{
    addCLSettingsGroup();
    std::string defaultValue = io::xprintf("NDRange(%d, %d, %d)", projectorLocalNDRange[0],
                                           projectorLocalNDRange[1], projectorLocalNDRange[2]);
    opt_cl_projectorlocalrange = og_cl_settings->add_option(
        "--projector-local-range", projectorLocalNDRange,
        io::xprintf("Specify local NDRange for projector, (0,1,1) is a special value to guess, "
                    "(0,0,0) means NDRange(), defaults to %s.",
                    defaultValue.c_str()));
    opt_cl_projectorlocalrange->expected(3);
}
void CArmArguments::addBackprojectorLocalNDRangeArgs()
{
    addCLSettingsGroup();
    std::string defaultValue
        = io::xprintf("NDRange(%d, %d, %d)", backprojectorLocalNDRange[0],
                      backprojectorLocalNDRange[1], backprojectorLocalNDRange[2]);
    opt_cl_backprojectorlocalrange = og_cl_settings->add_option(
        "--backprojector-local-range", backprojectorLocalNDRange,
        io::xprintf("Specify local NDRange for backprojector, (0,1,1) is a special value to guess, "
                    "(0,0,0) means NDRange(), defaults to %s.",
                    defaultValue.c_str()));
    opt_cl_backprojectorlocalrange->expected(3);
}
void CArmArguments::addRelaxedArg()
{
    addCLSettingsGroup();
    std::string defaultValue = (CLrelaxed ? "true" : "false");
    og_cl_settings->add_flag(
        "--relaxed,!--no-relaxed", CLrelaxed,
        io::xprintf("OpenCL define RELAXED, defaults to %s.", defaultValue.c_str()));
}

} // namespace KCT::util
