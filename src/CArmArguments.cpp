#include "CArmArguments.hpp"

namespace CTL::util {

CArmArguments::CArmArguments(int argc, char** argv, std::string appName)
    : Arguments(argc, argv, appName)
{
}

void CArmArguments::parsePlatformString()
{
    std::string ERR;
    // Remove spaces
    CLplatformString.erase(
        std::remove_if(CLplatformString.begin(), CLplatformString.end(), ::isspace),
        CLplatformString.end());
    // Separate it to two parts, before : and after
    std::list<std::string> platform_device_list;
    strtk::parse(CLplatformString, ":", platform_device_list);
    // Only platform, device defaults to 0
    CLdeviceIDs.clear();
    if(platform_device_list.size() == 1)
    {
        CLplatformID = std::stoi(CLplatformString.c_str());
        CLdeviceIDs.push_back(0);

    } else if(platform_device_list.size() == 2)
    {
        CLplatformID = std::stoi(platform_device_list.front().c_str());
        std::list<std::string> string_list;
        strtk::parse(platform_device_list.back(), ",", string_list);
        auto it = string_list.begin();
        while(it != string_list.end())
        {
            size_t numRangeSigns = std::count(it->begin(), it->end(), '-');
            if(numRangeSigns > 1)
            {
                std::string msg = io::xprintf("Wrong number of range specifiers in the string %s.",
                                              (*it).c_str());
                LOGE << msg;
                throw std::runtime_error(msg);
            } else if(numRangeSigns == 1)
            {
                std::vector<int> int_vector;
                strtk::parse((*it), "-", int_vector);
                if(0 <= int_vector[0] && int_vector[0] <= int_vector[1])
                {
                    for(int k = int_vector[0]; k != int_vector[1] + 1; k++)
                    {
                        CLdeviceIDs.push_back(k);
                    }
                } else
                {
                    std::string msg
                        = io::xprintf("String %s is invalid range specifier.", (*it).c_str());
                    LOGE << msg;
                    throw std::runtime_error(msg);
                }
            } else
            {
                int index = std::stoi(it->c_str());
                if(0 <= index)
                {
                    CLdeviceIDs.push_back(index);
                } else
                {
                    std::string msg = io::xprintf("Negative identifier!.", (*it).c_str());
                    LOGE << msg;
                    throw std::runtime_error(msg);
                }
            }
            it++;
        }
    } else
    {
        ERR = io::xprintf("Only one : delimiter is recognized but %d was provided",
                          platform_device_list.size() - 1);
        LOGE << ERR;
        throw std::runtime_error(ERR);
    }
    LOGD << io::xprintf("Selected %d devices on platformID %d.", CLdeviceIDs.size(), CLplatformID);
    std::string str = io::xprintf("DeviceIDs:%d", CLdeviceIDs[0]);
    for(uint32_t i = 1; i != CLdeviceIDs.size(); i++)
    {
        str = io::xprintf("%s,%d", str.c_str(), CLdeviceIDs[i]);
    }
    str = io::xprintf("%s.", str.c_str());
    LOGD << str;
} // namespace CTL::util

void CArmArguments::addGeometryGroup()
{
    if(og_geometry == nullptr)
    {
        og_geometry = cliApp->add_option_group(
            "Geometry specification", "Specification of the dimensions of the CT geometry.");
    }
}

void CArmArguments::addVolumeSizeArgs()
{
    using namespace CLI;
    addGeometryGroup();
    Option* vx = og_geometry->add_option(
        "--volume-sizex", volumeSizeX,
        io::xprintf("X dimension of volume as voxel count, defaults to %d.", volumeSizeX));
    Option* vy = og_geometry->add_option(
        "--volume-sizey", volumeSizeY,
        io::xprintf("Y dimension of volume as voxel count, defaults to %d.", volumeSizeY));
    Option* vz = og_geometry->add_option(
        "--volume-sizez", volumeSizeZ,
        io::xprintf("Z dimension of volume as voxel count, defaults to %d.", volumeSizeZ));
    vx->needs(vy)->needs(vz);
    vy->needs(vx)->needs(vz);
    vz->needs(vx)->needs(vy);
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
            io::xprintf("Maximum number of LSQR iterations, defaults to %d.", maxIterationCount))
        ->check(CLI::Range(1, 65535));
    og_settings
        ->add_option("--stopping-relative-error", stoppingRelativeError,
                     io::xprintf("Stopping relative error of ||Ax-b||/||b||, defaults to %f.",
                                 stoppingRelativeError))
        ->check(CLI::Range(0.0, 1.0));

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

void CArmArguments::addProjectorArgs()
{
    addSettingsGroup();
    std::string optValue;
    CLI::Option* optSid;
    CLI::Option* optPPE;
    CLI::Option* optTT;
    CLI::Option* optCVP;
    CLI::Option* optExactScaling;
    optValue = (useSidonProjector ? "true" : "false");
    optSid
        = og_settings->add_flag("--sidon", useSidonProjector,
                                io::xprintf("Use sidon projector and backprojector pair instead of "
                                            "cuting voxel projector, defaults to %s.",
                                            optValue.c_str()));
    optPPE = og_settings
                 ->add_option("--probes-per-edge", probesPerEdge,
                              io::xprintf("Number of probes in each pixel edge in Sidon raycaster, "
                                          "complexity scales with the "
                                          "square of this number. Defaults to %d.",
                                          probesPerEdge))
                 ->check(CLI::Range(1, 1000));

    optValue = (useTTProjector ? "true" : "false");
    optTT = og_settings->add_flag(
        "--tt", useTTProjector,
        io::xprintf("Use TT projector with A3 amplitude and adjoint backprojector pair instead of "
                    "cuting voxel projector, defaults to %s.",
                    optValue.c_str()));
    optValue = (useCVPProjector ? "true" : "false");
    optCVP = og_settings->add_flag(
        "--cvp", useTTProjector,
        io::xprintf("Use Cutting voxel projector, defaults to %s.", optValue.c_str()));
    optValue = (useExactScaling ? "true" : "false");
    optExactScaling = og_settings->add_flag(
        "--exact-scaling,!--cos-scaling", useExactScaling,
        io::xprintf("Use exact scaling as an oposite to cos scaling, defaults to %s.",
                    optValue.c_str()));
    optSid->excludes(optTT);
    optSid->excludes(optCVP);
    optPPE->needs(optSid);
    optTT->excludes(optSid);
    optTT->excludes(optCVP);
    optCVP->excludes(optSid);
    optCVP->excludes(optTT);
    optExactScaling->needs(optCVP);
}

} // namespace CTL::util
