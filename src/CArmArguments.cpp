#include "CArmArguments.hpp"

namespace CTL::util {

CArmArguments::CArmArguments(int argc, char** argv, std::string appName)
    : Arguments(argc, argv, appName)
{
    og_geometry = cliApp->add_option_group("Geometry specification",
                                           "Specification of the dimensions of the CT geometry.");
}

void CArmArguments::addVolumeSizeArgs()
{
    using namespace CLI;
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

void CArmArguments::addBasisSpecificationArgs(bool includeBasisSize)
{
    CLI::Option_group* og_basis = cliApp->add_option_group(
        "Basis functions specification and timings.",
        "Specification of the basis functions that include definitions of the timings.");
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

} // namespace CTL::util
