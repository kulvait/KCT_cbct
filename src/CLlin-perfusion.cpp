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
#include "ARGPARSE/parseArgs.h"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenProjectionMatrixReader.hpp"
#include "DEN/DenSupportedType.hpp"
#include "FUN/FourierSeries.hpp"
#include "FUN/LegendrePolynomialsExplicit.hpp"
#include "FUN/StepFunction.hpp"
#include "GLSQRPerfusionReconstructor.hpp"
#include "RunTimeInfo.hpp"

using namespace CTL;

/**Arguments parsed by the main function.
 */
struct Args
{
    // Output files
    std::string outputVolumePrefix;
    // Input files
    std::string inputProjectionMatrices;
    std::vector<std::string> inputProjections;
    // Geometry
    uint32_t projectionSizeX = 616;
    uint32_t projectionSizeY = 480;
    uint32_t projectionSizeZ;
    uint64_t totalProjectionsSize;
    double pixelSizeX = 0.616;
    double pixelSizeY = 0.616;
    double voxelSizeX = 1.0;
    double voxelSizeY = 1.0;
    double voxelSizeZ = 1.0;
    uint32_t volumeSizeX = 256;
    uint32_t volumeSizeY = 256;
    uint32_t volumeSizeZ = 199;
    uint64_t totalVolumeSize;
    // Basis and timings
    bool useLegendrePolynomials = false;
    bool useFourierBasis = false;
    std::string engineerBasis = "";
    uint32_t degree = 7;
    float pause_size = 1171;
    float frame_time = 16.8;
    uint32_t platformId = 0;
    bool debug = false;
    bool reportIntermediate = false;
    // It is evaluated from -0.5, pixels are centerred at integer coordinates
    uint32_t baseOffset = 0;
    uint32_t maxIterations = 100;
    double stoppingError = 0.00025;
    bool noFrameOffset = false;
    bool force = false;
    uint32_t itemsPerWorkgroup = 256;
    float start_offset = 0.0, end_offset = 0.0;

    /** Frame Time. (0018, 1063) Nominal time (in msec) per individual frame.
     *
     *The model assumes that there is delay between two consecutive frames of the frame_time.
     *First frame is aquired directly after pause. From DICOM it is 16.6666667ms. From
     *experiment 16.8ms.
     */
    /**Fourier functions degree*/

    int parseArguments(int argc, char* argv[]);
    std::string getVolumeName(uint32_t baseIND);
};

std::string Args::getVolumeName(uint32_t baseIND)
{
    if(baseIND >= degree)
    {
        LOGW << io::xprintf("Constructing %d-th volume name that probably won't be used due to "
                            "the size of the basis of %d elements.",
                            baseIND, degree);
    }
    std::string f = io::xprintf("%s_reconstructed%d", outputVolumePrefix.c_str(), baseIND);
    return f;
}

/**Argument parsing
 *
 * @param argc
 * @param argv[]
 *
 * @return Returns 0 on success and nonzero for some error.
 */
int Args::parseArguments(int argc, char* argv[])
{

    CLI::App app{ "OpenCL implementation of GLSQR applied on the perfusion operator." };
    app.add_option(
           "output_volume_pattern", outputVolumePrefix,
           "Output: pattern of volumes to reconstruct, PATTERN_reconstructed${i} will be used for "
           "output files, where i is a index of the basis element.")
        ->required();
    app.add_option("input_projection_matrices", inputProjectionMatrices,
                   "Input: projection matrices to specify cone beam geometry.")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("projection_files", inputProjections,
                   "Input: projection files in a DEN format. All of them must be orderred "
                   "according to the projection_matrices file. Therefore they need to "
                   "be reversed for backward sweep.")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_flag("--force", force,
                 "Overwrite output files given by output_volume_pattern if they exist.");

    // Reconstruction geometry
    CLI::Option_group* og_rec = app.add_option_group(
        "Reconstruction geometry", "Parameters that define reconstruction geometry.");
    CLI::Option* psx = og_rec->add_option("--pixel-sizex", pixelSizeX,
                                          "Spacing of detector cells, defaults to 0.616.");
    CLI::Option* psy = og_rec->add_option("--pixel-sizey", pixelSizeY,
                                          "Spacing of detector cells, defaults to 0.616.");
    CLI::Option* vx = og_rec->add_option("--volume-sizex", volumeSizeX,
                                         "Dimension of volume, defaults to 256.");
    CLI::Option* vy = og_rec->add_option("--volume-sizey", volumeSizeY,
                                         "Dimension of volume, defaults to 256.");
    CLI::Option* vz = og_rec->add_option("--volume-sizez", volumeSizeZ,
                                         "Dimension of volume, defaults to 199.");
    og_rec->add_option("--voxel-sizex", voxelSizeX, "Spacing of voxels, defaults to 1.0.");
    og_rec->add_option("--voxel-sizey", voxelSizeY, "Spacing of voxels, defaults to 1.0.");
    og_rec->add_option("--voxel-sizez", voxelSizeZ, "Spacing of voxels, defaults to 1.0.");
    psx->needs(psy);
    psy->needs(psx);
    vx->needs(vy)->needs(vz);
    vy->needs(vx)->needs(vz);
    vz->needs(vx)->needs(vy);

    // Specification of the basis of the volume data, each voxel is approximated as v_i(t) =  sum
    // v_i^j b_j(t).
    CLI::Option_group* og_basis = app.add_option_group(
        "Basis functions specification and timings.",
        "Specification of the basis functions that include definitions of the timings.");
    CLI::Option_group* og_basis_type = og_basis->add_option_group(
        "Basis type.",
        "Specification of the basis type.");
	
    CLI::Option* opt_legendre
        = og_basis_type->add_flag("--legendre", useLegendrePolynomials, "Use Legendre polynomials.");
    CLI::Option* opt_fourier
        = og_basis_type->add_flag("--fourier", useFourierBasis, "Use Fourier basis.");
    CLI::Option* opt_engineer = og_basis_type->add_option("--engineer", engineerBasis,
                                                     "Use basis that is stored in a DEN file.");
	og_basis_type->require_option(1);
    og_basis->add_option("--degree", degree, "Size of the basis. Defaults to 7.")
        ->check(CLI::Range(1, 65535));
    opt_legendre->excludes(opt_fourier);
    opt_legendre->excludes(opt_engineer);
    opt_fourier->excludes(opt_legendre);
    opt_fourier->excludes(opt_engineer);
    opt_engineer->excludes(opt_legendre);
    opt_engineer->excludes(opt_fourier);
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
    // Program flow parameters
    app.add_option("--max-iterations", maxIterations,
                   "Maximum number of LSQR iterations, defaults to 100.")
        ->check(CLI::Range(1, 65535))
        ->group("Platform settings");
    app.add_option("-p,--platform_id", platformId, "OpenCL platform ID to use.")
        ->check(CLI::Range(0, 65535))
        ->group("Platform settings");
    app.add_flag("-d,--debug", debug, "OpenCL compilation including debugging information.")
        ->group("Platform settings");
    app.add_option("--items-per-workgroup", itemsPerWorkgroup,
                   "OpenCL parameter that is important for norm computation, defaults to 256.")
        ->check(CLI::Range(1, 65535))
        ->group("Platform settings");
    app.add_flag("--report-intermediate", reportIntermediate,
                 "Report intermediate values of x, defaults to false.")
        ->group("Platform settings");
    try
    {
        app.parse(argc, argv);
        // If force is not set, then check if output file does not exist
        if(!force)
        {
            std::string f;
            for(uint32_t i = 0; i != degree; i++)
            {
                f = getVolumeName(i);
                if(io::pathExists(f))
                {
                    std::string msg = io::xprintf(
                        "Error: output file %f already exists, use --force to force overwrite it.",
                        f.c_str());
                    LOGE << msg;
                    return 1;
                }
            }
        }
        // How many projection matrices is there in total
        io::DenFileInfo pmi(inputProjectionMatrices);
        io::DenFileInfo inf(inputProjections[0]);
        projectionSizeX = inf.dimx();
        projectionSizeY = inf.dimy();
        projectionSizeZ = inf.dimz();
        totalVolumeSize = uint64_t(volumeSizeX) * uint64_t(volumeSizeY) * uint64_t(volumeSizeZ);
        totalProjectionsSize
            = uint64_t(projectionSizeX) * uint64_t(projectionSizeY) * uint64_t(projectionSizeZ);
        if(inf.dimz() != pmi.dimz())
        {
            std::string ERR = io::xprintf(
                "Projection matrices z dimension %d is different from projections z dimension %d.",
                pmi.dimz(), inf.dimz());
            LOGE << ERR;
            io::throwerr(ERR);
        }
        if(totalVolumeSize > INT_MAX)
        {
            io::throwerr(
                "Implement indexing by uint64_t matrix dimension overflow of voxels count.");
        }
        // End parsing arguments
        if(totalProjectionsSize > INT_MAX)
        {
            io::throwerr("Implement indexing by uint64_t matrix dimension overflow of projection "
                         "pixels count.");
        }
        io::DenSupportedType t = inf.getDataType();
        if(t != io::DenSupportedType::float_)
        {
            std::string ERR
                = io::xprintf("This program supports float projections only but the supplied "
                              "projection file %s is "
                              "of type %s",
                              inputProjections[0].c_str(), io::DenSupportedTypeToString(t).c_str());
            LOGE << ERR;
            io::throwerr(ERR);
        }
    } catch(const CLI::CallForHelp e)
    {
        app.exit(e); // Prints help message
        return 1;
    } catch(const CLI::ParseError& e)
    {
        int exitcode = app.exit(e);
        LOGE << io::xprintf("There was perse error with exit code %d catched.\n %s", exitcode,
                            app.help().c_str());
        return -1;
    } catch(...)
    {
        LOGE << "Unknown exception catched";
        return -1;
    }
    return 0;
}

int main(int argc, char* argv[])
{
    plog::Severity verbosityLevel = plog::debug; // debug, info, ...
    io::RunTimeInfo rti(argc, argv);
    std::string csvLogFile = io::xprintf(
        "/tmp/%s.csv", rti.getExecutableName().c_str()); // Set to empty string to disable
    std::string xpath = rti.getExecutableDirectoryPath();
    bool logToConsole = true;
    plog::PlogSetup plogSetup(verbosityLevel, csvLogFile, logToConsole);
    plogSetup.initLogging();
    auto start = std::chrono::steady_clock::now();
    LOGI << io::xprintf("START %s", argv[0]);
    // Argument parsing
    Args a;
    int parseResult = a.parseArguments(argc, argv);
    if(parseResult != 0)
    {
        if(parseResult > 0)
        {
            return 0; // Exited sucesfully, help message printed
        } else
        {
            return -1; // Exited somehow wrong
        }
    }
    std::shared_ptr<io::DenProjectionMatrixReader> dr
        = std::make_shared<io::DenProjectionMatrixReader>(a.inputProjectionMatrices);
    float* projection;
    float* basisVals;
    float* vals = new float[a.degree];

    std::vector<float*> projections;
    std::vector<float*> basisFunctionsValues;
    double mean_sweep_time = (a.projectionSizeZ - 1) * a.frame_time + a.pause_size;
    double startTime = a.start_offset;
    double endTime = (a.inputProjections.size() - 1) * mean_sweep_time
        + (a.projectionSizeZ - 1) * a.frame_time - a.end_offset;

    // Basis set initialization
    std::shared_ptr<util::VectorFunctionI> baseFunctionsEvaluator;
    if(a.useFourierBasis)
    {
        baseFunctionsEvaluator
            = std::make_shared<util::FourierSeries>(a.degree, startTime, endTime);
    } else if(a.useLegendrePolynomials)
    {
        baseFunctionsEvaluator
            = std::make_shared<util::LegendrePolynomialsExplicit>(a.degree - 1, startTime, endTime);
    } else
    {
        int numberOfFunctions = io::DenFileInfo(a.engineerBasis).dimz();
        baseFunctionsEvaluator = std::make_shared<util::StepFunction>(
            a.degree, a.engineerBasis, numberOfFunctions, startTime, endTime);
    }
    for(std::size_t j = 0; j != a.degree; j++)
    {
        basisVals = new float[a.projectionSizeZ * a.inputProjections.size()];
        basisFunctionsValues.push_back(basisVals);
    }
    for(std::size_t sweepID = 0; sweepID != a.inputProjections.size(); sweepID++)
    {
        projection = new float[a.totalProjectionsSize];
        io::readBytesFrom(a.inputProjections[sweepID], 6, (uint8_t*)projection,
                          a.totalProjectionsSize * 4);
        projections.push_back(projection);
        // Values of basis function in the time that corresponds to i-th sweep and j-th angle
        for(std::size_t angleID = 0; angleID != a.projectionSizeZ; angleID++)
        {
            if(sweepID % 2 == 0)
            {
                baseFunctionsEvaluator->valuesAt(sweepID * mean_sweep_time + angleID * a.frame_time,
                                                 vals);
            } else
            {
                baseFunctionsEvaluator->valuesAt(
                    sweepID * mean_sweep_time + (a.projectionSizeZ - 1 - angleID) * a.frame_time,
                    vals);
            }
            for(std::size_t basisIND = 0; basisIND != a.degree; basisIND++)
            {
                basisFunctionsValues[basisIND][sweepID * a.projectionSizeZ + angleID]
                    = vals[basisIND];
            }
        }
    }
#ifdef DEBUG
    std::vector<double> taxis;
    std::vector<std::vector<double>> values;
    for(uint32_t j = 0; j != a.degree; j++)
    {
        values.push_back(std::vector<double>());
    }

    for(std::size_t i = 0; i != a.inputProjections.size(); i++)
    {
        for(std::size_t j = 0; j != a.projectionSizeZ; j++)
        {
            double time;

            if(i % 2 == 0)
            {
                time = i * mean_sweep_time + j * a.frame_time;
            } else
            {
                time = i * mean_sweep_time + (a.projectionSizeZ - 1 - j) * a.frame_time;
            }
            taxis.push_back(time);
            for(std::size_t k = 0; k != a.degree; k++)
            {
                values[k].push_back(basisFunctionsValues[k][i * a.projectionSizeZ + j]);
            }
        }
        // Put discontinuity here
        taxis.push_back(0.0);
        for(std::size_t k = 0; k != a.degree; k++)
        {
            values[k].push_back(std::numeric_limits<double>::quiet_NaN());
        }
    }
    for(uint32_t j = 0; j != a.degree; j++)
    {
        plt::named_plot(io::xprintf("Function %d", j), taxis, values[j]);
    }
    plt::legend();
    plt::show();
#endif
    std::string startPath = io::getParent(a.outputVolumePrefix);
    std::string bname = io::getBasename(a.outputVolumePrefix);
    bname = bname.substr(0, bname.find_last_of("."));
    startPath = io::xprintf("%s/%s_", startPath.c_str(), bname.c_str());
    LOGI << io::xprintf("startpath=%s", startPath.c_str());
    LOGI << io::xprintf("Dimensions are [%d %d %d]", a.volumeSizeX, a.volumeSizeY, a.volumeSizeZ);
    std::shared_ptr<GLSQRPerfusionReconstructor> LSQR
        = std::make_shared<GLSQRPerfusionReconstructor>(
            a.projectionSizeX, a.projectionSizeY, a.projectionSizeZ, a.pixelSizeX, a.pixelSizeY,
            a.volumeSizeX, a.volumeSizeY, a.volumeSizeZ, a.voxelSizeX, a.voxelSizeY, a.voxelSizeZ,
            xpath, a.debug, a.itemsPerWorkgroup, a.reportIntermediate, startPath);
    int res = LSQR->initializeOpenCL(a.platformId);
    if(res < 0)
    {
        std::string ERR = io::xprintf("Could not initialize OpenCL platform %d.", a.platformId);
        LOGE << ERR;
        io::throwerr(ERR);
    }
    std::vector<float*> volumes;
    float* volume;
    for(std::size_t basisIND = 0; basisIND != a.degree; basisIND++)
    {
        volume = new float[a.totalVolumeSize]();
        volumes.push_back(volume);
    }
    // testing
    //    io::readBytesFrom("/tmp/X.den", 6, (uint8_t*)volume, a.totalVolumeSize * 4);

    LSQR->initializeData(projections, basisFunctionsValues, volumes);
    LSQR->reconstruct(dr, a.maxIterations);

    uint16_t buf[3];
    buf[0] = a.volumeSizeY;
    buf[1] = a.volumeSizeX;
    buf[2] = a.volumeSizeZ;
    std::string f;
    for(std::size_t i = 0; i != a.degree; i++)
    {
        f = a.getVolumeName(i);
        LOGI << io::xprintf("Writing out file %s of the size [%d %d %d].", f.c_str(), a.volumeSizeX,
                            a.volumeSizeY, a.volumeSizeZ);
        io::createEmptyFile(f, 0, true);
        io::appendBytes(f, (uint8_t*)buf, 6);
        io::appendBytes(f, (uint8_t*)volumes[i], a.totalVolumeSize * sizeof(float));
        delete[] volumes[i];
        delete[] basisFunctionsValues[i];
    }
    for(std::size_t i = 0; i < a.inputProjections.size(); i++)
    {
        delete[] projections[i];
    }
	delete[] vals;
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
    LOGI << io::xprintf("END %s, duration %d ms.", argv[0], duration.count());
}
