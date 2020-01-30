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
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenProjectionMatrixReader.hpp"
#include "DEN/DenSupportedType.hpp"
#include "FUN/ChebyshevPolynomialsExplicit.hpp"
#include "FUN/FourierSeries.hpp"
#include "FUN/LegendrePolynomialsExplicit.hpp"
#include "FUN/StepFunction.hpp"
#include "GLSQRPerfusionReconstructor.hpp"
#include "PROG/Program.hpp"
#include "PROG/RunTimeInfo.hpp"

using namespace CTL;
using namespace CTL::util;

/**Arguments parsed by the main function.
 */
class Args : public CArmArguments
{
public:
    Args(int argc, char** argv, std::string programName)
        : CArmArguments(argc, argv, programName){};
    void defineArguments();
    int preParse()
    {
        maxIterationCount = 100;
        stoppingRelativeError = 0.00025;
        return 0;
    };
    int postParse()
    {
        if(!force)
        {
            std::string f;
            for(uint32_t i = 0; i != basisSize; i++)
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
        return 0;
    }
    // Output files
    std::string outputVolumePrefix;
    // Input files
    std::string inputProjectionMatrices;
    std::vector<std::string> inputProjections;
    // Geometry
    uint64_t totalVolumeSize;
    uint64_t totalProjectionsSize;
    // It is evaluated from -0.5, pixels are centerred at integer coordinates
    uint32_t baseOffset = 0;
    bool noFrameOffset = false;
    bool force = false;

    /** Frame Time. (0018, 1063) Nominal time (in msec) per individual frame.
     *
     *The model assumes that there is delay between two consecutive frames of the frame_time.
     *First frame is aquired directly after pause. From DICOM it is 16.6666667ms. From
     *experiment 16.8ms.
     */
    /**Fourier functions basisSize*/

    int parseArguments(int argc, char* argv[]);
    std::string getVolumeName(uint32_t baseIND);
};

std::string Args::getVolumeName(uint32_t baseIND)
{
    if(baseIND >= basisSize)
    {
        LOGW << io::xprintf("Constructing %d-th volume name that probably won't be used due to "
                            "the size of the basis of %d elements.",
                            baseIND, basisSize);
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
void Args::defineArguments()
{
    cliApp
        ->add_option(
            "output_volume_pattern", outputVolumePrefix,
            "Output: pattern of volumes to reconstruct, PATTERN_reconstructed${i} will be used for "
            "output files, where i is a index of the basis element.")
        ->required();
    cliApp
        ->add_option("input_projection_matrices", inputProjectionMatrices,
                     "Input: projection matrices to specify cone beam geometry.")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp
        ->add_option("projection_files", inputProjections,
                     "Input: projection files in a DEN format. All of them must be orderred "
                     "according to the projection_matrices file. Therefore they need to "
                     "be reversed for backward sweep.")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp->add_flag("--force", force,
                     "Overwrite output files given by output_volume_pattern if they exist.");

    // Reconstruction geometry
    addVolumeSizeArgs();
    addVoxelSizeArgs();
    addPixelSizeArgs();
    addBasisSpecificationArgs();
    addSettingsArgs();
    addSidonArgs();

    // Specification of the basis of the volume data, each voxel is approximated as v_i(t) =  sum
    // v_i^j b_j(t).
    // Program flow parameters
}

int main(int argc, char* argv[])
{
    Program PRG(argc, argv);
    Args ARG(argc, argv, "OpenCL implementation of GLSQR applied on the perfusion operator.");
    // Argument parsing
    int parseResult = ARG.parse();
    if(parseResult > 0)
    {
        return 0; // Exited sucesfully, help message printed
    } else if(parseResult != 0)
    {
        return -1; // Exited somehow wrong
    }
    PRG.startLog(true);
    std::shared_ptr<io::DenProjectionMatrixReader> dr
        = std::make_shared<io::DenProjectionMatrixReader>(ARG.inputProjectionMatrices);
    float* projection;
    float* basisVals;
    float* vals = new float[ARG.basisSize];

    std::vector<float*> projections;
    std::vector<float*> basisFunctionsValues;
    double mean_sweep_time = (ARG.projectionSizeZ - 1) * ARG.frame_time + ARG.pause_size;
    double startTime = ARG.start_offset;
    double endTime = (ARG.inputProjections.size() - 1) * mean_sweep_time
        + (ARG.projectionSizeZ - 1) * ARG.frame_time - ARG.end_offset;

    // Basis set initialization
    std::shared_ptr<util::VectorFunctionI> baseFunctionsEvaluator;
    if(ARG.useFourierBasis)
    {
        baseFunctionsEvaluator
            = std::make_shared<util::FourierSeries>(ARG.basisSize, startTime, endTime);
    } else if(ARG.useLegendrePolynomials)
    {
        baseFunctionsEvaluator = std::make_shared<util::LegendrePolynomialsExplicit>(
            ARG.basisSize - 1, startTime, endTime);
    } else if(ARG.useChebyshevPolynomials)
    {
        baseFunctionsEvaluator = std::make_shared<util::ChebyshevPolynomialsExplicit>(
            ARG.basisSize - 1, startTime, endTime);
    } else
    {
        int numberOfFunctions = io::DenFileInfo(ARG.engineerBasis).dimz();
        baseFunctionsEvaluator = std::make_shared<util::StepFunction>(
            ARG.basisSize, ARG.engineerBasis, numberOfFunctions, startTime, endTime);
    }
    for(std::size_t j = 0; j != ARG.basisSize; j++)
    {
        basisVals = new float[ARG.projectionSizeZ * ARG.inputProjections.size()];
        basisFunctionsValues.push_back(basisVals);
    }
    for(std::size_t sweepID = 0; sweepID != ARG.inputProjections.size(); sweepID++)
    {
        projection = new float[ARG.totalProjectionsSize];
        io::readBytesFrom(ARG.inputProjections[sweepID], 6, (uint8_t*)projection,
                          ARG.totalProjectionsSize * 4);
        projections.push_back(projection);
        // Values of basis function in the time that corresponds to i-th sweep and j-th angle
        for(std::size_t angleID = 0; angleID != ARG.projectionSizeZ; angleID++)
        {
            if(sweepID % 2 == 0)
            {
                baseFunctionsEvaluator->valuesAt(
                    sweepID * mean_sweep_time + angleID * ARG.frame_time, vals);
            } else
            {
                baseFunctionsEvaluator->valuesAt(sweepID * mean_sweep_time
                                                     + (ARG.projectionSizeZ - 1 - angleID)
                                                         * ARG.frame_time,
                                                 vals);
            }
            for(std::size_t basisIND = 0; basisIND != ARG.basisSize; basisIND++)
            {
                basisFunctionsValues[basisIND][sweepID * ARG.projectionSizeZ + angleID]
                    = vals[basisIND];
            }
        }
    }
#ifdef DEBUG
    std::vector<double> taxis;
    std::vector<std::vector<double>> values;
    for(uint32_t j = 0; j != ARG.basisSize; j++)
    {
        values.push_back(std::vector<double>());
    }

    for(std::size_t i = 0; i != ARG.inputProjections.size(); i++)
    {
        for(std::size_t j = 0; j != ARG.projectionSizeZ; j++)
        {
            double time;

            if(i % 2 == 0)
            {
                time = i * mean_sweep_time + j * ARG.frame_time;
            } else
            {
                time = i * mean_sweep_time + (ARG.projectionSizeZ - 1 - j) * ARG.frame_time;
            }
            taxis.push_back(time);
            for(std::size_t k = 0; k != ARG.basisSize; k++)
            {
                values[k].push_back(basisFunctionsValues[k][i * ARG.projectionSizeZ + j]);
            }
        }
        // Put discontinuity here
        taxis.push_back(0.0);
        for(std::size_t k = 0; k != ARG.basisSize; k++)
        {
            values[k].push_back(std::numeric_limits<double>::quiet_NaN());
        }
    }
    for(uint32_t j = 0; j != ARG.basisSize; j++)
    {
        plt::named_plot(io::xprintf("Function %d", j), taxis, values[j]);
    }
    plt::legend();
    plt::show();
#endif
    std::string startPath = io::getParent(ARG.outputVolumePrefix);
    std::string bname = io::getBasename(ARG.outputVolumePrefix);
    bname = bname.substr(0, bname.find_last_of("."));
    startPath = io::xprintf("%s/%s_", startPath.c_str(), bname.c_str());
    LOGI << io::xprintf("startpath=%s", startPath.c_str());
    LOGI << io::xprintf("Dimensions are [%d %d %d]", ARG.volumeSizeX, ARG.volumeSizeY,
                        ARG.volumeSizeZ);
    std::string xpath = PRG.getRunTimeInfo().getExecutableDirectoryPath();
    std::shared_ptr<GLSQRPerfusionReconstructor> LSQR
        = std::make_shared<GLSQRPerfusionReconstructor>(
            ARG.projectionSizeX, ARG.projectionSizeY, ARG.projectionSizeZ, ARG.pixelSizeX,
            ARG.pixelSizeY, ARG.volumeSizeX, ARG.volumeSizeY, ARG.volumeSizeZ, ARG.voxelSizeX,
            ARG.voxelSizeY, ARG.voxelSizeZ, xpath, ARG.CLdebug, ARG.CLitemsPerWorkgroup,
            ARG.reportKthIteration, startPath, ARG.useSidonProjector);
    int res = LSQR->initializeOpenCL(ARG.CLplatformID);
    if(res < 0)
    {
        std::string ERR = io::xprintf("Could not initialize OpenCL platform %d.", ARG.CLplatformID);
        LOGE << ERR;
        io::throwerr(ERR);
    }
    std::vector<float*> volumes;
    float* volume;
    for(std::size_t basisIND = 0; basisIND != ARG.basisSize; basisIND++)
    {
        volume = new float[ARG.totalVolumeSize]();
        volumes.push_back(volume);
    }
    // testing
    //    io::readBytesFrom("/tmp/X.den", 6, (uint8_t*)volume, ARG.totalVolumeSize * 4);

    LSQR->initializeData(projections, basisFunctionsValues, volumes);
    LSQR->reconstruct(dr, ARG.maxIterationCount, ARG.stoppingRelativeError);

    uint16_t buf[3];
    buf[0] = ARG.volumeSizeY;
    buf[1] = ARG.volumeSizeX;
    buf[2] = ARG.volumeSizeZ;
    std::string f;
    for(std::size_t i = 0; i != ARG.basisSize; i++)
    {
        f = ARG.getVolumeName(i);
        LOGI << io::xprintf("Writing out file %s of the size [%d %d %d].", f.c_str(),
                            ARG.volumeSizeX, ARG.volumeSizeY, ARG.volumeSizeZ);
        io::createEmptyFile(f, 0, true);
        io::appendBytes(f, (uint8_t*)buf, 6);
        io::appendBytes(f, (uint8_t*)volumes[i], ARG.totalVolumeSize * sizeof(float));
        delete[] volumes[i];
        delete[] basisFunctionsValues[i];
    }
    for(std::size_t i = 0; i < ARG.inputProjections.size(); i++)
    {
        delete[] projections[i];
    }
    delete[] vals;
    PRG.endLog(true);
}
