// Logging
#include "PLOG/PlogSetup.h"

// Internal libraries
#include "CArmArguments.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenProjectionMatrixReader.hpp"
#include "FUN/ChebyshevPolynomialsExplicit.hpp"
#include "FUN/FourierSeries.hpp"
#include "FUN/LegendrePolynomialsExplicit.hpp"
#include "FUN/StepFunction.hpp"
#include "PROG/Program.hpp"
#include "Perfusion/PerfusionOperator.hpp"

using namespace KCT;
using namespace KCT::util;
using namespace KCT::io;

class Args : public CArmArguments
{
    virtual void defineArguments();
    virtual int preParse() { return 0; };
    virtual int postParse()
    {
        DenFileInfo vi(inputVolumes[0]);
        volumeSizeX = vi.dimx();
        volumeSizeY = vi.dimy();
        volumeSizeZ = vi.dimz();
        DenFileInfo mi(inputProjectionMatrices);
        projectionSizeZ = mi.dimz();
        totalVolumeSize = uint64_t(volumeSizeX) * uint64_t(volumeSizeY) * uint64_t(volumeSizeZ);
        totalProjectionsSize
            = uint64_t(projectionSizeX) * uint64_t(projectionSizeY) * uint64_t(projectionSizeZ);
        basisSize = inputVolumes.size();
        return 0;
    };

public:
    Args(int argc, char** argv, std::string prgName)
        : Arguments(argc, argv, prgName)
        , CArmArguments(argc, argv, prgName){};

    std::string projectionFilePath(uint32_t sweepID)
    {
        return io::xprintf("%s%02d.den", outputProjectionPrefix.c_str(), sweepID + 1);
    };

    std::string compareProjectionPath(uint32_t sweepID)
    {
        std::string f = io::xprintf("%s%02d.images", compareProjectionPrefix.c_str(), sweepID + 1);
        LOGI << f;
        return f;
    };

    std::string outputProjectionPrefix;
    std::string compareProjectionPrefix = "";
    // Input files
    std::string inputProjectionMatrices;
    std::vector<std::string> inputVolumes;
    uint32_t sweepCount = 10;
    uint32_t basisSize;
    uint64_t totalVolumeSize;
    uint64_t totalProjectionsSize;
    bool debug = false;
    uint32_t itemsPerWorkgroup = 256;
    bool reportIntermediate = false;
    uint32_t platformId = 1;
};

void Args::defineArguments()
{
    cliApp
        ->add_option("output_projection_pattern", outputProjectionPrefix,
                     "Output: pattern of projections to reconstruct, PATTERN_perfproj${02i}.den "
                     "will be used for "
                     "output files, where i is a sweep ID.")
        ->required();
    cliApp
        ->add_option("input_projection_matrices", inputProjectionMatrices,
                     "Input: projection matrices to specify cone beam geometry.")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp
        ->add_option("volume_coefficients", inputVolumes,
                     "Input: projection files in a DEN format generated by the n sweep protocol. ")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp->add_option(
        "--compare-projections-prefix", compareProjectionPrefix,
        "Expected projection data to compare will be in the format PREFIX${02i}.images. ");
    addVoxelSizeArgs();
    addProjectionSizeArgs();
    addVolumeCenterArgs();
    addBasisSpecificationArgs(false);
    addCLSettingsArgs();
    addRelaxedArg();
}

int main(int argc, char* argv[])
{
    Program PRG(argc, argv);
    // Program initialization
    // Parsing arguments
    Args ARG(argc, argv,
             "Projection of the volumes defined by the basis functions and coefficient values by "
             "the perfusion operator.");
    int parseResult = ARG.parse();
    if(parseResult > 0)
    {
        return 0; // Exited sucesfully, help message printed
    } else if(parseResult != 0)
    {
        return -1; // Exited somehow wrong
    }
    PRG.startLog();
    // STARTBODY
    std::shared_ptr<io::DenProjectionMatrixReader> dr
        = std::make_shared<io::DenProjectionMatrixReader>(ARG.inputProjectionMatrices);
    float* projection;
    float* basisVals;
    float* vals = new float[ARG.basisSize];

    std::vector<float*> projections;
    std::vector<float*> basisFunctionsValues;
    double mean_sweep_time = (ARG.projectionSizeZ - 1) * ARG.frame_time + ARG.pause_size;
    double startTime = ARG.start_offset;
    double endTime = (ARG.sweepCount - 1) * mean_sweep_time
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
            ARG.engineerBasis, numberOfFunctions, startTime, endTime);
    }
    for(std::size_t j = 0; j != ARG.basisSize; j++)
    {
        basisVals = new float[ARG.projectionSizeZ * ARG.sweepCount];
        basisFunctionsValues.push_back(basisVals);
    }
    for(std::size_t sweepID = 0; sweepID != ARG.sweepCount; sweepID++)
    {
        projection = new float[ARG.totalProjectionsSize];
        if(!ARG.compareProjectionPrefix.empty())
        {
            io::readBytesFrom(ARG.compareProjectionPath(sweepID), 6, (uint8_t*)projection,
                              ARG.totalProjectionsSize * 4);
        }
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

    for(std::size_t i = 0; i != ARG.sweepCount; i++)
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
    std::string startPath = io::getParent(ARG.outputProjectionPrefix);
    std::string bname = io::getBasename(ARG.outputProjectionPrefix);
    bname = bname.substr(0, bname.find_last_of("."));
    startPath = io::xprintf("%s/%s_", startPath.c_str(), bname.c_str());
    LOGI << io::xprintf("startpath=%s", startPath.c_str());
    LOGI << io::xprintf("Dimensions are [%d %d %d]", ARG.volumeSizeX, ARG.volumeSizeY,
                        ARG.volumeSizeZ);
    std::string xpath = PRG.getRunTimeInfo().getExecutableDirectoryPath();
    std::vector<float*> volumes;
    float* volume;
    for(std::size_t basisIND = 0; basisIND != ARG.basisSize; basisIND++)
    {
        volume = new float[ARG.totalVolumeSize]();
        io::readBytesFrom(ARG.inputVolumes[basisIND], 6, (uint8_t*)volume, ARG.totalVolumeSize * 4);
        volumes.push_back(volume);
    }
    std::vector<std::shared_ptr<matrix::CameraI>> cameraVector;
    std::shared_ptr<matrix::CameraI> pm;
    for(std::size_t k = 0; k != dr->count(); k++)
    {
        pm = std::make_shared<matrix::LightProjectionMatrix>(dr->readMatrix(k));
        cameraVector.emplace_back(pm);
    }
    PerfusionOperator PO(ARG.projectionSizeX, ARG.projectionSizeY, ARG.projectionSizeZ,
                         ARG.volumeSizeX, ARG.volumeSizeY, ARG.volumeSizeZ,
                         ARG.CLitemsPerWorkgroup);
    PO.setReportingParameters(true, ARG.reportKthIteration, startPath);
    if(ARG.useSiddonProjector)
    {
        PO.initializeSiddonProjector(ARG.probesPerEdge, ARG.probesPerEdge);
    } else if(ARG.useTTProjector)
    {

        PO.initializeTTProjector();
    } else
    {
        PO.initializeCVPProjector(ARG.useExactScaling);
    }
    int ecd = PO.initializeOpenCL(ARG.CLplatformID, &ARG.CLdeviceIDs[0], ARG.CLdeviceIDs.size(),
                                  xpath, ARG.CLdebug, ARG.CLrelaxed);
    if(ecd < 0)
    {
        std::string ERR = io::xprintf("Could not initialize OpenCL platform %d.", ARG.CLplatformID);
        LOGE << ERR;
        throw std::runtime_error(ERR);
    }
    ecd = PO.problemSetup(projections, basisFunctionsValues, volumes, true, cameraVector,
                          ARG.voxelSizeX, ARG.voxelSizeY, ARG.voxelSizeZ, ARG.volumeCenterX,
                          ARG.volumeCenterY, ARG.volumeCenterZ);
    if(ecd != 0)
    {
        std::string ERR = io::xprintf("OpenCL buffers initialization failed.");
        KCTERR(ERR);
    }
    PO.project();
    // testing
    //    io::readBytesFrom("/tmp/X.den", 6, (uint8_t*)volume, ARG.totalVolumeSize * 4);
    uint16_t buf[3];
    buf[0] = ARG.projectionSizeY;
    buf[1] = ARG.projectionSizeX;
    buf[2] = ARG.projectionSizeZ;
    std::string f;
    for(std::size_t sweepID = 0; sweepID != ARG.sweepCount; sweepID++)
    {
        f = ARG.projectionFilePath(sweepID);
        LOGI << io::xprintf("Writing out file %s of the size [%d %d %d].", f.c_str(),
                            ARG.projectionSizeX, ARG.projectionSizeY, ARG.projectionSizeZ);
        io::createEmptyFile(f, 0, true);
        io::appendBytes(f, (uint8_t*)buf, 6);
        io::appendBytes(f, (uint8_t*)projections[sweepID],
                        ARG.totalProjectionsSize * sizeof(float));
        delete[] projections[sweepID];
    }
    for(std::size_t basisIND = 0; basisIND < ARG.basisSize; basisIND++)
    {
        delete[] volumes[basisIND];
        delete[] basisFunctionsValues[basisIND];
    }
    delete[] vals;

    // ENDBODY
    PRG.endLog(true);
    return 0;
}
