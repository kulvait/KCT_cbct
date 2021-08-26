#include "catch.hpp"

// Internal libs
#include "FUN/LegendrePolynomialsExplicit.hpp"
#include "FUN/VectorFunctionI.h"
#include "GLSQRReconstructor.hpp"
#include "PROG/RunTimeInfo.hpp"
#include "Perfusion/PerfusionOperator.hpp"
#include "rawop.h"
#include "stringFormatter.h"

using namespace KCT;

std::string basedir(); // Defined in main file so that it will be accessible to linker
uint32_t CLplatformID = 1;
// uint32_t CLplatformID = 0;
uint32_t CLdeviceID = 0;
bool CVPBarierImplementation = false;

void findFirstPlatform()
{
    uint32_t platformsOpenCL = util::OpenCLManager::platformCount();
    std::string ERR;
    if(platformsOpenCL == 0)
    {
        ERR = io::xprintf("No OpenCL platform available to this program.");
        LOGE << ERR;
    }
    for(uint32_t platformID = 0; platformID != platformsOpenCL; platformID++)
    {
        uint32_t devicesOnPlatform = util::OpenCLManager::deviceCount(platformID);
        if(devicesOnPlatform > 0)
        {
            CLplatformID = platformID;
            CLdeviceID = 0;
            return;
        }
    }
}

/*
 *See http://sepwww.stanford.edu/sep/prof/pvi/conj/paper_html/node9.html for details
 */
TEST_CASE("CVP.AdjointDotProduct.nobarrier", "[adjointop][cuttingvox][NOVIZ]")
{
    findFirstPlatform();
    double tol = 1e-5;
    uint32_t projectionSizeX = 616;
    uint32_t projectionSizeY = 480;
    uint32_t projectionSizeZ = 248;
    uint32_t volumeSizeX = 256;
    uint32_t volumeSizeY = 256;
    uint32_t volumeSizeZ = 199;
    double voxelSizeX = 1.0;
    double voxelSizeY = 1.0;
    double voxelSizeZ = 1.0;
    util::RunTimeInfo rti;
    std::string xpath = rti.getExecutableDirectoryPath(); // build dir
    bool debug = false;
    bool CLrelaxed = false;
    uint32_t itemsPerWorkgroup = 256;
    std::string startPath = "";
    uint64_t totalVolumeSize
        = uint64_t(volumeSizeX) * uint64_t(volumeSizeY) * uint64_t(volumeSizeZ);
    uint64_t totalProjectionsSize
        = uint64_t(projectionSizeX) * uint64_t(projectionSizeY) * uint64_t(projectionSizeZ);
    // BaseReconstructor is abstract class so can not be costructed
    std::shared_ptr<GLSQRReconstructor> glsqr = std::make_shared<GLSQRReconstructor>(
        projectionSizeX, projectionSizeY, projectionSizeZ, volumeSizeX, volumeSizeY, volumeSizeZ,
        itemsPerWorkgroup);
    bool barrier = false;
    glsqr->initializeCVPProjector(true, barrier);
    int ecd = glsqr->initializeOpenCL(CLplatformID, &CLdeviceID, 1, xpath, debug, CLrelaxed);
    if(ecd < 0)
    {
        std::string ERR = io::xprintf("Could not initialize OpenCL platform %d.", CLplatformID);
        LOGE << ERR;
        throw std::runtime_error(ERR);
    }
    // Pseudorandom vectors
    std::random_device randomInts;
    int seed;
    seed = randomInts(); // To get random results
    seed = 5; // Fix for tests
    std::mt19937 engine(seed);
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    float* randomX = new float[totalVolumeSize];
    float* randomB = new float[totalProjectionsSize];
    auto gen = [&dis, &engine]() { return dis(engine); };
    std::generate(randomX, randomX + totalVolumeSize, gen);
    std::generate(randomB, randomB + totalProjectionsSize, gen);
    // LOGE << io::xprintf("X%f, %f, %f", randomX[0], randomX[1], randomX[totalVolumeSize - 1]);
    // LOGE << io::xprintf("B%f, %f, %f", randomB[0], randomB[1], randomB[totalProjectionsSize -
    // 1]);
    std::string cameraMatrices = io::xprintf("%s/tests/files/camera.matrices", basedir().c_str());
    std::shared_ptr<io::DenProjectionMatrixReader> cameraMatricesReader
        = std::make_shared<io::DenProjectionMatrixReader>(cameraMatrices);
    std::vector<std::shared_ptr<CameraI>> cameraVector
        = BaseReconstructor::encodeProjectionMatrices(cameraMatricesReader);
    glsqr->problemSetup(randomB, randomX, true, cameraVector, voxelSizeX, voxelSizeY, voxelSizeZ);
    double adjointProductRatio = glsqr->adjointProductTest();
    LOGI << io::xprintf("Ratio is %f", adjointProductRatio);
    REQUIRE(std::abs(adjointProductRatio - 1.0) < tol);
    delete[] randomX;
    delete[] randomB;
}

TEST_CASE("CVP.AdjointDotProduct.barrier_relaxed", "[adjointop][cuttingvox][NOVIZ]")
{
    findFirstPlatform();
    double tol = 1e-5;
    uint32_t projectionSizeX = 616;
    uint32_t projectionSizeY = 480;
    uint32_t projectionSizeZ = 248;
    uint32_t volumeSizeX = 256;
    uint32_t volumeSizeY = 256;
    uint32_t volumeSizeZ = 199;
    double voxelSizeX = 1.0;
    double voxelSizeY = 1.0;
    double voxelSizeZ = 1.0;
    util::RunTimeInfo rti;
    std::string xpath = rti.getExecutableDirectoryPath(); // build dir
    bool debug = false;
    bool CLrelaxed = true;
    uint32_t itemsPerWorkgroup = 256;
    std::string startPath = "";
    uint64_t totalVolumeSize
        = uint64_t(volumeSizeX) * uint64_t(volumeSizeY) * uint64_t(volumeSizeZ);
    uint64_t totalProjectionsSize
        = uint64_t(projectionSizeX) * uint64_t(projectionSizeY) * uint64_t(projectionSizeZ);
    // BaseReconstructor is abstract class so can not be costructed
    std::shared_ptr<GLSQRReconstructor> glsqr = std::make_shared<GLSQRReconstructor>(
        projectionSizeX, projectionSizeY, projectionSizeZ, volumeSizeX, volumeSizeY, volumeSizeZ,
        itemsPerWorkgroup);
    bool barrier = true;
    glsqr->initializeCVPProjector(true, barrier);
    int ecd = glsqr->initializeOpenCL(CLplatformID, &CLdeviceID, 1, xpath, debug, CLrelaxed);
    if(ecd < 0)
    {
        std::string ERR = io::xprintf("Could not initialize OpenCL platform %d.", CLplatformID);
        LOGE << ERR;
        throw std::runtime_error(ERR);
    }
    // Pseudorandom vectors
    std::random_device randomInts;
    int seed;
    seed = randomInts(); // To get random results
    seed = 5; // Fix for tests
    std::mt19937 engine(seed);
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    float* randomX = new float[totalVolumeSize];
    float* randomB = new float[totalProjectionsSize];
    auto gen = [&dis, &engine]() { return dis(engine); };
    std::generate(randomX, randomX + totalVolumeSize, gen);
    std::generate(randomB, randomB + totalProjectionsSize, gen);
    // LOGE << io::xprintf("X%f, %f, %f", randomX[0], randomX[1], randomX[totalVolumeSize - 1]);
    // LOGE << io::xprintf("B%f, %f, %f", randomB[0], randomB[1], randomB[totalProjectionsSize -
    // 1]);
    std::string cameraMatrices = io::xprintf("%s/tests/files/camera.matrices", basedir().c_str());
    std::shared_ptr<io::DenProjectionMatrixReader> cameraMatricesReader
        = std::make_shared<io::DenProjectionMatrixReader>(cameraMatrices);
    std::vector<std::shared_ptr<CameraI>> cameraVector
        = BaseReconstructor::encodeProjectionMatrices(cameraMatricesReader);
    glsqr->problemSetup(randomB, randomX, true, cameraVector, voxelSizeX, voxelSizeY, voxelSizeZ);
    double adjointProductRatio = glsqr->adjointProductTest();
    LOGI << io::xprintf("Ratio is %f", adjointProductRatio);
    REQUIRE(std::abs(adjointProductRatio - 1.0) < tol);
    delete[] randomX;
    delete[] randomB;
}

TEST_CASE("PerfusionOperator AdjointDotProduct TEST", "[adjointop][cuttingvox][NOVIZ]")
{
    findFirstPlatform();
    double tol = 1e-5;
    // Problem data
    uint32_t projectionSizeX = 616;
    uint32_t projectionSizeY = 480;
    uint32_t projectionSizeZ = 248;
    uint32_t volumeSizeX = 256;
    uint32_t volumeSizeY = 256;
    uint32_t volumeSizeZ = 199;
    double voxelSizeX = 1.0;
    double voxelSizeY = 1.0;
    double voxelSizeZ = 1.0;
    bool useExactScaling = true;
    util::RunTimeInfo rti;
    std::string xpath = rti.getExecutableDirectoryPath(); // build dir
    bool debug = false;
    bool CLrelaxed = false;
    uint32_t itemsPerWorkgroup = 256;
    std::string startPath = "";
    uint64_t totalVolumeSize
        = uint64_t(volumeSizeX) * uint64_t(volumeSizeY) * uint64_t(volumeSizeZ);
    uint64_t totalProjectionsSize
        = uint64_t(projectionSizeX) * uint64_t(projectionSizeY) * uint64_t(projectionSizeZ);
    uint32_t basisSize = 6;
    uint32_t sweepCount = 10;
    double frame_time = 16.8;
    double pause_size = 1171;
    double mean_sweep_time = (projectionSizeZ - 1) * frame_time + pause_size;
    double startTime = 0;
    double endTime = (sweepCount - 1) * mean_sweep_time + (projectionSizeZ - 1) * frame_time;
    std::shared_ptr<util::VectorFunctionI> baseFunctionsEvaluator
        = std::make_shared<util::LegendrePolynomialsExplicit>(basisSize - 1, startTime, endTime);
    // Pseudorandom vectors
    std::random_device randomInts;
    int seed;
    seed = randomInts(); // To get random results
    seed = 5; // Fix for tests
    std::mt19937 engine(seed);
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    auto gen = [&dis, &engine]() { return dis(engine); };
    std::vector<float*> randomX;
    std::vector<float*> randomB;
    std::vector<float*> basisFunctionsValues;
    float *x, *b, *basisVals;
    float* vals;
    vals = new float[basisSize];

    for(std::size_t i = 0; i != basisSize; i++)
    {
        basisVals = new float[projectionSizeZ * sweepCount];
        basisFunctionsValues.push_back(basisVals);
        x = new float[totalVolumeSize];
        std::generate(x, x + totalVolumeSize, gen);
        randomX.push_back(x);
    }
    for(std::size_t sweepID = 0; sweepID != sweepCount; sweepID++)
    {
        b = new float[totalProjectionsSize];
        std::generate(b, b + totalProjectionsSize, gen);
        randomB.push_back(b);
        for(std::size_t angleID = 0; angleID != projectionSizeZ; angleID++)
        {
            if(sweepID % 2 == 0)
            {
                baseFunctionsEvaluator->valuesAt(sweepID * mean_sweep_time + angleID * frame_time,
                                                 vals);
            } else
            {
                baseFunctionsEvaluator->valuesAt(
                    sweepID * mean_sweep_time + (projectionSizeZ - 1 - angleID) * frame_time, vals);
            }
            for(std::size_t basisIND = 0; basisIND != basisSize; basisIND++)
            {
                basisFunctionsValues[basisIND][sweepID * projectionSizeZ + angleID]
                    = vals[basisIND];
            }
        }
    }
    std::string cameraMatrices = io::xprintf("%s/tests/files/camera.matrices", basedir().c_str());
    std::shared_ptr<io::DenProjectionMatrixReader> cameraMatricesReader
        = std::make_shared<io::DenProjectionMatrixReader>(cameraMatrices);
    std::vector<std::shared_ptr<matrix::CameraI>> cameraVector;
    std::shared_ptr<matrix::CameraI> pm;
    for(std::size_t k = 0; k != cameraMatricesReader->count(); k++)
    {
        pm = std::make_shared<matrix::LightProjectionMatrix>(cameraMatricesReader->readMatrix(k));
        cameraVector.emplace_back(pm);
    }
    PerfusionOperator PO(projectionSizeX, projectionSizeY, projectionSizeZ, volumeSizeX,
                         volumeSizeY, volumeSizeZ, itemsPerWorkgroup);
    PO.setReportingParameters(true);
    PO.initializeCVPProjector(useExactScaling);
    int ecd = PO.initializeOpenCL(CLplatformID, &CLdeviceID, 1, xpath, debug, CLrelaxed);
    if(ecd < 0)
    {
        std::string ERR = io::xprintf("Could not initialize OpenCL platform %d.", CLplatformID);
        LOGE << ERR;
        throw std::runtime_error(ERR);
    }
    ecd = PO.problemSetup(randomB, basisFunctionsValues, randomX, true, cameraVector, voxelSizeX,
                          voxelSizeY, voxelSizeZ);
    if(ecd != 0)
    {
        std::string ERR = io::xprintf("OpenCL buffers initialization failed.");
        LOGE << ERR;
        throw std::runtime_error(ERR);
    }
    double adjointProductRatio = PO.adjointProductTest();
    LOGI << io::xprintf("Ratio is %f", adjointProductRatio);
    REQUIRE(std::abs(adjointProductRatio - 1.0) < tol);
    for(std::size_t i = 0; i != basisSize; i++)
    {
        delete[] randomX[i];
        delete[] basisFunctionsValues[i];
    }
    for(std::size_t i = 0; i < sweepCount; i++)
    {
        delete[] randomB[i];
    }
    delete[] vals;
}

TEST_CASE("GLSQRReconstructor AdjointDotProduct TA3 projector TEST", "[adjointop][tt][NOVIZ]")
{
    double tol = 1e-5;
    uint32_t projectionSizeX = 616;
    uint32_t projectionSizeY = 480;
    uint32_t projectionSizeZ = 248;
    uint32_t volumeSizeX = 256;
    uint32_t volumeSizeY = 256;
    uint32_t volumeSizeZ = 199;
    double voxelSizeX = 1.0;
    double voxelSizeY = 1.0;
    double voxelSizeZ = 1.0;
    util::RunTimeInfo rti;
    std::string xpath = rti.getExecutableDirectoryPath(); // build dir
    bool debug = false;
    bool CLrelaxed = false;
    uint32_t itemsPerWorkgroup = 256;
    std::string startPath = "";
    uint64_t totalVolumeSize
        = uint64_t(volumeSizeX) * uint64_t(volumeSizeY) * uint64_t(volumeSizeZ);
    uint64_t totalProjectionsSize
        = uint64_t(projectionSizeX) * uint64_t(projectionSizeY) * uint64_t(projectionSizeZ);

    std::shared_ptr<GLSQRReconstructor> glsqr = std::make_shared<GLSQRReconstructor>(
        projectionSizeX, projectionSizeY, projectionSizeZ, volumeSizeX, volumeSizeY, volumeSizeZ,
        itemsPerWorkgroup);
    glsqr->initializeTTProjector();
    int ecd = glsqr->initializeOpenCL(CLplatformID, &CLdeviceID, 1, xpath, debug, CLrelaxed);
    if(ecd < 0)
    {
        std::string ERR = io::xprintf("Could not initialize OpenCL platform %d.", CLplatformID);
        LOGE << ERR;
        throw std::runtime_error(ERR);
    }

    // Pseudorandom vectors
    std::random_device randomInts;
    int seed;
    seed = randomInts(); // To get random results
    seed = 5; // Fix for tests
    std::mt19937 engine(seed);
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    float* randomX = new float[totalVolumeSize];
    float* randomB = new float[totalProjectionsSize];
    auto gen = [&dis, &engine]() { return dis(engine); };
    std::generate(randomX, randomX + totalVolumeSize, gen);
    std::generate(randomB, randomB + totalProjectionsSize, gen);
    LOGE << io::xprintf("X%f, %f, %f", randomX[0], randomX[1], randomX[totalVolumeSize - 1]);
    LOGE << io::xprintf("B%f, %f, %f", randomB[0], randomB[1], randomB[totalProjectionsSize - 1]);
    std::string cameraMatrices = io::xprintf("%s/tests/files/camera.matrices", basedir().c_str());
    std::shared_ptr<io::DenProjectionMatrixReader> cameraMatricesReader
        = std::make_shared<io::DenProjectionMatrixReader>(cameraMatrices);
    std::vector<std::shared_ptr<CameraI>> cameraVector
        = BaseReconstructor::encodeProjectionMatrices(cameraMatricesReader);
    glsqr->problemSetup(randomB, randomX, true, cameraVector, voxelSizeX, voxelSizeY, voxelSizeZ);
    double adjointProductRatio = glsqr->adjointProductTest();
    REQUIRE(std::abs(adjointProductRatio - 1.0) < tol);
    delete[] randomX;
    delete[] randomB;
}

TEST_CASE("GLSQRPerfusionReconstructor AdjointDotProduct TA3 projector TEST",
          "[adjointop][TT][NOVIZ]")
{
    findFirstPlatform();
    double tol = 1e-5;
    // Problem data
    uint32_t projectionSizeX = 616;
    uint32_t projectionSizeY = 480;
    uint32_t projectionSizeZ = 248;
    uint32_t volumeSizeX = 256;
    uint32_t volumeSizeY = 256;
    uint32_t volumeSizeZ = 199;
    double voxelSizeX = 1.0;
    double voxelSizeY = 1.0;
    double voxelSizeZ = 1.0;
    util::RunTimeInfo rti;
    std::string xpath = rti.getExecutableDirectoryPath(); // build dir
    bool debug = false;
    bool CLrelaxed = false;
    uint32_t itemsPerWorkgroup = 256;
    std::string startPath = "";
    uint64_t totalVolumeSize
        = uint64_t(volumeSizeX) * uint64_t(volumeSizeY) * uint64_t(volumeSizeZ);
    uint64_t totalProjectionsSize
        = uint64_t(projectionSizeX) * uint64_t(projectionSizeY) * uint64_t(projectionSizeZ);
    uint32_t basisSize = 6;
    uint32_t sweepCount = 10;
    double frame_time = 16.8;
    double pause_size = 1171;
    double mean_sweep_time = (projectionSizeZ - 1) * frame_time + pause_size;
    double startTime = 0;
    double endTime = (sweepCount - 1) * mean_sweep_time + (projectionSizeZ - 1) * frame_time;
    std::shared_ptr<util::VectorFunctionI> baseFunctionsEvaluator
        = std::make_shared<util::LegendrePolynomialsExplicit>(basisSize - 1, startTime, endTime);
    // Pseudorandom vectors
    std::random_device randomInts;
    int seed;
    seed = randomInts(); // To get random results
    seed = 5; // Fix for tests
    std::mt19937 engine(seed);
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    auto gen = [&dis, &engine]() { return dis(engine); };
    std::vector<float*> randomX;
    std::vector<float*> randomB;
    std::vector<float*> basisFunctionsValues;
    float *x, *b, *basisVals;
    float* vals;
    vals = new float[basisSize];

    for(std::size_t i = 0; i != basisSize; i++)
    {
        basisVals = new float[projectionSizeZ * sweepCount];
        basisFunctionsValues.push_back(basisVals);
        x = new float[totalVolumeSize];
        std::generate(x, x + totalVolumeSize, gen);
        randomX.push_back(x);
    }
    for(std::size_t sweepID = 0; sweepID != sweepCount; sweepID++)
    {
        b = new float[totalProjectionsSize];
        std::generate(b, b + totalProjectionsSize, gen);
        randomB.push_back(b);
        for(std::size_t angleID = 0; angleID != projectionSizeZ; angleID++)
        {
            if(sweepID % 2 == 0)
            {
                baseFunctionsEvaluator->valuesAt(sweepID * mean_sweep_time + angleID * frame_time,
                                                 vals);
            } else
            {
                baseFunctionsEvaluator->valuesAt(
                    sweepID * mean_sweep_time + (projectionSizeZ - 1 - angleID) * frame_time, vals);
            }
            for(std::size_t basisIND = 0; basisIND != basisSize; basisIND++)
            {
                basisFunctionsValues[basisIND][sweepID * projectionSizeZ + angleID]
                    = vals[basisIND];
            }
        }
    }
    std::string cameraMatrices = io::xprintf("%s/tests/files/camera.matrices", basedir().c_str());
    std::shared_ptr<io::DenProjectionMatrixReader> cameraMatricesReader
        = std::make_shared<io::DenProjectionMatrixReader>(cameraMatrices);
    std::vector<std::shared_ptr<matrix::CameraI>> cameraVector;
    std::shared_ptr<matrix::CameraI> pm;
    for(std::size_t k = 0; k != cameraMatricesReader->count(); k++)
    {
        pm = std::make_shared<matrix::LightProjectionMatrix>(cameraMatricesReader->readMatrix(k));
        cameraVector.emplace_back(pm);
    }
    PerfusionOperator PO(projectionSizeX, projectionSizeY, projectionSizeZ, volumeSizeX,
                         volumeSizeY, volumeSizeZ, itemsPerWorkgroup);
    PO.setReportingParameters(true);
    PO.initializeTTProjector();
    int ecd = PO.initializeOpenCL(CLplatformID, &CLdeviceID, 1, xpath, debug, CLrelaxed);
    if(ecd < 0)
    {
        std::string ERR = io::xprintf("Could not initialize OpenCL platform %d.", CLplatformID);
        LOGE << ERR;
        throw std::runtime_error(ERR);
    }
    ecd = PO.problemSetup(randomB, basisFunctionsValues, randomX, true, cameraVector, voxelSizeX,
                          voxelSizeY, voxelSizeZ);
    if(ecd != 0)
    {
        std::string ERR = io::xprintf("OpenCL buffers initialization failed.");
        LOGE << ERR;
        throw std::runtime_error(ERR);
    }
    double adjointProductRatio = PO.adjointProductTest();
    LOGI << io::xprintf("Ratio is %f", adjointProductRatio);
    REQUIRE(std::abs(adjointProductRatio - 1.0) < tol);
    for(std::size_t i = 0; i != basisSize; i++)
    {
        delete[] randomX[i];
        delete[] basisFunctionsValues[i];
    }
    for(std::size_t i = 0; i < sweepCount; i++)
    {
        delete[] randomB[i];
    }
    delete[] vals;
}

TEST_CASE("GLSQRReconstructor AdjointDotProduct Sidon projector TEST", "[adjointop][sidon][NOVIZ]")
{
    double tol = 1e-5;
    uint32_t projectionSizeX = 616;
    uint32_t projectionSizeY = 480;
    uint32_t projectionSizeZ = 248;
    uint32_t volumeSizeX = 256;
    uint32_t volumeSizeY = 256;
    uint32_t volumeSizeZ = 199;
    double voxelSizeX = 1.0;
    double voxelSizeY = 1.0;
    double voxelSizeZ = 1.0;
    util::RunTimeInfo rti;
    std::string xpath = rti.getExecutableDirectoryPath(); // build dir
    bool debug = false;
    bool CLrelaxed = false;
    uint32_t itemsPerWorkgroup = 256;
    std::string startPath = "";
    uint64_t totalVolumeSize
        = uint64_t(volumeSizeX) * uint64_t(volumeSizeY) * uint64_t(volumeSizeZ);
    uint64_t totalProjectionsSize
        = uint64_t(projectionSizeX) * uint64_t(projectionSizeY) * uint64_t(projectionSizeZ);

    std::shared_ptr<GLSQRReconstructor> glsqr = std::make_shared<GLSQRReconstructor>(
        projectionSizeX, projectionSizeY, projectionSizeZ, volumeSizeX, volumeSizeY, volumeSizeZ,
        itemsPerWorkgroup);
    glsqr->initializeSidonProjector(1, 1);
    int ecd = glsqr->initializeOpenCL(CLplatformID, &CLdeviceID, 1, xpath, debug, CLrelaxed);
    if(ecd < 0)
    {
        std::string ERR = io::xprintf("Could not initialize OpenCL platform %d.", CLplatformID);
        LOGE << ERR;
        throw std::runtime_error(ERR);
    }

    // Pseudorandom vectors
    std::random_device randomInts;
    int seed;
    seed = randomInts(); // To get random results
    seed = 5; // Fix for tests
    std::mt19937 engine(seed);
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    float* randomX = new float[totalVolumeSize];
    float* randomB = new float[totalProjectionsSize];
    auto gen = [&dis, &engine]() { return dis(engine); };
    std::generate(randomX, randomX + totalVolumeSize, gen);
    std::generate(randomB, randomB + totalProjectionsSize, gen);
    LOGE << io::xprintf("X%f, %f, %f", randomX[0], randomX[1], randomX[totalVolumeSize - 1]);
    LOGE << io::xprintf("B%f, %f, %f", randomB[0], randomB[1], randomB[totalProjectionsSize - 1]);
    std::string cameraMatrices = io::xprintf("%s/tests/files/camera.matrices", basedir().c_str());
    std::shared_ptr<io::DenProjectionMatrixReader> cameraMatricesReader
        = std::make_shared<io::DenProjectionMatrixReader>(cameraMatrices);
    std::vector<std::shared_ptr<CameraI>> cameraVector
        = BaseReconstructor::encodeProjectionMatrices(cameraMatricesReader);
    glsqr->problemSetup(randomB, randomX, true, cameraVector, voxelSizeX, voxelSizeY, voxelSizeZ);
    double adjointProductRatio = glsqr->adjointProductTest();
    REQUIRE(std::abs(adjointProductRatio - 1.0) < tol);
    delete[] randomX;
    delete[] randomB;
}

TEST_CASE("GLSQRPerfusionReconstructor AdjointDotProduct Sidon projector TEST",
          "[adjointop][sidon][NOVIZ]")
{
    findFirstPlatform();
    double tol = 1e-5;
    // Problem data
    uint32_t projectionSizeX = 616;
    uint32_t projectionSizeY = 480;
    uint32_t projectionSizeZ = 248;
    uint32_t volumeSizeX = 256;
    uint32_t volumeSizeY = 256;
    uint32_t volumeSizeZ = 199;
    double voxelSizeX = 1.0;
    double voxelSizeY = 1.0;
    double voxelSizeZ = 1.0;
    util::RunTimeInfo rti;
    std::string xpath = rti.getExecutableDirectoryPath(); // build dir
    bool debug = false;
    bool CLrelaxed = false;
    uint32_t itemsPerWorkgroup = 256;
    std::string startPath = "";
    uint64_t totalVolumeSize
        = uint64_t(volumeSizeX) * uint64_t(volumeSizeY) * uint64_t(volumeSizeZ);
    uint64_t totalProjectionsSize
        = uint64_t(projectionSizeX) * uint64_t(projectionSizeY) * uint64_t(projectionSizeZ);
    uint32_t basisSize = 6;
    uint32_t sweepCount = 10;
    double frame_time = 16.8;
    double pause_size = 1171;
    double mean_sweep_time = (projectionSizeZ - 1) * frame_time + pause_size;
    double startTime = 0;
    double endTime = (sweepCount - 1) * mean_sweep_time + (projectionSizeZ - 1) * frame_time;
    std::shared_ptr<util::VectorFunctionI> baseFunctionsEvaluator
        = std::make_shared<util::LegendrePolynomialsExplicit>(basisSize - 1, startTime, endTime);
    // Pseudorandom vectors
    std::random_device randomInts;
    int seed;
    seed = randomInts(); // To get random results
    seed = 5; // Fix for tests
    std::mt19937 engine(seed);
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    auto gen = [&dis, &engine]() { return dis(engine); };
    std::vector<float*> randomX;
    std::vector<float*> randomB;
    std::vector<float*> basisFunctionsValues;
    float *x, *b, *basisVals;
    float* vals;
    vals = new float[basisSize];

    for(std::size_t i = 0; i != basisSize; i++)
    {
        basisVals = new float[projectionSizeZ * sweepCount];
        basisFunctionsValues.push_back(basisVals);
        x = new float[totalVolumeSize];
        std::generate(x, x + totalVolumeSize, gen);
        randomX.push_back(x);
    }
    for(std::size_t sweepID = 0; sweepID != sweepCount; sweepID++)
    {
        b = new float[totalProjectionsSize];
        std::generate(b, b + totalProjectionsSize, gen);
        randomB.push_back(b);
        for(std::size_t angleID = 0; angleID != projectionSizeZ; angleID++)
        {
            if(sweepID % 2 == 0)
            {
                baseFunctionsEvaluator->valuesAt(sweepID * mean_sweep_time + angleID * frame_time,
                                                 vals);
            } else
            {
                baseFunctionsEvaluator->valuesAt(
                    sweepID * mean_sweep_time + (projectionSizeZ - 1 - angleID) * frame_time, vals);
            }
            for(std::size_t basisIND = 0; basisIND != basisSize; basisIND++)
            {
                basisFunctionsValues[basisIND][sweepID * projectionSizeZ + angleID]
                    = vals[basisIND];
            }
        }
    }
    std::string cameraMatrices = io::xprintf("%s/tests/files/camera.matrices", basedir().c_str());
    std::shared_ptr<io::DenProjectionMatrixReader> cameraMatricesReader
        = std::make_shared<io::DenProjectionMatrixReader>(cameraMatrices);
    std::vector<std::shared_ptr<matrix::CameraI>> cameraVector;
    std::shared_ptr<matrix::CameraI> pm;
    for(std::size_t k = 0; k != cameraMatricesReader->count(); k++)
    {
        pm = std::make_shared<matrix::LightProjectionMatrix>(cameraMatricesReader->readMatrix(k));
        cameraVector.emplace_back(pm);
    }
    PerfusionOperator PO(projectionSizeX, projectionSizeY, projectionSizeZ, volumeSizeX,
                         volumeSizeY, volumeSizeZ, itemsPerWorkgroup);
    PO.setReportingParameters(true);
    PO.initializeSidonProjector(1, 1);
    int ecd = PO.initializeOpenCL(CLplatformID, &CLdeviceID, 1, xpath, debug, CLrelaxed);
    if(ecd < 0)
    {
        std::string ERR = io::xprintf("Could not initialize OpenCL platform %d.", CLplatformID);
        LOGE << ERR;
        throw std::runtime_error(ERR);
    }
    ecd = PO.problemSetup(randomB, basisFunctionsValues, randomX, true, cameraVector, voxelSizeX,
                          voxelSizeY, voxelSizeZ);
    if(ecd != 0)
    {
        std::string ERR = io::xprintf("OpenCL buffers initialization failed.");
        LOGE << ERR;
        throw std::runtime_error(ERR);
    }
    double adjointProductRatio = PO.adjointProductTest();
    LOGI << io::xprintf("Ratio is %f", adjointProductRatio);
    REQUIRE(std::abs(adjointProductRatio - 1.0) < tol);
    for(std::size_t i = 0; i != basisSize; i++)
    {
        delete[] randomX[i];
        delete[] basisFunctionsValues[i];
    }
    for(std::size_t i = 0; i < sweepCount; i++)
    {
        delete[] randomB[i];
    }
    delete[] vals;
}
