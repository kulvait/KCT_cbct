#include "catch.hpp"

// Internal libs
#include "PROG/RunTimeInfo.hpp"
#include "GLSQRReconstructor.hpp"
#include "rawop.h"
#include "stringFormatter.h"

using namespace CTL;

std::string basedir(); // Defined in main file so that it will be accessible to linker

/*
 *See http://sepwww.stanford.edu/sep/prof/pvi/conj/paper_html/node9.html for details
 */
TEST_CASE("TEST: CTL::io::AdjointDotProduct", "[adjointop][NOVIZ]")
{
	double tol = 1e-5;
    uint32_t projectionSizeX = 616;
    uint32_t projectionSizeY = 480;
    uint32_t projectionSizeZ = 248;
    double pixelSizeX = 0.616;
    double pixelSizeY = 0.616;
    uint32_t volumeSizeX = 256;
    uint32_t volumeSizeY = 256;
    uint32_t volumeSizeZ = 199;
    double voxelSizeX = 1.0;
    double voxelSizeY = 1.0;
    double voxelSizeZ = 1.0;
    util::RunTimeInfo rti;
    std::string xpath = rti.getExecutableDirectoryPath(); // build dir
    bool debug = false;
    uint32_t itemsPerWorkgroup = 256;
    uint32_t reportIterations = 0;
    std::string startPath = "";
    uint64_t totalVolumeSize
        = uint64_t(volumeSizeX) * uint64_t(volumeSizeY) * uint64_t(volumeSizeZ);
    uint64_t totalProjectionsSize
        = uint64_t(projectionSizeX) * uint64_t(projectionSizeY) * uint64_t(projectionSizeZ);

    std::shared_ptr<GLSQRReconstructor> glsqr = std::make_shared<GLSQRReconstructor>(
        projectionSizeX, projectionSizeY, projectionSizeZ, pixelSizeX, pixelSizeY, volumeSizeX,
        volumeSizeY, volumeSizeZ, voxelSizeX, voxelSizeY, voxelSizeZ, xpath, debug,
        itemsPerWorkgroup, reportIterations, startPath);
    int res = glsqr->initializeOpenCL(1);
    if(res < 0)
    {
        std::string ERR = io::xprintf("Could not initialize OpenCL platform 1");
        LOGE << ERR;
        io::throwerr(ERR);
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
    glsqr->initializeVectors(randomB, randomX);
    std::string cameraMatrices = io::xprintf("%s/tests/files/camera.matrices", basedir().c_str());
    std::shared_ptr<io::DenProjectionMatrixReader> cameraMatricesReader
        = std::make_shared<io::DenProjectionMatrixReader>(cameraMatrices);
    double adjointProductRatio = glsqr->adjointProductTest(cameraMatricesReader);
	LOGI << io::xprintf("Ratio is %f", adjointProductRatio);
	REQUIRE(std::abs(adjointProductRatio - 1.0) < tol);
    delete[] randomX;
    delete[] randomB;
}
