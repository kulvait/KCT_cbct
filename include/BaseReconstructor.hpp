#pragma once

// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <CL/cl.hpp>
#include <chrono>
#include <ctime>
#include <iostream>

// Internal libraries
#include "AlgorithmsBarierBuffers.hpp"
#include "DEN/DenProjectionMatrixReader.hpp"
#include "Kniha.hpp"
#include "MATRIX/LightProjectionMatrix.hpp"
#include "MATRIX/ProjectionMatrix.hpp"
#include "MATRIX/utils.hpp"
#include "OPENCL/OpenCLManager.hpp"
#include "rawop.h"
#include "stringFormatter.h"

using namespace CTL::matrix;
namespace CTL {

class BaseReconstructor : public virtual Kniha, public AlgorithmsBarierBuffers
{
public:
    BaseReconstructor(uint32_t pdimx,
                      uint32_t pdimy,
                      uint32_t pdimz,
                      uint32_t vdimx,
                      uint32_t vdimy,
                      uint32_t vdimz,
                      uint32_t workGroupSize = 256)
        : AlgorithmsBarierBuffers(pdimx, pdimy, pdimz, vdimx, vdimy, vdimz, workGroupSize)
    {
        pdims = cl_int2({ int(pdimx), int(pdimy) });
        pdims_uint = cl_uint2({ pdimx, pdimy });
        vdims = cl_int3({ int(vdimx), int(vdimy), int(vdimz) });
        timestamp = std::chrono::steady_clock::now();
        if(vdimx % 8 == 0 && vdimy % 8 == 0 && workGroupSize >= 64)
        {
            localRange = cl::NDRange(1, 8, 8);
        } else
        {
            localRange = cl::NDRange(1, 1, 1);
        }
    }

    void initializeCVPProjector(bool useExactScaling);
    void initializeSidonProjector(uint32_t probesPerEdgeX, uint32_t probesPerEdgeY);
    void initializeTTProjector();

    void useJacobiVectorCLCode();

    int problemSetup(float* projection,
                     float* volume,
                     bool volumeContainsX0,
                     std::vector<std::shared_ptr<matrix::CameraI>> camera,
                     double voxelSpacingX,
                     double voxelSpacingY,
                     double voxelSpacingZ,
                     double volumeOffsetX = 0.0,
                     double volumeOffsetY = 0.0,
                     double volumeOffsetZ = 0.0);

    int allocateXBuffers(uint32_t xBufferCount);
    int allocateBBuffers(uint32_t bBufferCount);
    int allocateTmpXBuffers(uint32_t xBufferCount);
    int allocateTmpBBuffers(uint32_t bBufferCount);
    std::shared_ptr<cl::Buffer> getBBuffer(uint32_t i);
    std::shared_ptr<cl::Buffer> getXBuffer(uint32_t i);
    std::shared_ptr<cl::Buffer> getTmpBBuffer(uint32_t i);
    std::shared_ptr<cl::Buffer> getTmpXBuffer(uint32_t i);

    virtual int reconstruct(uint32_t maxItterations, float minDiscrepancyError) = 0;
    double adjointProductTest();
    int vectorIntoBuffer(cl::Buffer X, float* v, std::size_t size);

    static std::vector<std::shared_ptr<CameraI>>
    encodeProjectionMatrices(std::shared_ptr<io::DenProjectionMatrixReader> pm);

    void setReportingParameters(bool verbose,
                                uint32_t reportKthIteration = 0,
                                std::string progressPrefixPath = "");

protected:
    const cl_float FLOATZERO = 0.0;
    const cl_double DOUBLEZERO = 0.0;
    float FLOATONE = 1.0f;
    // Constructor defined variables
    cl_int2 pdims;
    cl_uint2 pdims_uint;
    cl_int3 vdims;

    // Problem setup variables
    double volumeOffsetX, volumeOffsetY, volumeOffsetZ;
    double voxelSpacingX, voxelSpacingY, voxelSpacingZ;
    cl_double3 voxelSizes;
    cl_double3 volumeCenter;
    std::vector<std::shared_ptr<CameraI>> cameraVector;
    std::vector<cl_double16> PM12Vector;
    std::vector<cl_double16> ICM16Vector;
    std::vector<float> scalingFactorVector;

    // Variables for projectors and openCL initialization
    bool useCVPProjector = true;
    bool exactProjectionScaling = true;
    bool useSidonProjector = false;
    cl_uint2 pixelGranularity = { 1, 1 };
    bool useTTProjector = false;
    bool useVolumeAsInitialX0 = false;

    uint32_t xBufferCount, bBufferCount, tmpXBufferCount, tmpBBufferCount;

    // Class functions
    int initializeVectors(float* projection, float* volume, bool volumeContainsX0);
    void writeVolume(cl::Buffer& X, std::string path);
    void writeProjections(cl::Buffer& B, std::string path);
    std::vector<cl_double16> inverseProjectionMatrices();

    // Printing and reporting
    void setTimestamp(bool finishCommandQueue);
    std::chrono::milliseconds millisecondsFromTimestamp(bool setNewTimestamp);
    std::string printTime(std::string msg, bool finishCommandQueue, bool setNewTimestamp);
    void reportTime(std::string msg, bool finishCommandQueue, bool setNewTimestamp);

    // Functions to manipulate with buffers
    int multiplyVectorsIntoFirstVector(cl::Buffer& A, cl::Buffer& B, uint64_t size);
    int vectorA_multiple_B_equals_C(cl::Buffer& A, cl::Buffer& B, cl::Buffer& C, uint64_t size);
    int copyFloatVector(cl::Buffer& from, cl::Buffer& to, unsigned int size);
    int scaleFloatVector(cl::Buffer& v, float f, unsigned int size);
    int B_equals_A_plus_B_offsets(
        cl::Buffer& A, unsigned int ao, cl::Buffer& B, unsigned int bo, unsigned int size);
    int
    addIntoFirstVectorSecondVectorScaled(cl::Buffer& a, cl::Buffer& b, float f, unsigned int size);
    int
    addIntoFirstVectorScaledSecondVector(cl::Buffer& a, cl::Buffer& b, float f, unsigned int size);
    int invertFloatVector(cl::Buffer& X, unsigned int size);
    std::vector<float> computeScalingFactors();

    cl::NDRange localRange;
    /**
     * Backprojection X = AT(B)
     *
     * @param B Buffer of all projections from all reconstructed angles of the minimal size
     * BDIM*sizeof(float).
     * @param X Buffer of the size at least XDIM*sizeof(float) to be backprojected to.
     *
     * @return 0 on success
     */
    int backproject(cl::Buffer& B, cl::Buffer& X);

    /**
     * Projection B = A (X)
     *
     * @param X Buffer of the size at least XDIM*sizeof(float) to be projected.
     * @param B Buffer to write all projections from all reconstructed angles of the minimal size
     * BDIM*sizeof(float).
     *
     * @return
     */
    int project(cl::Buffer& X, cl::Buffer& B);

    float* x = nullptr; // Volume data
    float* b = nullptr; // Projection data

    // OpenCL buffers
    std::shared_ptr<cl::Buffer> b_buf = nullptr;
    std::shared_ptr<cl::Buffer> x_buf = nullptr;
    // tmp_b_buf for rescaling, tmp_x_buf for LSQR
    std::shared_ptr<cl::Buffer> tmp_x_buf = nullptr, tmp_b_buf = nullptr;
    std::vector<std::shared_ptr<cl::Buffer>> x_buffers, tmp_x_buffers;
    std::vector<std::shared_ptr<cl::Buffer>> b_buffers, tmp_b_buffers;
    std::chrono::time_point<std::chrono::steady_clock> timestamp;

    bool verbose = false;
    std::string progressPrefixPath = "";
    uint32_t reportKthIteration = 0;
};

} // namespace CTL
