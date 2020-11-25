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
#include "MATRIX/ProjectionMatrix.hpp"
#include "OPENCL/OpenCLManager.hpp"
#include "Watches.hpp"
#include "rawop.h"
#include "stringFormatter.h"

namespace CTL {

class BasePerfusionReconstructor : public virtual Kniha, public AlgorithmsBarierBuffers
{
public:
    /**
     * Initialize Cutting Voxel Projector
     *
     * @param volume Pointer to volume file
     * @param vdimx Volume x dimension
     * @param vdimy Volume y dimension
     * @param vdimz Volume z dimension
     * @param xpath Path of cl kernel files
     * @param debug Should debugging be used by suppliing source and -g as options
     */
    BasePerfusionReconstructor(uint32_t pdimx,
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
    }

    void initializeCVPProjector(bool useExactScaling);
    void initializeSidonProjector(uint32_t probesPerEdgeX, uint32_t probesPerEdgeY);
    void initializeTTProjector();

    int problemSetup(std::vector<float*> projections,
                     std::vector<float*> basisVectorValues,
                     std::vector<float*> volumes,
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
    std::vector<std::shared_ptr<cl::Buffer>> getBBuffers(uint32_t i);
    std::vector<std::shared_ptr<cl::Buffer>> getXBuffers(uint32_t i);
    std::vector<std::shared_ptr<cl::Buffer>> getTmpBBuffers(uint32_t i);
    std::vector<std::shared_ptr<cl::Buffer>> getTmpXBuffers(uint32_t i);

    int initializeVectors(std::vector<float*> projections,
                          std::vector<float*> basisVectorValues,
                          std::vector<float*> volumes,
                          bool volumeContainsX0);

    virtual int
    reconstruct(uint32_t maxIterations = 100, float errCondition = 0.01, bool blocking = false) = 0;

    void setReportingParameters(bool verbose,
                                uint32_t reportKthIteration = 0,
                                std::string progressPrefixPath = "");
    double adjointProductTest();

protected:
    /**
     * Backprojection X = AT(B)
     *
     * @param B Buffer of all projections from all reconstructed angles of the minimal size
     * BDIM*sizeof(float).
     * @param X Buffer of the size at least XDIM*sizeof(float) to be backprojected to.
     *
     * @return 0 on success
     */
    int backproject_partial(cl::Buffer& B, cl::Buffer& X, uint32_t angleID);
    int backproject(std::vector<std::shared_ptr<cl::Buffer>>& B,
                    std::vector<std::shared_ptr<cl::Buffer>>& X,
                    bool blocking = false);

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
    int project(std::vector<std::shared_ptr<cl::Buffer>>& X,
                std::vector<std::shared_ptr<cl::Buffer>>& B,
                bool blocking = false);

    void zeroXBuffers(std::vector<std::shared_ptr<cl::Buffer>>& X);
    void zeroBBuffers(std::vector<std::shared_ptr<cl::Buffer>>& B);

    const cl_float FLOATZERO = 0.0;
    const cl_double DOUBLEZERO = 0.0;
    float FLOATONE = 1.0f;
    // Constructor defined variables
    cl_int2 pdims;
    cl_uint2 pdims_uint;
    cl_int3 vdims;
    uint32_t XVNUM, BVNUM; // Basis size and sweep count

    // Variables for projectors and openCL initialization
    bool useCVPProjector = false;
    bool exactProjectionScaling = false;
    bool useSidonProjector = false;
    cl_uint2 pixelGranularity = { 1, 1 };
    bool useTTProjector = false;
    bool useVolumeAsInitialX0 = false;

    // Problem setup variables
    double volumeOffsetX, volumeOffsetY, volumeOffsetZ;
    double voxelSpacingX, voxelSpacingY, voxelSpacingZ;
    cl_double3 voxelSizes;
    cl_double3 volumeCenter;
    std::vector<std::shared_ptr<CameraI>> cameraVector;
    std::vector<cl_double16> PM12Vector;
    std::vector<cl_double16> ICM16Vector;
    std::vector<float> scalingFactorVector;

    double normBBuffer_barier_double(std::vector<std::shared_ptr<cl::Buffer>>& B);
    double normXBuffer_barier_double(std::vector<std::shared_ptr<cl::Buffer>>& X);
    double scalarProductXBuffer_barier_double(std::vector<std::shared_ptr<cl::Buffer>>& A,
                                              std::vector<std::shared_ptr<cl::Buffer>>& B);
    double scalarProductBBuffer_barier_double(std::vector<std::shared_ptr<cl::Buffer>>& A,
                                              std::vector<std::shared_ptr<cl::Buffer>>& B);
    int scaleFloatVector(std::vector<std::shared_ptr<cl::Buffer>>& A, float c, unsigned int size);
    void setTimepoint();

    void setTimestamp(bool finishCommandQueue);
    std::chrono::milliseconds millisecondsFromTimestamp(bool setNewTimestamp);
    void reportTime(std::string msg, bool finishCommandQueue, bool setNewTimestamp);
    void writeVolume(cl::Buffer& X, std::string path);
    void writeProjections(cl::Buffer& B, std::string path);
    void writeVolume(std::vector<std::shared_ptr<cl::Buffer>>& X, std::string path);
    void writeProjections(std::vector<std::shared_ptr<cl::Buffer>>& B, std::string path);
    // Backprojecting from B to X
    int copyFloatVector(cl::Buffer& from, cl::Buffer& to, unsigned int size);
    int copyFloatVector(std::vector<std::shared_ptr<cl::Buffer>>& from,
                        std::vector<std::shared_ptr<cl::Buffer>>& to,
                        unsigned int size);
    int
    addIntoFirstVectorSecondVectorScaled(cl::Buffer& a, cl::Buffer& b, float f, unsigned int size);
    int
    addIntoFirstVectorScaledSecondVector(cl::Buffer& a, cl::Buffer& b, float f, unsigned int size);
    int addIntoFirstVectorSecondVectorScaled(std::vector<std::shared_ptr<cl::Buffer>>& a,
                                             std::vector<std::shared_ptr<cl::Buffer>>& b,
                                             float f,
                                             unsigned int size);
    int addIntoFirstVectorScaledSecondVector(std::vector<std::shared_ptr<cl::Buffer>>& a,
                                             std::vector<std::shared_ptr<cl::Buffer>>& b,
                                             float f,
                                             unsigned int size);
    int algFLOATvector_A_equals_cB(std::vector<std::shared_ptr<cl::Buffer>>& A,
                                   std::vector<std::shared_ptr<cl::Buffer>>& B,
                                   float f,
                                   unsigned int size,
                                   bool blocking=false);
    std::vector<matrix::ProjectionMatrix>
    encodeProjectionMatrices(std::shared_ptr<io::DenProjectionMatrixReader> pm);
    std::vector<float> computeScalingFactors(std::vector<matrix::ProjectionMatrix> PM);

    std::vector<float*> x; // Volume data
    std::vector<float*>
        basisFunctionsValues; // Basis functions at time points of particular projections
    std::vector<float*> b; // Projection data

    // OpenCL buffers
    std::vector<std::shared_ptr<cl::Buffer>> b_buf;
    std::vector<std::shared_ptr<cl::Buffer>> x_buf;

    std::shared_ptr<cl::Buffer> tmp_x_buf = nullptr, tmp_b_buf = nullptr;
    std::vector<std::vector<std::shared_ptr<cl::Buffer>>> x_buffers, tmp_x_buffers;
    std::vector<std::vector<std::shared_ptr<cl::Buffer>>> b_buffers, tmp_b_buffers;
    // tmp_b_buf for rescaling, tmp_x_buf for LSQR
    std::chrono::time_point<std::chrono::steady_clock> timestamp;

    bool verbose = false;
    uint32_t reportKthIteration = 0;
    std::string progressPrefixPath = "";
};

} // namespace CTL
