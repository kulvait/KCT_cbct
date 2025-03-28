#pragma once

// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <CL/cl.hpp>
#include <chrono>
#include <ctime>
#include <iostream>

// Internal libraries
#include "AlgorithmsBarrierBuffers.hpp"
#include "BufferedFrame2D.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenProjectionMatrixReader.hpp"
#include "GEOMETRY/Geometry3DParallelI.hpp"
#include "Kniha.hpp"
#include "MATRIX/LightProjectionMatrix.hpp"
#include "MATRIX/ProjectionMatrix.hpp"
#include "MATRIX/utils.hpp"
#include "NDRange/PBCT2DLocalNDRangeFactory.hpp"
#include "OPENCL/OpenCLManager.hpp"
#include "rawop.h"
#include "stringFormatter.h"

using namespace KCT::matrix;
namespace KCT {

/**
 * This class encapsulates partial projection operation.
 * The main difference between PartialPBCT2DOperator and BasePBCTOperator is that
 * PartialPBCT2DOperator will not erase destination array before projection or backprojection. So it
 * behaves additivelly.
 */
class PartialPBCT2DOperator : public virtual Kniha, public AlgorithmsBarrierBuffers
{
public:
    cl::NDRange guessProjectionLocalNDRange(bool barrierCalls);
    cl::NDRange guessBackprojectorLocalNDRange();

    PartialPBCT2DOperator(uint32_t pdimx,
                          uint32_t pdimy,
                          uint32_t pzblock_maxsize,
                          uint32_t vdimx,
                          uint32_t vdimy,
                          uint32_t vzblock_maxsize,
                          uint32_t workGroupSize = 256)
        : AlgorithmsBarrierBuffers()
        , pdimx(pdimx)
        , pdimy(pdimy)
        , pzblock_maxsize(pzblock_maxsize)
        , vdimx(vdimx)
        , vdimy(vdimy)
        , vzblock_maxsize(vzblock_maxsize)
        , workGroupSize(workGroupSize)
    {
        XDIM_maxsize = (uint64_t)vdimx * (uint64_t)vdimy * (uint64_t)vzblock_maxsize;
        BDIM_maxsize = (uint64_t)pdimx * (uint64_t)pdimy * (uint64_t)pzblock_maxsize;
        initReductionParameters(pdimx, pdimy, pzblock_maxsize, vdimx, vdimy, vzblock_maxsize, workGroupSize);
        pdims = cl_int2({ int(pdimx), int(pdimy) });
        pdims_uint = cl_uint2({ pdimx, pdimy });
        timestamp = std::chrono::steady_clock::now();
    }

    void initializeCVPProjector(bool barrierVariant, uint32_t LOCALARRAYSIZE = 7680);
    void initializeSiddonProjector(uint32_t probesPerEdgeX, uint32_t probesPerEdgeY);
    void initializeTTProjector();
    void initializeVolumeConvolution();

    int initializeOpenCL(uint32_t platformID,
                         uint32_t* deviceIds,
                         uint32_t deviceIdsLength,
                         std::string xpath,
                         bool debug,
                         bool relaxed,
                         cl::NDRange projectorLocalNDRange = cl::NullRange,
                         cl::NDRange backprojectorLocalNDRange = cl::NullRange);

    void useJacobiVectorCLCode();

    int problemSetup(std::vector<std::shared_ptr<geometry::Geometry3DParallelI>> geometries,
                     double voxelSpacingX,
                     double voxelSpacingY,
                     double voxelSpacingZ,
                     double volumeCenterX = 0.0,
                     double volumeCenterY = 0.0,
                     double volumeCenterZ = 0.0);

    int allocateXBuffers(uint32_t xBufferCount);
    int allocateBBuffers(uint32_t bBufferCount);
    int allocateTmpXBuffers(uint32_t xBufferCount);
    int allocateTmpBBuffers(uint32_t bBufferCount);
    std::shared_ptr<cl::Buffer> getBBuffer(uint32_t i);
    std::shared_ptr<cl::Buffer> getXBuffer(uint32_t i);
    std::shared_ptr<cl::Buffer> getTmpBBuffer(uint32_t i);
    std::shared_ptr<cl::Buffer> getTmpXBuffer(uint32_t i);

    double adjointProductTest(float* x, float* b);

    void setVerbose(bool verbose, std::string intermediatePrefix = "");

    std::vector<float> computeScalingFactors();

    /**
     * Backprojection X = AT(B)
     *
     * @param B Buffer of all projections from all reconstructed angles of the minimal size
     * BDIM*sizeof(float).
     * @param X Buffer of the size at least XDIM*sizeof(float) to be backprojected to.
     * @param initialProjectionIndex For OS SART 0 by default
     * @param projectionIncrement For OS SART 1 by default
     *
     * @return 0 on success
     */
    int backproject(cl::Buffer& B, cl::Buffer& X, uint32_t initialProjectionIndex = 0, uint32_t projectionIncrement = 1);

    /**
     * Projection B = A x restricted on particular angles by using just projections in the range
     * [geometriesFrom, geometriesTo) and particular strips of k indices in the range [k_from, k_count)
     *
     * @param X Buffer of the size at least XDIM*sizeof(float) to be projected.
     * @param B Buffer to write all projections from  angles in the range [reconstruction_from,
     * reconstruction_to) zero index starts at the projection reconstruction_from
     * BDIM*sizeof(float).
     * @param xslab_z_offset Offset of the volume in X buffer and z offset of the projection in B buffer
     * @param k_count Volume xslab hieght
     * @param geometries_from indexing of gemetry object start inclusive
     * @param geometries_to indexing of gemetry object end exclusive
     * @param initialProjectionIndex For OS SART 0 by default
     * @param projectionIncrement For OS SART 1 by default
     *
     * @return
     */
    int project_partial(uint32_t QID,
                        cl::Buffer& X,
                        cl::Buffer& B,
                        uint32_t xslab_z_offset,
                        uint32_t xslab_vdimz_local,
                        uint32_t geometries_from,
                        uint32_t geometries_to,
                        uint32_t initialProjectionIndex = 0,
                        uint32_t projectionIncrement1 = 1);

protected:
    const cl_float FLOATZERO = 0.0f;
    const cl_double DOUBLEZERO = 0.0;
    float FLOATONE = 1.0f;
    // Constructor defined variables
    cl_int2 pdims;
    cl_uint2 pdims_uint;

    // Problem setup variables
    double voxelSpacingX, voxelSpacingY, voxelSpacingZ;
    cl_double3 voxelSizes;
    cl_double3 volumeCenter;
    std::vector<std::shared_ptr<geometry::Geometry3DParallelI>> geometries;
    std::vector<cl_double8> PM8Vector;
    std::vector<float> scalingFactorVector;

    // Variables for projectors and openCL initialization
    bool useCVPProjector = true;
    bool useBarrierImplementation = false;
    uint32_t LOCALARRAYSIZE = 0;
    bool useSiddonProjector = false;
    cl_uint2 pixelGranularity = { 1, 1 };
    bool useTTProjector = false;
    bool useVolumeAsInitialX0 = false;

    uint32_t xBufferCount, bBufferCount, tmpXBufferCount, tmpBBufferCount;

    // Class functions
    /**
     * Write volume of the size XDIM into the DEN file.
     *
     * @param X buffer to write
     * @param x auxiliary vector to store float data from X
     * @param path output DEN file
     */
    void writeVolume(cl::Buffer& X, float* x, std::string path);
    /**
     * Write projections of the size BDIM into DEN file.
     *
     * @param B buffer to write
     * @param b auxiliary vector to store float data from B
     * @param path output DEN file
     */
    void writeProjections(cl::Buffer& B, float* b, std::string path);
    std::vector<cl_double16> inverseProjectionMatrices();

    // Printing and reporting
    void setTimestamp(bool finishCommandQueue);
    std::chrono::milliseconds millisecondsFromTimestamp(bool setNewTimestamp);
    std::string printTime(std::string msg, bool finishCommandQueue, bool setNewTimestamp);
    void reportTime(std::string msg, bool finishCommandQueue, bool setNewTimestamp);

    cl::NDRange projectorLocalNDRange;
    cl::NDRange projectorLocalNDRangeBarrier;
    cl::NDRange backprojectorLocalNDRange;

    std::vector<std::shared_ptr<cl::Buffer>> x_buffers, tmp_x_buffers;
    std::vector<std::shared_ptr<cl::Buffer>> b_buffers, tmp_b_buffers;
    std::chrono::time_point<std::chrono::steady_clock> timestamp;

    bool verbose = false;
    std::string intermediatePrefix = "";

    const uint32_t pdimx, pdimy, pzblock_maxsize, vdimx, vdimy, vzblock_maxsize;
    uint64_t XDIM_maxsize, BDIM_maxsize;
    const uint32_t workGroupSize;
};

} // namespace KCT
