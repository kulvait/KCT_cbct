#pragma once

// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <CL/cl.hpp>
#include <iostream>
#include <limits>

// Internal libraries
#include "BasePBCTOperator.hpp"
#include "Kniha.hpp"
#include "MATRIX/LUDoolittleForm.hpp"
#include "MATRIX/ProjectionMatrix.hpp"
#include "MATRIX/SquareMatrix.hpp"
#include "OPENCL/OpenCLManager.hpp"
#include "rawop.h"
#include "stringFormatter.h"

namespace KCT {

class ParallelBeamProjector : public virtual BasePBCTOperator
{
public:
    /**
     * Class that encapsulates projector and backprojector implementation of Cutting Voxel Projector
     * and other algorithms.
     *
     * @param pdimx Number of pixels
     * @param pdimy Number of pixels
     * @param vdimx Number of voxels
     * @param vdimy Number of voxels
     * @param vdimz Number of voxels
     */
    ParallelBeamProjector(uint32_t pdimx,
                          uint32_t pdimy,
                          uint32_t pdimz,
                          uint32_t vdimx,
                          uint32_t vdimy,
                          uint32_t vdimz,
                          uint32_t workGroupSize = 256)
        : BasePBCTOperator(pdimx,
                           pdimy,
                           pdimz,
                           vdimx,
                           vdimy,
                           vdimz,
                           workGroupSize)
    {
        totalVolumeBufferSize = XDIM * sizeof(float);
        totalProjectionBufferSize = BDIM * sizeof(float);
        frameSize = pdimx * pdimy;
    }
    /**
     * Initialize volume buffer by given size.
     *
     * @param volumeSizeX
     * @param volumeSizeY
     * @param volumeSizeZ
     * @param volumeArray If its nullptr, initialize by zero.
     *
     * @return
     */
    int initializeOrUpdateVolumeBuffer(float* volumeArray = nullptr);
    /**
     * Initialize volume buffer by given size, updates voxel size of the projector.
     *
     * @param volumeSizeX
     * @param volumeSizeY
     * @param volumeSizeZ
     * @param volumeArray If its nullptr, initialize by zero.
     *
     * @return
     */
    int initializeOrUpdateVolumeBuffer(uint32_t vdimx,
                                       uint32_t vdimy,
                                       uint32_t vdimz,
                                       float* volumeArray = nullptr);

    int fillVolumeBufferByConstant(float constant);

    /**
     * Initialize projection buffer by given size. With or without updating dimensions
     *
     * @param projectionSizeX
     * @param projectionSizeY
     * @param projectionSizeZ
     * @param projectionArray If is nullptr, initialize by zero.
     *
     * @return
     */
    int initializeOrUpdateProjectionBuffer(uint32_t pdimx,
                                           uint32_t pdimy,
                                           uint32_t pdimz,
                                           float* projectionArray = nullptr);
    int initializeOrUpdateProjectionBuffer(uint32_t projectionSizeZ,
                                           float* projectionArray = nullptr);
    int initializeOrUpdateProjectionBuffer(float* projectionArray = nullptr);

    int fillProjectionBufferByConstant(float constant);

    int project(float* volume, float* projection);
    int project_print_discrepancy(float* volume, float* projection, float* rhs);
    int backproject(float* projection, float* volume);

    double normSquare(float* projection, uint32_t pdimx, uint32_t pdimy);
    double normSquareDifference(float* projection, uint32_t pdimx, uint32_t pdimy);

private:
    int arrayIntoBuffer(float* c_array, cl::Buffer cl_buffer, uint64_t size);
    int bufferIntoArray(cl::Buffer cl_buffer, float* c_array, uint64_t size);
    const cl_float FLOATZERO = 0.0f;
    const cl_double DOUBLEZERO = 0.0;
    float FLOATONE = 1.0f;
    float* volume = nullptr;
    uint64_t totalVolumeBufferSize;
    uint64_t frameSize;
    uint64_t totalProjectionBufferSize;
    cl::NDRange projectorLocalNDRange;
    cl::NDRange projectorLocalNDRangeBarrier;

    bool centerVoxelProjector = false;
    cl_int3 vdims;
    cl_int2 pdims;
    cl_uint2 pdims_uint;
    cl_double3 voxelSizes;
    cl_double3 volumeCenter;
    bool useCVPProjector = true;
    bool useCVPExactProjectionsScaling = true;
    bool useCVPElevationCorrection = false;
    bool useBarrierImplementation = false;
    uint32_t LOCALARRAYSIZE = 0;
    bool useSiddonProjector = false;
    cl_uint2 pixelGranularity = { 1, 1 };
    bool useTTProjector = false;

    std::shared_ptr<cl::Buffer> volumeBuffer = nullptr;
    std::shared_ptr<cl::Buffer> projectionBuffer = nullptr;
    std::shared_ptr<cl::Buffer> tmpBuffer = nullptr;
    size_t tmpBuffer_size = 0;
    std::chrono::time_point<std::chrono::steady_clock> timestamp;
};

} // namespace KCT
