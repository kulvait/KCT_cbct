#pragma once

// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <CL/cl.hpp>
#include <iostream>
#include <limits>

// Internal libraries
#include "Kniha.hpp"
#include "MATRIX/LUDoolittleForm.hpp"
#include "MATRIX/ProjectionMatrix.hpp"
#include "MATRIX/SquareMatrix.hpp"
#include "OPENCL/OpenCLManager.hpp"
#include "rawop.h"
#include "stringFormatter.h"

namespace CTL {

class VolumeConvolutionOperator : public virtual Kniha
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
    VolumeConvolutionOperator(uint64_t vdimx,
                              uint64_t vdimy,
                              uint64_t vdimz,
                              cl::NDRange projectorLocalNDRange = cl::NDRange())
        : vdimx(vdimx)
        , vdimy(vdimy)
        , vdimz(vdimz)
    {
        vdims = cl_int3({ int(vdimx), int(vdimy), int(vdimz) });
        totalVoxelNum = vdimx * vdimy * vdimz;
        totalVolumeBufferSize = totalVoxelNum * sizeof(float);
        timestamp = std::chrono::steady_clock::now();
        std::size_t projectorLocalNDRangeDim = projectorLocalNDRange.dimensions();
        if(projectorLocalNDRangeDim == 3)
        {
            if(projectorLocalNDRange[0] == 0 && projectorLocalNDRange[1] == 0
               && projectorLocalNDRange[2] == 0)
            {
                this->projectorLocalNDRange = cl::NDRange();
                this->projectorLocalNDRangeBarrier = cl::NDRange();
            } else if(projectorLocalNDRange[0] == 0 || projectorLocalNDRange[1] == 0
                      || projectorLocalNDRange[2] == 0)
            {
                this->projectorLocalNDRange = guessProjectionLocalNDRange(false);
                this->projectorLocalNDRangeBarrier = guessProjectionLocalNDRange(true);
            } else
            {
                this->projectorLocalNDRange = projectorLocalNDRange;
                this->projectorLocalNDRangeBarrier = projectorLocalNDRange;
            }
        } else
        {
            if(projectorLocalNDRangeDim != 0)
            {
                LOGE << io::xprintf(
                    "Wrong specification of projectorLocalNDRange, trying guessing!");
            }
            this->projectorLocalNDRange = guessProjectionLocalNDRange(false);
            this->projectorLocalNDRangeBarrier = guessProjectionLocalNDRange(true);
        }
    }

    cl::NDRange guessProjectionLocalNDRange(bool barrierCalls);

    void initializeConvolution();
    void initializeAllAlgorithms();
    int problemSetup(double voxelSizeX, double voxelSizeY, double voxelSizeZ);
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
    int initializeOrUpdateOutputBuffer();
    int initializeOrUpdateGradientOutputBuffers();

    int convolve(std::string kernelName, float* volume);
    int sobelGradient3D(cl_float3 voxelSizes, float* vx, float* vy, float* vz);

private:
    const cl_float FLOATZERO = 0.0f;
    const cl_double DOUBLEZERO = 0.0;
    float FLOATONE = 1.0f;
    float* volume = nullptr;
    uint32_t vdimx, vdimy, vdimz;
    uint64_t totalVoxelNum, totalVolumeBufferSize, totalOutputBufferSize;
    uint64_t frameSize;
    cl::NDRange projectorLocalNDRange;
    cl::NDRange projectorLocalNDRangeBarrier;

    cl_int3 vdims;
    cl_double3 voxelSizes;
    cl_double3 volumeCenter;
    bool useBarrierImplementation = false;
    uint32_t LOCALARRAYSIZE = 0;

    std::shared_ptr<cl::Buffer> volumeBuffer = nullptr;
    std::shared_ptr<cl::Buffer> outputBuffer = nullptr;
    uint64_t totalOutputGradientBuffersSize;
    std::shared_ptr<cl::Buffer> outputGradientX = nullptr;
    std::shared_ptr<cl::Buffer> outputGradientY = nullptr;
    std::shared_ptr<cl::Buffer> outputGradientZ = nullptr;
    size_t tmpBuffer_size = 0;
    std::vector<cl_double16> invertProjectionMatrices(std::vector<matrix::ProjectionMatrix> CM);
    std::vector<float> computeScalingFactors(std::vector<matrix::ProjectionMatrix> PM);
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&>> FLOAT_CopyVector;
    std::chrono::time_point<std::chrono::steady_clock> timestamp;
};

} // namespace CTL
