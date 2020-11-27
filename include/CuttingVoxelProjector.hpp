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

class CuttingVoxelProjector : public virtual Kniha
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
    CuttingVoxelProjector(uint64_t pdimx,
                          uint64_t pdimy,
                          uint64_t vdimx,
                          uint64_t vdimy,
                          uint64_t vdimz)
        : pdimx(pdimx)
        , pdimy(pdimy)
        , vdimx(vdimx)
        , vdimy(vdimy)
        , vdimz(vdimz)
    {
        pdimz = 1; // Default
        pdims = cl_int2({ int(pdimx), int(pdimy) });
        pdims_uint = cl_uint2({ uint32_t(pdimx), uint32_t(pdimy) });
        vdims = cl_int3({ int(vdimx), int(vdimy), int(vdimz) });
        totalVoxelNum = vdimx * vdimy * vdimz;
        totalVolumeBufferSize = totalVoxelNum * sizeof(float);
        frameSize = pdimx * pdimy;
        timestamp = std::chrono::steady_clock::now();
    }

    void initializeCVPProjector(bool useExactScaling);
    void initializeSidonProjector(uint32_t probesPerEdgeX, uint32_t probesPerEdgeY);
    void initializeTTProjector();
    void initializeAllAlgorithms();
    int problemSetup(double voxelSizeX,
                     double voxelSizeY,
                     double voxelSizeZ,
                     double volumeCenterX,
                     double volumeCenterY,
                     double volumeCenterZ);
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

    int project(float* projection, std::shared_ptr<matrix::CameraI> pm);
    int projectCos(float* projection, std::shared_ptr<matrix::CameraI> pm);
    int projectorWithoutScaling(float* projection, std::shared_ptr<matrix::CameraI> pm);

    int projectExact(float* projection, std::shared_ptr<matrix::CameraI> pm);
    int projectTA3(float* projection, std::shared_ptr<matrix::CameraI> pm);

    int projectSidon(float* projection, std::shared_ptr<matrix::CameraI> pm);

    double normSquare(float* projection, uint32_t pdimx, uint32_t pdimy);
    double normSquareDifference(float* projection, uint32_t pdimx, uint32_t pdimy);
    int backproject(float* volume,
                    std::vector<std::shared_ptr<matrix::CameraI>>& cameraVector,
                    uint64_t baseOffset = 0);

    int backproject_minmax(float* volume,
                           std::vector<std::shared_ptr<matrix::CameraI>>& cameraVector,
                           uint64_t baseOffset = 0);

private:
    const cl_float FLOATZERO = 0.0f;
    const cl_double DOUBLEZERO = 0.0;
    float FLOATONE = 1.0f;
    float* volume = nullptr;
    uint32_t pdimx, pdimy, pdimz;
    uint32_t vdimx, vdimy, vdimz;
    uint64_t totalVoxelNum, totalVolumeBufferSize;
    uint64_t frameSize;
    uint64_t totalPixelNum, totalProjectionBufferSize;

    bool centerVoxelProjector = false;
    cl_int3 vdims;
    cl_int2 pdims;
    cl_uint2 pdims_uint;
    cl_double3 voxelSizes;
    cl_double3 volumeCenter;
    bool useCVPProjector = true;
    bool exactProjectionScaling = true;
    bool useSidonProjector = false;
    cl_uint2 pixelGranularity = { 1, 1 };
    bool useTTProjector = false;

    std::shared_ptr<cl::Buffer> volumeBuffer = nullptr;
    std::shared_ptr<cl::Buffer> projectionBuffer = nullptr;
    std::shared_ptr<cl::Buffer> tmpBuffer = nullptr;
    size_t tmpBuffer_size = 0;
    std::vector<cl_double16> invertProjectionMatrices(std::vector<matrix::ProjectionMatrix> CM);
    std::vector<float> computeScalingFactors(std::vector<matrix::ProjectionMatrix> PM);
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&>> FLOAT_CopyVector;
    std::chrono::time_point<std::chrono::steady_clock> timestamp;
};

} // namespace CTL
