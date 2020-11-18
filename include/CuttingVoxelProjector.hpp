#pragma once

// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <CL/cl.hpp>
#include <iostream>
#include <limits>

// Internal libraries
#include "MATRIX/LUDoolittleForm.hpp"
#include "MATRIX/ProjectionMatrix.hpp"
#include "MATRIX/SquareMatrix.hpp"
#include "OPENCL/OpenCLManager.hpp"
#include "rawop.h"
#include "stringFormatter.h"

namespace CTL {

class CuttingVoxelProjector
{
public:
    /**
     * Class that encapsulates projector and backprojector implementation of Cutting Voxel Projector
     * and other algorithms.
     *
     * @param voxelSizeX X size of voxel
     * @param voxelSizeY Y size of voxel
     * @param voxelSizeZ Z size of voxel
     * @param pixelSizeX
     * @param pixelSizeY
     * @param xpath
     * @param debug Should debugging be used by suppliing source and -g as options
     * @param centerVoxelProjector Use center voxel projector istead of cutting voxels.
     * @param exactProjectionScaling
     */
    CuttingVoxelProjector(double voxelSizeX,
                          double voxelSizeY,
                          double voxelSizeZ,
                          double pixelSizeX,
                          double pixelSizeY,
                          std::string xpath,
                          bool debug,
                          bool centerVoxelProjector,
                          bool exactProjectionScaling = true)
        : voxelSizeX(voxelSizeX)
        , voxelSizeY(voxelSizeY)
        , voxelSizeZ(voxelSizeZ)
        , pixelSizeX(pixelSizeX)
        , pixelSizeY(pixelSizeY)
        , xpath(xpath)
        , debug(debug)
        , centerVoxelProjector(centerVoxelProjector)
        , exactProjectionScaling(exactProjectionScaling)
    {
        voxelSizes = cl_double3({ voxelSizeX, voxelSizeY, voxelSizeZ });
        pixelSizes = cl_double2({ pixelSizeX, pixelSizeY });
    }

    /** Initializes OpenCL.
     *
     * Initialization is done via C++ layer that works also with OpenCL 1.1.
     *
     *
     * @return
     * @see [OpenCL C++
     * manual](https://www.khronos.org/registry/OpenCL/specs/opencl-cplusplus-1.1.pdf)
     * @see [OpenCL C++
     * tutorial](http://simpleopencl.blogspot.com/2013/06/tutorial-simple-start-with-opencl-and-c.html)
     */
    int initializeOpenCL(uint32_t platformId = 0);

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
    int initializeOrUpdateVolumeBuffer(uint32_t volumeSizeX,
                                       uint32_t volumeSizeY,
                                       uint32_t volumeSizeZ,
                                       float* volumeArray = nullptr);

    int fillVolumeBufferByConstant(float constant);

    /**
     * Initialize projection buffer by given size.
     *
     * @param projectionSizeX
     * @param projectionSizeY
     * @param projectionSizeZ
     * @param projectionArray If is nullptr, initialize by zero.
     *
     * @return
     */
    int initializeOrUpdateProjectionBuffer(uint32_t projectionSizeX,
                                           uint32_t projectionSizeY,
                                           uint32_t projectionSizeZ,
                                           float* projectionArray = nullptr);

    int fillProjectionBufferByConstant(float constant);

    int project(float* projection,
                uint32_t pdimx,
                uint32_t pdimy,
                matrix::ProjectionMatrix P,
                float scalingFactor);
    int projectCos(float* projection,
                   uint32_t pdimx,
                   uint32_t pdimy,
                   matrix::ProjectionMatrix P,
                   float scalingFactor);
    int projectorWithoutScaling(float* projection,
                                uint32_t pdimx,
                                uint32_t pdimy,
                                double normalProjectionX,
                                double normalProjectionY,
                                double sourceToDetector,
                                matrix::ProjectionMatrix P);
    int projectExact(float* projection,
                     uint32_t pdimx,
                     uint32_t pdimy,
                     double normalProjectionX,
                     double normalProjectionY,
                     double sourceToDetector,
                     matrix::ProjectionMatrix P);
    int projectTA3(float* projection,
                   uint32_t pdimx,
                   uint32_t pdimy,
                   double normalProjectionX,
                   double normalProjectionY,
                   double sourceToDetector,
                   matrix::ProjectionMatrix P);

    int projectSiddon(float* projection,
                      uint32_t pdimx,
                      uint32_t pdimy,
                      matrix::ProjectionMatrix matrix,
                      float scalingFactor,
                      uint32_t probesPerEdge);

    double normSquare(float* projection, uint32_t pdimx, uint32_t pdimy);
    double normSquareDifference(float* projection, uint32_t pdimx, uint32_t pdimy);
    int backproject(float* volume,
                    uint32_t vdimx,
                    uint32_t vdimy,
                    uint32_t vdimz,
                    std::vector<matrix::ProjectionMatrix>& CMS,
                    uint64_t baseOffset = 0);

    int backproject_minmax(float* volume,
                           uint32_t vdimx,
                           uint32_t vdimy,
                           uint32_t vdimz,
                           std::vector<matrix::ProjectionMatrix>& CMS,
                           uint64_t baseOffset = 0);

private:
    const cl_float FLOATZERO = 0.0f;
    const cl_double DOUBLEZERO = 0.0;
    float FLOATONE = 1.0f;
    float* volume = nullptr;
    uint32_t volumeSizeX, volumeSizeY, volumeSizeZ;
    uint64_t totalVolumeSize, totalVolumeBufferSize;
    double voxelSizeX, voxelSizeY, voxelSizeZ;
    double pixelSizeX, pixelSizeY;
    uint32_t projectionSizeX, projectionSizeY, projectionSizeZ;
    uint64_t totalProjectionSize, totalProjectionBufferSize;
    std::string xpath; // Path where the program executes
    bool debug;
    bool centerVoxelProjector = false;
    cl_int3 vdims;
    cl_int2 pdims;
    cl_double3 voxelSizes;
    cl_double2 pixelSizes;
    bool useCVPProjector = true;
    bool exactProjectionScaling = true;
    bool useSidonProjector = false;
    cl_uint2 pixelGranularity = { 1, 1 };
    bool useTTProjector = false;

    std::shared_ptr<cl::Device> device = nullptr;
    std::shared_ptr<cl::Context> context = nullptr;
    std::shared_ptr<cl::CommandQueue> Q = nullptr;
    std::shared_ptr<cl::Buffer> volumeBuffer = nullptr;
    std::shared_ptr<cl::Buffer> projectionBuffer = nullptr;
    std::shared_ptr<cl::Buffer> tmpBuffer = nullptr;
    size_t tmpBuffer_size = 0;
    std::vector<cl_double16> invertProjectionMatrices(std::vector<matrix::ProjectionMatrix> CM);
    std::vector<float> computeScalingFactors(std::vector<matrix::ProjectionMatrix> PM);
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&>> FLOAT_CopyVector;
    int copyFloatVector(cl::Buffer& from, cl::Buffer& to, unsigned int size);

    // Projectors
    std::shared_ptr<cl::make_kernel<cl::Buffer&,
                                    cl::Buffer&,
                                    unsigned int&,
                                    cl_double16&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int3&,
                                    cl_double3&,
                                    cl_int2&,
                                    float&>>
        projector;
    std::shared_ptr<cl::make_kernel<cl::Buffer&,
                                    cl::Buffer&,
                                    unsigned int&,
                                    cl_double16&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int3&,
                                    cl_double3&,
                                    cl_int2&,
                                    float&>>
        projector_ta3;
    std::shared_ptr<cl::make_kernel<cl::Buffer&,
                                    cl::Buffer&,
                                    unsigned int&,
                                    cl_double16&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int3&,
                                    cl_double3&,
                                    cl_int2&,
                                    float&,
                                    cl_uint2&>>
        projector_sidon;
    // Backprojectors
    std::shared_ptr<cl::make_kernel<cl::Buffer&,
                                    cl::Buffer&,
                                    unsigned int&,
                                    cl_double16&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int3&,
                                    cl_double3&,
                                    cl_int2&,
                                    float&>>
        FLOATcutting_voxel_backproject;
    std::shared_ptr<cl::make_kernel<cl::Buffer&,
                                    cl::Buffer&,
                                    unsigned int&,
                                    cl_double16&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int3&,
                                    cl_double3&,
                                    cl_int2&,
                                    float&>>
        FLOATcutting_voxel_minmaxbackproject;
    std::shared_ptr<cl::make_kernel<cl::Buffer&,
                                    cl::Buffer&,
                                    unsigned int&,
                                    cl_double16&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int3&,
                                    cl_double3&,
                                    cl_int2&,
                                    float&>>
        FLOATta3_backproject;
    std::shared_ptr<cl::make_kernel<cl::Buffer&,
                                    cl::Buffer&,
                                    unsigned int&,
                                    cl_double16&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int3&,
                                    cl_double3&,
                                    cl_int2&,
                                    float&,
                                    cl_uint2&>>
        FLOATbackprojector_sidon;
    // Utils
    std::shared_ptr<cl::make_kernel<cl::Buffer&,
                                    unsigned int&,
                                    cl_double16&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_uint2&,
                                    float&>>
        scalingProjectionsCos;
    std::shared_ptr<
        cl::make_kernel<cl::Buffer&, unsigned int&, cl_uint2&, cl_double2&, cl_double2&, double&>>
        scalingProjectionsExact;
    std::shared_ptr<
        cl::make_kernel<cl::Buffer&, unsigned int&, cl_uint2&, cl_double2&, cl_double2&, double&>>
        scalingBackprojectionsExact;
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, float&>>
        FLOAT_addIntoFirstVectorSecondVectorScaled;
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&>> NormSquare;
    std::shared_ptr<cl::make_kernel<cl::Buffer&, float&, float&>> SubstituteLowerThan;
    std::shared_ptr<cl::make_kernel<cl::Buffer&, float&, float&>> SubstituteGreaterThan;
    std::shared_ptr<cl::make_kernel<cl::Buffer&>> ZeroInfinity;
};

} // namespace CTL
