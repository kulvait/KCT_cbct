#pragma once

// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <CL/cl.hpp>
#include <chrono>
#include <ctime>
#include <iostream>

// Internal libraries
#include "DEN/DenProjectionMatrixReader.hpp"
#include "MATRIX/ProjectionMatrix.hpp"
#include "OPENCL/OpenCLManager.hpp"
#include "rawop.h"
#include "stringFormatter.h"

namespace CTL {

class BaseReconstructor
{
public:
    BaseReconstructor(uint32_t pdimx,
                      uint32_t pdimy,
                      uint32_t pdimz,
                      double pixelSpacingX,
                      double pixelSpacingY,
                      uint32_t vdimx,
                      uint32_t vdimy,
                      uint32_t vdimz,
                      double voxelSpacingX,
                      double voxelSpacingY,
                      double voxelSpacingZ,
                      uint32_t workGroupSize = 256)
        : pdimx(pdimx)
        , pdimy(pdimy)
        , pdimz(pdimz)
        , pixelSpacingX(pixelSpacingX)
        , pixelSpacingY(pixelSpacingY)
        , vdimx(vdimx)
        , vdimy(vdimy)
        , vdimz(vdimz)
        , voxelSpacingX(voxelSpacingX)
        , voxelSpacingY(voxelSpacingY)
        , voxelSpacingZ(voxelSpacingZ)
        , workGroupSize(workGroupSize)
    {
        pdims = cl_int2({ int(pdimx), int(pdimy) });
        pdims_uint = cl_uint2({ pdimx, pdimy });
        vdims = cl_int3({ int(vdimx), int(vdimy), int(vdimz) });
        pixelSizes = cl_double2({ pixelSpacingX, pixelSpacingY });
        voxelSizes = cl_double3({ voxelSpacingX, voxelSpacingY, voxelSpacingZ });
        uint32_t UINT32_MAXXX = ((uint32_t)-1);
        uint64_t xdim = uint64_t(vdimx) * uint64_t(vdimy) * uint64_t(vdimz);
        uint64_t bdim = uint64_t(pdimx) * uint64_t(pdimy) * uint64_t(pdimz);
        uint64_t xdim_aligned = xdim + (workGroupSize - xdim % workGroupSize) % workGroupSize;
        uint64_t bdim_aligned = bdim + (workGroupSize - bdim % workGroupSize) % workGroupSize;
        XDIM = xdim;
        XDIM_ALIGNED = xdim_aligned;
        XDIM_REDUCED1 = xdim_aligned / workGroupSize;
        XDIM_REDUCED1_ALIGNED
            = XDIM_REDUCED1 + (workGroupSize - XDIM_REDUCED1 % workGroupSize) % workGroupSize;
        XDIM_REDUCED2 = XDIM_REDUCED1_ALIGNED / workGroupSize;
        XDIM_REDUCED2_ALIGNED
            = XDIM_REDUCED2 + (workGroupSize - XDIM_REDUCED2 % workGroupSize) % workGroupSize;
        BDIM = bdim;
        BDIM_ALIGNED = bdim_aligned;
        BDIM_REDUCED1 = bdim_aligned / workGroupSize;
        BDIM_REDUCED1_ALIGNED
            = BDIM_REDUCED1 + (workGroupSize - BDIM_REDUCED1 % workGroupSize) % workGroupSize;
        BDIM_REDUCED2 = BDIM_REDUCED1_ALIGNED / workGroupSize;
        BDIM_REDUCED2_ALIGNED
            = BDIM_REDUCED2 + (workGroupSize - BDIM_REDUCED2 % workGroupSize) % workGroupSize;
        if(xdim_aligned > UINT32_MAXXX)
        {
            std::string err = "Too big dimensions";
            LOGE << err;
            throw std::runtime_error(err);
        } else if(xdim_aligned * 4 > UINT32_MAXXX)
        {
            LOGI << "Beware buffer overflows for x buffer.";
        }
        if(bdim_aligned > UINT32_MAXXX)
        {
            std::string err = "Too big dimensions";
            LOGE << err;
            throw std::runtime_error(err);
        } else if(bdim_aligned * 4 > UINT32_MAXXX)
        {
            LOGI << "Beware buffer overflows for b buffer.";
        }
        timepoint = std::chrono::steady_clock::now();
    }

    void initializeCVPProjector(bool useExactScaling);
    void initializeSidonProjector(uint32_t probesPerEdgeX, uint32_t probesPerEdgeY);
    void initializeTTProjector();

    /** Initializes OpenCL.
     * @brief Initialize OpenCL engine. Before calling this function one of initializeProjector
     * function should be called to select the projector code to compile into openCL kernel.
     *
     * @param xpath
     * @param platformId
     *
     * @return
     * @see [OpenCL C++
     * manual](https://www.khronos.org/registry/OpenCL/specs/opencl-cplusplus-1.1.pdf)
     * @see [OpenCL C++
     * tutorial](http://simpleopencl.blogspot.com/2013/06/tutorial-simple-start-with-opencl-and-c.html)
     */
    int initializeOpenCL(uint32_t platformId,
                         uint32_t* deviceIds,
                         uint32_t deviceIdsLength,
                         std::string xpath,
                         bool debug);

    int initializeVectors(float* projection, float* volume);
    int allocateXBuffers(uint32_t xBufferCount);
    int allocateBBuffers(uint32_t bBufferCount);
    int allocateTmpXBuffers(uint32_t xBufferCount);
    int allocateTmpBBuffers(uint32_t bBufferCount);
    std::shared_ptr<cl::Buffer> getBBuffer(uint32_t i);
    std::shared_ptr<cl::Buffer> getXBuffer(uint32_t i);
    std::shared_ptr<cl::Buffer> getTmpBBuffer(uint32_t i);
    std::shared_ptr<cl::Buffer> getTmpXBuffer(uint32_t i);

    virtual int reconstruct(std::shared_ptr<io::DenProjectionMatrixReader> matrices,
                            uint32_t maxItterations,
                            float minDiscrepancyError)
        = 0;
    double adjointProductTest(std::shared_ptr<io::DenProjectionMatrixReader> matrices);

protected:
    const cl_float FLOATZERO = 0.0;
    const cl_double DOUBLEZERO = 0.0;
    float FLOATONE = 1.0f;
    cl_int2 pdims;
    cl_uint2 pdims_uint;
    cl_int3 vdims;
    cl_double2 pixelSizes;
    cl_double3 voxelSizes;
    // Constructor defined variables
    const uint32_t pdimx, pdimy, pdimz;
    const double pixelSpacingX, pixelSpacingY;
    const uint32_t vdimx, vdimy, vdimz;
    const double voxelSpacingX, voxelSpacingY, voxelSpacingZ;
    const uint32_t workGroupSize = 256;
    uint32_t XDIM, BDIM, XDIM_ALIGNED, BDIM_ALIGNED, XDIM_REDUCED1, BDIM_REDUCED1,
        XDIM_REDUCED1_ALIGNED, BDIM_REDUCED1_ALIGNED, XDIM_REDUCED2, BDIM_REDUCED2,
        XDIM_REDUCED2_ALIGNED, BDIM_REDUCED2_ALIGNED;

    // Variables for projectors and openCL initialization
    bool openCLinitialized = false;
    bool useCVPProjector = true;
    bool exactProjectionScaling = true;
    bool useSidonProjector = false;
    cl_uint2 pixelGranularity = { 1, 1 };
    bool useTTProjector = false;

    uint32_t xBufferCount, bBufferCount, tmpXBufferCount, tmpBBufferCount;

    // Class functions
    void setTimepoint();
    void reportTime(std::string msg);
    void writeVolume(cl::Buffer& X, std::string path);
    void writeProjections(cl::Buffer& B, std::string path);
    std::vector<cl_double16> inverseProjectionMatrices(std::vector<matrix::ProjectionMatrix>);

    // Functions to manipulate with buffers
    float normBBuffer_barier(cl::Buffer& B);
    float normXBuffer_barier(cl::Buffer& X);
    float normBBuffer_frame(cl::Buffer& B);
    float normXBuffer_frame(cl::Buffer& X);
    double normBBuffer_barier_double(cl::Buffer& B);
    double normXBuffer_barier_double(cl::Buffer& X);
    double normBBuffer_frame_double(cl::Buffer& B);
    double normXBuffer_frame_double(cl::Buffer& X);
    double scalarProductBBuffer_barier_double(cl::Buffer& A, cl::Buffer& B);
    double scalarProductXBuffer_barier_double(cl::Buffer& A, cl::Buffer& B);
    int copyFloatVector(cl::Buffer& from, cl::Buffer& to, unsigned int size);
    int scaleFloatVector(cl::Buffer& v, float f, unsigned int size);
    int copyFloatVectorOffset(cl::Buffer& from,
                              unsigned int from_offset,
                              cl::Buffer& to,
                              unsigned int to_offset,
                              unsigned int size);
    int
    addIntoFirstVectorSecondVectorScaled(cl::Buffer& a, cl::Buffer& b, float f, unsigned int size);
    int
    addIntoFirstVectorScaledSecondVector(cl::Buffer& a, cl::Buffer& b, float f, unsigned int size);
    std::vector<matrix::ProjectionMatrix>
    encodeProjectionMatrices(std::shared_ptr<io::DenProjectionMatrixReader> pm);
    std::vector<float> computeScalingFactors(std::vector<matrix::ProjectionMatrix> PM);

    // Backprojecting from B to X
    int backproject(cl::Buffer& B,
                    cl::Buffer& X,
                    std::vector<matrix::ProjectionMatrix>& V,
                    std::vector<cl_double16>& invertedProjectionMatrices,
                    std::vector<float>& scalingFactors);
    int project(cl::Buffer& X,
                cl::Buffer& B,
                std::vector<matrix::ProjectionMatrix>& V,
                std::vector<cl_double16>& invertedProjectionMatrices,
                std::vector<float>& scalingFactors);

    float* x = nullptr; // Volume data
    float* b = nullptr; // Projection data

    // OpenCL objects
    std::shared_ptr<cl::Device> device = nullptr;
    std::shared_ptr<cl::Context> context = nullptr;
    std::shared_ptr<cl::CommandQueue> Q = nullptr;
    std::shared_ptr<cl::Buffer> b_buf = nullptr, tmp_b_red1 = nullptr, tmp_b_red2 = nullptr;
    std::shared_ptr<cl::Buffer> x_buf = nullptr, tmp_x_red1 = nullptr, tmp_x_red2 = nullptr;
    // tmp_b_buf for rescaling, tmp_x_buf for LSQR
    std::shared_ptr<cl::Buffer> tmp_x_buf = nullptr, tmp_b_buf = nullptr;
    std::vector<std::shared_ptr<cl::Buffer>> x_buffers, tmp_x_buffers;
    std::vector<std::shared_ptr<cl::Buffer>> b_buffers, tmp_b_buffers;

    // OpenCL functiors
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, float&>>
        FLOAT_addIntoFirstVectorSecondVectorScaled;
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, float&>>
        FLOAT_addIntoFirstVectorScaledSecondVector;
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&>> FLOAT_NormSquare;
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&>> FLOAT_SumPartial;
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::LocalSpaceArg&, unsigned int&>>
        FLOAT_NormSquare_barier;
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::LocalSpaceArg&, unsigned int&>>
        FLOAT_Sum_barier;
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&>> NormSquare;
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&>> SumPartial;
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::LocalSpaceArg&, unsigned int&>>
        NormSquare_barier;
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::LocalSpaceArg&, unsigned int&>>
        Sum_barier;
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&>> FLOAT_CopyVector;
    std::shared_ptr<cl::make_kernel<cl::Buffer&, unsigned int&, cl::Buffer&, unsigned int&>>
        FLOAT_CopyVector_offset;
    std::shared_ptr<cl::make_kernel<cl::Buffer&, float&>> FLOAT_scaleVector;
    std::shared_ptr<
        cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::LocalSpaceArg&, unsigned int&>>
        ScalarProductPartial_barier;
    // CVP
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
        FLOATcutting_voxel_project;
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
    std::shared_ptr<
        cl::make_kernel<cl::Buffer&, unsigned int&, cl_uint2&, cl_double2&, cl_double2&, double&>>
        scalingProjectionsExact;
    std::shared_ptr<cl::make_kernel<cl::Buffer&,
                                    unsigned int&,
                                    cl_double16&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_uint2&,
                                    float&>>
        scalingProjectionsCos;
    // TT
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
        FLOATta3_project;
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
    // SIDON
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
        FLOATprojector_sidon;
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

    std::chrono::time_point<std::chrono::steady_clock> timepoint;
};

} // namespace CTL
