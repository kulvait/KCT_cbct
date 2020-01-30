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

class GLSQRReconstructor
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
    GLSQRReconstructor(uint32_t pdimx,
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
                       std::string xpath,
                       bool debug,
                       uint32_t workGroupSize = 256,
                       uint32_t reportKthIteration = 0,
                       std::string progressBeginPath = "")
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
        , xpath(xpath)
        , debug(debug)
        , workGroupSize(workGroupSize)
        , reportKthIteration(reportKthIteration)
    {
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
        if(reportKthIteration > 0)
        {
            this->progressBeginPath = progressBeginPath;
        }
        timepoint = std::chrono::steady_clock::now();
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

    int initializeVectors(float* projection, float* volume);

    int reconstruct(std::shared_ptr<io::DenProjectionMatrixReader> matrices,
                    uint32_t maxIterations = 100,
                    float errCondition = 0.01);
    int reconstructTikhonov(std::shared_ptr<io::DenProjectionMatrixReader> matrices,
                            double lambda,
                            uint32_t maxIterations = 100,
                            float errCondition = 0.01);
	double adjointProductTest(std::shared_ptr<io::DenProjectionMatrixReader> matrices);

private:
    const cl_float FLOATZERO = 0.0;
    const cl_double DOUBLEZERO = 0.0;
    // Constructor defined variables
    const uint32_t pdimx, pdimy, pdimz;
    const double pixelSpacingX, pixelSpacingY;
    const uint32_t vdimx, vdimy, vdimz;
    const double voxelSpacingX, voxelSpacingY, voxelSpacingZ;
    const std::string xpath; // Path where the program executes
    const bool debug;
    const uint32_t workGroupSize = 256;
    uint32_t XDIM, BDIM, XDIM_ALIGNED, BDIM_ALIGNED, XDIM_REDUCED1, BDIM_REDUCED1,
        XDIM_REDUCED1_ALIGNED, BDIM_REDUCED1_ALIGNED, XDIM_REDUCED2, BDIM_REDUCED2,
        XDIM_REDUCED2_ALIGNED, BDIM_REDUCED2_ALIGNED;
    uint32_t reportKthIteration = 0;
    std::string progressBeginPath = "";

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

    void setTimepoint();
    void reportTime(std::string msg);
    void writeVolume(cl::Buffer& X, std::string path);
    void writeProjections(cl::Buffer& B, std::string path);
    // Backprojecting from B to X
    int backproject(cl::Buffer& B,
                    cl::Buffer& X,
                    std::vector<matrix::ProjectionMatrix>& V,
                    std::vector<cl_double16>& invertedProjectionMatrices,
                    std::vector<float>& scalingFactors);
    int project(cl::Buffer& B,
                cl::Buffer& X,
                std::vector<matrix::ProjectionMatrix>& V,
                std::vector<cl_double16>& invertedProjectionMatrices,
                std::vector<float>& scalingFactors);
    int copyFloatVector(cl::Buffer& from, cl::Buffer& to, unsigned int size);
    int scaleFloatVector(cl::Buffer& v, float f, unsigned int size);
    int
    addIntoFirstVectorSecondVectorScaled(cl::Buffer& a, cl::Buffer& b, float f, unsigned int size);
    int
    addIntoFirstVectorScaledSecondVector(cl::Buffer& a, cl::Buffer& b, float f, unsigned int size);
    std::vector<matrix::ProjectionMatrix>
    encodeProjectionMatrices(std::shared_ptr<io::DenProjectionMatrixReader> pm);
    std::vector<cl_double16> inverseProjectionMatrices(std::vector<matrix::ProjectionMatrix>);
    std::vector<float> computeScalingFactors(std::vector<matrix::ProjectionMatrix> PM);

    float* x = nullptr; // Volume data
    float* b = nullptr; // Projection data

    std::shared_ptr<cl::Device> device = nullptr;
    std::shared_ptr<cl::Context> context = nullptr;
    std::shared_ptr<cl::Image3D> volumeImage = nullptr;
    std::shared_ptr<cl::CommandQueue> Q = nullptr;
    std::shared_ptr<cl::Buffer> b_buf = nullptr, ba_buf = nullptr, bb_buf = nullptr,
                                bc_buf = nullptr, bd_buf = nullptr, be_buf = nullptr,
                                tmp_b_red1 = nullptr, tmp_b_red2 = nullptr, tmp_b_buf = nullptr;
    std::shared_ptr<cl::Buffer> x_buf = nullptr, xa_buf = nullptr, xb_buf = nullptr,
                                xc_buf = nullptr, xd_buf = nullptr, xe_buf = nullptr,
                                xf_buf = nullptr, xg_buf = nullptr, xh_buf = nullptr,
                                xi_buf = nullptr, xj_buf = nullptr, tmp_x_red1 = nullptr,
                                tmp_x_red2 = nullptr, xk_buf = nullptr, xl_buf = nullptr,
                                xm_buf = nullptr, xn_buf = nullptr;

    // Functions
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
    std::shared_ptr<cl::make_kernel<cl::Buffer&, float&>> FLOAT_scaleVector;
    std::shared_ptr<
        cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::LocalSpaceArg&, unsigned int&>>
        ScalarProductPartial_barier;
    std::shared_ptr<cl::make_kernel<cl::Buffer&,
                                    unsigned int&,
                                    cl_double16&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int2&,
                                    float&>>
        scalingProjections;
    std::chrono::time_point<std::chrono::steady_clock> timepoint;
};

} // namespace CTL
