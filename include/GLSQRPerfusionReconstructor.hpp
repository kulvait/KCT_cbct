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

class GLSQRPerfusionReconstructor
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
    GLSQRPerfusionReconstructor(uint32_t pdimx,
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
                                uint32_t reportEachK = 0,
                                std::string progressBeginPath = "",
                                bool sidon = false)
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
        , reportEachK(reportEachK)
        , sidon(sidon)
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
        if(reportEachK > 0)
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

    int initializeData(std::vector<float*> projections,
                       std::vector<float*> basisVectorValues,
                       std::vector<float*> volumes);

    int reconstruct(std::shared_ptr<io::DenProjectionMatrixReader> matrices,
                    uint32_t maxIterations = 1000,
                    float errCondition = 0.001);
    int updateB(std::vector<float*> projections);
    int updateX(std::vector<float*> volumes);
    int projectXtoB(std::shared_ptr<io::DenProjectionMatrixReader> matrices);
    int backprojectBtoX(std::shared_ptr<io::DenProjectionMatrixReader> matrices);
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
    uint32_t reportEachK = 0;
    std::string progressBeginPath = "";

    float normBBuffer_barier(cl::Buffer& B);
    float normXBuffer_barier(cl::Buffer& X);
    float normBBuffer_frame(cl::Buffer& B);
    float normXBuffer_frame(cl::Buffer& X);
    double normBBuffer_barier_double(cl::Buffer& B);
    double normXBuffer_barier_double(cl::Buffer& X);
    double normBBuffer_barier_double(std::vector<std::shared_ptr<cl::Buffer>>& B);
    double normXBuffer_barier_double(std::vector<std::shared_ptr<cl::Buffer>>& X);
    double normBBuffer_frame_double(cl::Buffer& B);
    double normXBuffer_frame_double(cl::Buffer& X);
    double scalarProductBBuffer_barier_double(cl::Buffer& A, cl::Buffer& B);
    double scalarProductBBuffer_barier_double(std::vector<std::shared_ptr<cl::Buffer>>& A,
                                              std::vector<std::shared_ptr<cl::Buffer>>& B);
    double scalarProductXBuffer_barier_double(cl::Buffer& A, cl::Buffer& B);
    double scalarProductXBuffer_barier_double(std::vector<std::shared_ptr<cl::Buffer>>& A,
                                              std::vector<std::shared_ptr<cl::Buffer>>& B);

    void setTimepoint();
    void reportTime(std::string msg);
    void writeVolume(std::vector<std::shared_ptr<cl::Buffer>>& X, std::string path);
    void writeProjections(std::vector<std::shared_ptr<cl::Buffer>>& B, std::string path);
    // Backprojecting from B to X
    int backproject(std::vector<std::shared_ptr<cl::Buffer>>& B,
                    std::vector<std::shared_ptr<cl::Buffer>>& X,
                    std::vector<matrix::ProjectionMatrix>& V,
                    std::vector<cl_double16>& invertedProjectionMatrices,
                    std::vector<float>& scalingFactors);
    int project(std::vector<std::shared_ptr<cl::Buffer>>& X,
                std::vector<std::shared_ptr<cl::Buffer>>& B,
                std::vector<matrix::ProjectionMatrix>& V,
                std::vector<cl_double16>& invertedProjectionMatrices,
                std::vector<float>& scalingFactors);
    int zeroFloatVector(cl::Buffer& b, unsigned int size);
    int copyFloatVector(cl::Buffer& from, cl::Buffer& to, unsigned int size);
    int copyFloatVector(std::vector<std::shared_ptr<cl::Buffer>>& from,
                        std::vector<std::shared_ptr<cl::Buffer>>& to,
                        unsigned int size);
    int
    addIntoFirstVectorSecondVectorScaled(cl::Buffer& a, cl::Buffer& b, float f, unsigned int size);
    int
    addIntoFirstVectorScaledSecondVector(cl::Buffer& a, cl::Buffer& b, float f, unsigned int size);
    int addIntoFirstVectorSecondVectorScaled(std::vector<std::shared_ptr<cl::Buffer>>& A,
                                             std::vector<std::shared_ptr<cl::Buffer>>& B,
                                             float f,
                                             unsigned int size);
    int addIntoFirstVectorSecondVectorScaledOffset(
        cl::Buffer& a, cl::Buffer& b, float f, unsigned int size, unsigned int offset);
    int addIntoFirstVectorScaledSecondVector(std::vector<std::shared_ptr<cl::Buffer>>& A,
                                             std::vector<std::shared_ptr<cl::Buffer>>& B,
                                             float f,
                                             unsigned int size);
    int scaleFloatVector(cl::Buffer& v, float f, unsigned int size);
    int scaleFloatVector(std::vector<std::shared_ptr<cl::Buffer>>& A, float f, unsigned int size);
    void zeroXBuffers(std::vector<std::shared_ptr<cl::Buffer>>& X);
    void zeroBBuffers(std::vector<std::shared_ptr<cl::Buffer>>& B);
    std::vector<matrix::ProjectionMatrix>
    encodeProjectionMatrices(std::shared_ptr<io::DenProjectionMatrixReader> pm);
    std::vector<float> computeScalingFactors(std::vector<matrix::ProjectionMatrix> PM);
    std::vector<cl_double16> inverseProjectionMatrices(std::vector<matrix::ProjectionMatrix> CM);

    std::vector<float*> x; // Volume data
    std::vector<float*>
        basisFunctionsValues; // Basis functions at time points of particular projections
    std::vector<float*> b; // Projection data

    std::shared_ptr<cl::Device> device = nullptr;
    std::shared_ptr<cl::Context> context = nullptr;
    std::shared_ptr<cl::Image3D> volumeImage = nullptr;
    std::shared_ptr<cl::CommandQueue> Q = nullptr;
    std::vector<std::shared_ptr<cl::Buffer>> b_buf, ba_buf, bb_buf, bc_buf, bd_buf, be_buf,
        tmp_b_buf;
    std::vector<std::shared_ptr<cl::Buffer>> x_buf, xa_buf, xb_buf, xc_buf, xd_buf, xe_buf, xf_buf,
        xg_buf, xh_buf, xi_buf, xj_buf, xk_buf, xl_buf, xm_buf, xn_buf, xB_tmp_buf;
    std::shared_ptr<cl::Buffer> tmp_b = nullptr, tmp_b_red1 = nullptr, tmp_b_red2 = nullptr;
    std::shared_ptr<cl::Buffer> tmp_x = nullptr, tmp_x_red1 = nullptr, tmp_x_red2 = nullptr;

    // Functions
    std::shared_ptr<cl::make_kernel<cl::Buffer&, float&>> FLOAT_scaleVector;
    std::shared_ptr<cl::make_kernel<cl::Buffer&>> FLOAT_ZeroVector;
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, float&>>
        FLOAT_addIntoFirstVectorSecondVectorScaled;
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, float&, unsigned int&>>
        FLOAT_addIntoFirstVectorSecondVectorScaledOffset;
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
    bool sidon = false;
};

} // namespace CTL
