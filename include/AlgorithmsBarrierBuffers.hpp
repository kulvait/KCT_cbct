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
#include "Kniha.hpp"
#include "MATRIX/LightProjectionMatrix.hpp"
#include "MATRIX/ProjectionMatrix.hpp"
#include "MATRIX/utils.hpp"
#include "OPENCL/OpenCLManager.hpp"
#include "rawop.h"
#include "stringFormatter.h"

using namespace CTL::matrix;
namespace CTL {

class AlgorithmsBarrierBuffers : public virtual Kniha
{
public:
    AlgorithmsBarrierBuffers(uint32_t pdimx,
                            uint32_t pdimy,
                            uint32_t pdimz,
                            uint32_t vdimx,
                            uint32_t vdimy,
                            uint32_t vdimz,
                            uint32_t workGroupSize = 256)
        : pdimx(pdimx)
        , pdimy(pdimy)
        , pdimz(pdimz)
        , vdimx(vdimx)
        , vdimy(vdimy)
        , vdimz(vdimz)
        , workGroupSize(workGroupSize)
    {
        const uint32_t UINT32_MAXXX = ((uint32_t)-1);
        const uint64_t xdim = uint64_t(vdimx) * uint64_t(vdimy) * uint64_t(vdimz);
        const uint64_t bdim = uint64_t(pdimx) * uint64_t(pdimy) * uint64_t(pdimz);
        const uint64_t xdim_aligned = xdim + (workGroupSize - xdim % workGroupSize) % workGroupSize;
        const uint64_t bdim_aligned = bdim + (workGroupSize - bdim % workGroupSize) % workGroupSize;
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
        CLINCLUDEutils();
    }

protected:
    const uint32_t pdimx, pdimy, pdimz, vdimx, vdimy, vdimz;
    const uint32_t workGroupSize = 256;
    uint32_t XDIM, BDIM, XDIM_ALIGNED, BDIM_ALIGNED, XDIM_REDUCED1, BDIM_REDUCED1,
        XDIM_REDUCED1_ALIGNED, BDIM_REDUCED1_ALIGNED, XDIM_REDUCED2, BDIM_REDUCED2,
        XDIM_REDUCED2_ALIGNED, BDIM_REDUCED2_ALIGNED;

    // Initialization of buffers
    int initializeAlgorithmsBuffers();
    bool algorithmsBuffersInitialized = false;

    // Functions to manipulate with buffers
    float normBBuffer_barrier(cl::Buffer& B);
    float normXBuffer_barrier(cl::Buffer& X);
    float normBBuffer_frame(cl::Buffer& B);
    float normXBuffer_frame(cl::Buffer& X);
    float sumBBuffer_barrier_float(cl::Buffer& B);
    float sumXBuffer_barrier_float(cl::Buffer& X);
    float maxBBuffer_barrier_float(cl::Buffer& B);
    float maxXBuffer_barrier_float(cl::Buffer& X);
    double normBBuffer_barrier_double(cl::Buffer& B);
    double normXBuffer_barrier_double(cl::Buffer& X);
    double normBBuffer_frame_double(cl::Buffer& B);
    double normXBuffer_frame_double(cl::Buffer& X);
    double scalarProductBBuffer_barrier_double(cl::Buffer& A, cl::Buffer& B);
    double scalarProductXBuffer_barrier_double(cl::Buffer& A, cl::Buffer& B);

    /**
     * Copy given float vector into the buffer. The buffer must have appropriate size.
     *
     * @param X Buffer
     * @param v vector
     * @param size size
     *
     * @return
     */
    int vectorIntoBuffer(cl::Buffer X, float* v, std::size_t size);

    std::shared_ptr<cl::Buffer> tmp_b_red1 = nullptr, tmp_b_red2 = nullptr;
    std::shared_ptr<cl::Buffer> tmp_x_red1 = nullptr, tmp_x_red2 = nullptr;
};

} // namespace CTL
