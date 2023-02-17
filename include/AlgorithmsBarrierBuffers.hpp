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
#include "PROG/KCTException.hpp"
#include "rawop.h"
#include "stringFormatter.h"

using namespace KCT::matrix;
namespace KCT {

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
        const uint64_t xframesize = uint64_t(vdimx) * uint64_t(vdimy);
        const uint64_t bframesize = uint64_t(pdimx) * uint64_t(pdimy);
        const uint64_t xdim_aligned = xdim + (workGroupSize - xdim % workGroupSize) % workGroupSize;
        const uint64_t bdim_aligned = bdim + (workGroupSize - bdim % workGroupSize) % workGroupSize;
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
        std::string err;
        if(xframesize > UINT32_MAXXX)
        {
            err = io::xprintf(
                "Algorithms are based on the assumption that the x y volume slice can be "
                "indexed by uint32_t but xframesize=%lu that is bigger than UINT32_MAXXX=%lu",
                xframesize, UINT32_MAXXX);
            KCTERR(err);
        }
        if(bframesize > UINT32_MAXXX)
        {
            err = io ::xprintf(
                "Algorithms are based on the assumption that the projection size can be "
                "indexed by uint32_t but bframesize=%lu that is bigger than UINT32_MAXXX=%lu",
                bframesize, UINT32_MAXXX);
            KCTERR(err);
        }
        if(XDIM_REDUCED2 > UINT32_MAXXX)
        {
            err = io::xprintf(
                "Barrier algorithms are based on the assumption that XDIM_REDUCED2=%lu fits into "
                "UINT32_MAXXX=%u. In the last step they call algFLOATvector_SumPartial.",
                XDIM_REDUCED2, UINT32_MAXXX);
            KCTERR(err);
        }
        if(BDIM_REDUCED2 > UINT32_MAXXX)
        {
            err = io::xprintf("BDIM_REDUCED2_ALIGNED * 8=%lu is bigger than UINT32_MAXXX=%u",
                              BDIM_REDUCED2_ALIGNED, UINT32_MAXXX);
            KCTERR(err);
        }
        if(xdim_aligned > UINT32_MAXXX)
        {
            err = io::xprintf(
                "Size of the volume buffer xdim_aligned=%lu is bigger than UINT32_MAXXX=%u",
                xdim_aligned, UINT32_MAXXX);
            LOGW << err;
        } else if(xdim_aligned * 4 > UINT32_MAXXX)
        {
            err = io::xprintf(
                "Byte size of the volume buffer xdim_aligned=%lu is bigger than UINT32_MAXXX=%u",
                4 * xdim_aligned, UINT32_MAXXX);
            LOGW << err;
        }
        if(bdim_aligned > UINT32_MAXXX)
        {
            err = io::xprintf(
                "Size of the projection buffer bdim_aligned=%lu is bigger than UINT32_MAXXX=%u",
                bdim_aligned, UINT32_MAXXX);
            LOGW << err;
        } else if(bdim_aligned * 4 > UINT32_MAXXX)
        {
            err = io::xprintf("Byte size of the projection buffer bdim_aligned=%lu is bigger than "
                              "UINT32_MAXXX=%u",
                              4 * bdim_aligned, UINT32_MAXXX);
            LOGW << err;
        }

        CLINCLUDEutils();
    }

protected:
    const uint32_t pdimx, pdimy, pdimz, vdimx, vdimy, vdimz;
    const uint32_t workGroupSize = 256;
    uint64_t XDIM, BDIM, XDIM_ALIGNED, BDIM_ALIGNED, XDIM_REDUCED1, BDIM_REDUCED1,
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
     * Copy float* array of size elements into the CL::buffer. The buffer must have appropriate
     * size.
     *
     * @param c_array Block of C memory
     * @param cl_buffer Block of OpenCL memory
     * @param size number of elements in c_array
     *
     * @return 0 on success
     */
    int arrayIntoBuffer(float* c_array, cl::Buffer cl_buffer, uint64_t size);
    /**
     * Copy CL:buffer into the float* array of size elements. The buffer must have appropriate size.
     *
     * @param cl_buffer Block of OpenCL memory
     * @param c_array Block of C memory
     * @param size number of elements in c_array
     *
     * @return 0 on success
     */
    int bufferIntoArray(cl::Buffer cl_buffer, float* c_array, uint64_t size);

    std::shared_ptr<cl::Buffer> tmp_b_red1 = nullptr, tmp_b_red2 = nullptr;
    std::shared_ptr<cl::Buffer> tmp_x_red1 = nullptr, tmp_x_red2 = nullptr;
};

} // namespace KCT
