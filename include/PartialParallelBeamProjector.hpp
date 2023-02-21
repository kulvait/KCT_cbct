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
#include "PartialPBCTOperator.hpp"
#include "rawop.h"
#include "stringFormatter.h"

namespace KCT {

class PartialParallelBeamProjector
{
private:
    uint32_t pdimx, pdimy, pFrameSize, pdimz, pdimz_partial, vdimz_partial_last, pzblocks;
    uint32_t vdimx, vdimy, vFrameSize, vdimz, vdimz_partial, pdimz_partial_last, vzblocks;
    uint64_t XDIM, BDIM, XBYTES, BBYTES, XDIM_partial, BDIM_partial, XDIM_partial_last,
        BDIM_partial_last;
    uint64_t maximumByteSize;

public:
    /**
     * Class that encapsulates projector and backprojector implementation for partial projections.
     *
     * @param pdimx Number of pixels
     * @param pdimy Number of pixels
     * @param vdimx Number of voxels
     * @param vdimy Number of voxels
     * @param vdimz Number of voxels
     */
    PartialParallelBeamProjector(uint32_t pdimx,
                                 uint32_t pdimy,
                                 uint32_t pdimz,
                                 uint32_t vdimx,
                                 uint32_t vdimy,
                                 uint32_t vdimz,
                                 uint32_t workGroupSize = 256,
                                 uint64_t maximumByteSize = 2147483648, // 256*1024*1024*8
                                 cl::NDRange projectorLocalNDRange = cl::NullRange,
                                 cl::NDRange backprojectorLocalNDRange = cl::NullRange)
    {
        this->pdimx = pdimx;
        this->pdimy = pdimy;
        this->pdimz = pdimz;
        this->vdimx = vdimx;
        this->vdimy = vdimy;
        this->vdimz = vdimz;
        this->workGroupSize = workGroupSize;
        if(maximumByteSize == 0)
        {
            maximumByteSize = 2147483648;
        }
        this->maximumByteSize = maximumByteSize;
        this->projectorLocalNDRange = projectorLocalNDRange;
        this->backprojectorLocalNDRange = backprojectorLocalNDRange;
        uint64_t projectionFrameSize = pdimx * pdimy;
        uint64_t volumeFrameSize = vdimx * vdimy;
        uint64_t maximumElementSize = maximumByteSize / sizeof(float);
        std::string ERR;
        if(projectionFrameSize > maximumElementSize || volumeFrameSize > maximumElementSize)
        {
            ERR = io::xprintf("XY volume or projection frame is bigger than maximumElementSize=%d",
                              maximumElementSize);
            KCTERR(ERR);
        }
        pFrameSize = projectionFrameSize;
        vFrameSize = volumeFrameSize;
        BDIM = projectionFrameSize * pdimz;
        XDIM = volumeFrameSize * vdimz;
        BBYTES = BDIM * sizeof(float);
        XBYTES = XDIM * sizeof(float);
        if(XBYTES <= maximumSliceSize && BBYTES <= maximumSliceSize)
        {
            pdimz_partial = pdimz;
            vdimz_partial = vdimz;
            pzblocks = 1;
            vzblocks = 1;
            XDIM_partial = XDIM;
            BDIM_partial = BDIM;
        }
        if(XDIM > maximumElementSize)
        {
            vdimz_partial = vdimz;
            vzblocks = 1;
            while(vdimz_partial % 2 == 0 && vdimz_partial * volumeFrameSize > maximumElementSize)
            {
                vdimz_partial /= 2;
                vzblocks *= 2;
            }
            vdimz_partial_last = vdimz_partial;
            if(vdimz_partial * volumeFrameSize > maximumElementSize)
            {
                vzblocks = (XDIM + maximumElementSize - 1) / maximumElementSize;
                vdimz_partial = (vdimz + vzblocks - 1) / vzblocks;
                if(vdimz % vdimz_partial != 0)
                {
                    vdimz_partial_last = vdimz % vdimz_partial;
                    LOGI << io::xprintf(
                        "There will be even partitioning of the vdimz %d into %d vzblocks with "
                        "the size vdimz_partial %d ",
                        vdimz, vzblocks, vdimz_partial);
                } else
                {
                    vdimz_partial_last = vdimz_partial;
                    LOGI << io::xprintf(
                        "There will be uneven partitioning of the vdimz %d into %d vzblocks with "
                        "the size vdimz_partial %d and the last block with the size "
                        "vdimz_partial_last %d",
                        vdimz, vzblocks, vdimz_partial, vdimz_partial_last);
                }
            } else
            {
                LOGI << io::xprintf(
                    "There will be bisection based even partitioning of the vdimz %d into "
                    "%d vzblocks with "
                    "the size vdimz_partial %d ",
                    vdimz, vzblocks, vdimz_partial);
            }
            XDIM_partial = vdimz_partial * volumeFrameSize;
            XDIM_partial_last = vdimz_partial_last * volumeFrameSize;
        } else
        {
            vzblocks = 1;
            vdimz_partial = vdimz;
            vdimz_partial_last = vdimz;
            XDIM_partial = XDIM;
            XDIM_partial_last = XDIM;
            LOGI << io::xprintf("I don't have to cut XDIM to grant maximumByteSize=%lu "
                                "of used host arrays.",
                                maximumByteSize);
        }
        if(BDIM > maximumElementSize)
        {
            pdimz_partial = pdimz;
            pzblocks = 1;
            while(pdimz_partial % 2 == 0
                  && pdimz_partial * projectionFrameSize > maximumElementSize)
            {
                pdimz_partial /= 2;
                pzblocks *= 2;
            }
            pdimz_partial_last = pdimz_partial;
            if(pdimz_partial * volumeFrameSize > maximumElementSize)
            {
                pzblocks = (BDIM + maximumElementSize - 1) / maximumElementSize;
                pdimz_partial = (pdimz + pzblocks - 1) / pzblocks;
                if(pdimz % pdimz_partial != 0)
                {
                    pdimz_partial_last = pdimz % pdimz_partial;
                    LOGI << io::xprintf(
                        "There will be  uneven partitioning of the pdimz %d into %d "
                        "pzblocks with the size pdimz_partial %d and tha last block with "
                        "size pdimz_partial_last %d",
                        pdimz, pzblocks, pdimz_partial, pdimz_partial_last);
                } else
                {
                    pdimz_partial_last = pdimz_partial;
                    LOGI << io::xprintf(
                        "There will be even partitioning of the pdimz %d into %d pzblocks "
                        "with the size pdimz_partial %d",
                        pdimz, pzblocks, pdimz_partial);
                }
            } else
            {
                LOGI << io::xprintf(
                    "There will be bisection based even partitioning of the pdimz %d into "
                    "%d pzblocks with the size pdimz_partial %d",
                    pdimz, pzblocks, pdimz_partial);
            }
            BDIM_partial = pdimz_partial * projectionFrameSize;
            BDIM_partial_last = pdimz_partial_last * projectionFrameSize;
        } else
        {
            pzblocks = 1;
            pdimz_partial = pdimz;
            pdimz_partial_last = pdimz;
            BDIM_partial = BDIM;
            BDIM_partial_last = BDIM;
            LOGI << io::xprintf("I don't have to cut BDIM to grant maximumByteSize=%lu "
                                "of used host arrays.",
                                maximumByteSize);
        }
        CT = std::make_shared<PartialPBCTOperator>(
            pdimx, pdimy, pdimz_partial, vdimx, vdimy, vdimz_partial, workGroupSize,
            projectorLocalNDRange, backprojectorLocalNDRange);
    }

    std::shared_ptr<PartialPBCTOperator> getCTOperator() { return CT; }
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

    int fillBufferByConstant(cl::Buffer cl_buffer, float constant, uint64_t bytecount);
    int project_partial(float* volume, float* projection);
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

    std::shared_ptr<cl::Buffer> volumeBuffer = nullptr;
    std::shared_ptr<cl::Buffer> projectionBuffer = nullptr;
    std::shared_ptr<cl::Buffer> tmpBuffer = nullptr;
    size_t tmpBuffer_size = 0;
    std::chrono::time_point<std::chrono::steady_clock> timestamp;

    std::shared_ptr<PartialPBCTOperator> CT;
    uint32_t volumePartitions;
    uint32_t workGroupSize;
    uint64_t maximumSliceSize; // 256*1024*1024*8
    cl::NDRange backprojectorLocalNDRange;
};

} // namespace KCT
