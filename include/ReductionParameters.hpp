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

namespace KCT {

class ReductionParameters
{
public:
    const uint32_t UINT32_MAXXX = ((uint32_t)-1);
    const uint32_t pdimx, pdimy, pdimz, vdimx, vdimy, vdimz;
    const uint32_t workGroupSize;
    uint32_t pFrameSize, vFrameSize;
    uint64_t XDIM, BDIM, XDIM_ALIGNED, BDIM_ALIGNED, XDIM_REDUCED1, BDIM_REDUCED1,
        XDIM_REDUCED1_ALIGNED, BDIM_REDUCED1_ALIGNED, XDIM_REDUCED2, BDIM_REDUCED2,
        XDIM_REDUCED2_ALIGNED, BDIM_REDUCED2_ALIGNED;
    uint64_t DIM_REDUCED1_MIN, DIM_REDUCED2_MIN;
    uint64_t BYTESIZE_REDUCED1_MIN, BYTESIZE_REDUCED2_MIN;

    ReductionParameters(uint32_t pdimx,
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
        XDIM = uint64_t(vdimx) * uint64_t(vdimy) * uint64_t(vdimz);
        BDIM = uint64_t(pdimx) * uint64_t(pdimy) * uint64_t(pdimz);
        vFrameSize = computeFrameSize(vdimx, vdimy);
        pFrameSize = computeFrameSize(pdimx, pdimy);
        XDIM_ALIGNED = XDIM + (workGroupSize - XDIM % workGroupSize) % workGroupSize;
        BDIM_ALIGNED = BDIM + (workGroupSize - BDIM % workGroupSize) % workGroupSize;
        XDIM_REDUCED1 = XDIM_ALIGNED / workGroupSize; // It is divisible by design
        BDIM_REDUCED1 = BDIM_ALIGNED / workGroupSize;
        XDIM_REDUCED1_ALIGNED
            = XDIM_REDUCED1 + (workGroupSize - XDIM_REDUCED1 % workGroupSize) % workGroupSize;
        BDIM_REDUCED1_ALIGNED
            = BDIM_REDUCED1 + (workGroupSize - BDIM_REDUCED1 % workGroupSize) % workGroupSize;
        XDIM_REDUCED2 = XDIM_REDUCED1_ALIGNED / workGroupSize;
        BDIM_REDUCED2 = BDIM_REDUCED1_ALIGNED / workGroupSize;
        XDIM_REDUCED2_ALIGNED
            = XDIM_REDUCED2 + (workGroupSize - XDIM_REDUCED2 % workGroupSize) % workGroupSize;
        BDIM_REDUCED2_ALIGNED
            = BDIM_REDUCED2 + (workGroupSize - BDIM_REDUCED2 % workGroupSize) % workGroupSize;
        if(XDIM > BDIM)
        {
            DIM_REDUCED1_MIN = XDIM_REDUCED1;
            DIM_REDUCED2_MIN = XDIM_REDUCED2;
        } else
        {
            DIM_REDUCED1_MIN = BDIM_REDUCED1;
            DIM_REDUCED2_MIN = BDIM_REDUCED2;
        }
        // For non-barier reduction algorithms to work
        if(DIM_REDUCED1_MIN < pdimz)
        {
            DIM_REDUCED1_MIN = pdimz;
        }
        if(DIM_REDUCED1_MIN < vdimz)
        {
            DIM_REDUCED1_MIN = vdimz;
        }
        BYTESIZE_REDUCED1_MIN = sizeof(double) * DIM_REDUCED1_MIN;
        BYTESIZE_REDUCED2_MIN = sizeof(double) * DIM_REDUCED2_MIN;
    }

private:
    uint32_t computeFrameSize(uint32_t dx, uint32_t dy)
    {
        uint64_t fs = (uint64_t)dx * (uint64_t)dy;
        std::string err;
        if(fs > UINT32_MAXXX)
        {
            err = io::xprintf(
                "Algorithms for reduction are based on the assumption that frameSize=%lu fits into "
                "UINT32_MAXXX=%lu, but it is not the case here.",
                fs, UINT32_MAXXX);
            KCTERR(err);
        }
        return (uint32_t)fs;
    }
};

} // namespace KCT
