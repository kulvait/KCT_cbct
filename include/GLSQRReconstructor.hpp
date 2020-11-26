#pragma once

// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <CL/cl.hpp>
#include <chrono>
#include <ctime>
#include <iostream>

// Internal libraries
#include "BaseReconstructor.hpp"
#include "DEN/DenProjectionMatrixReader.hpp"
#include "MATRIX/ProjectionMatrix.hpp"
#include "OPENCL/OpenCLManager.hpp"
#include "rawop.h"
#include "stringFormatter.h"

namespace CTL {

class GLSQRReconstructor : public BaseReconstructor
{
public:
    GLSQRReconstructor(uint32_t pdimx,
                       uint32_t pdimy,
                       uint32_t pdimz,
                       uint32_t vdimx,
                       uint32_t vdimy,
                       uint32_t vdimz,
                       uint32_t workGroupSize = 256)
        : BaseReconstructor(pdimx, pdimy, pdimz, vdimx, vdimy, vdimz, workGroupSize)
    {
    }

    virtual int reconstruct(uint32_t maxIterations = 100, float errCondition = 0.01);
    int reconstructTikhonov(double lambda, uint32_t maxIterations = 100, float errCondition = 0.01);
};

} // namespace CTL
