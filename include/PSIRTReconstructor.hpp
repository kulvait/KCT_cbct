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
#include "MATRIX/LightProjectionMatrix.hpp"
#include "MATRIX/ProjectionMatrix.hpp"
#include "OPENCL/OpenCLManager.hpp"
#include "rawop.h"
#include "stringFormatter.h"

namespace CTL {

class PSIRTReconstructor : public BaseReconstructor
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
    PSIRTReconstructor(uint32_t pdimx,
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
    int reconstruct_sirt(uint32_t maxIterations = 100, float errCondition = 0.01);

    void setup(double alpha = 1.99);

private:
    double alpha;
};

} // namespace CTL
