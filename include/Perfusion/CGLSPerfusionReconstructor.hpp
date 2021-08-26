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
#include "Perfusion/BasePerfusionReconstructor.hpp"
#include "rawop.h"
#include "stringFormatter.h"

namespace KCT {

class CGLSPerfusionReconstructor : public BasePerfusionReconstructor
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
    CGLSPerfusionReconstructor(uint32_t pdimx,
                               uint32_t pdimy,
                               uint32_t pdimz,
                               uint32_t vdimx,
                               uint32_t vdimy,
                               uint32_t vdimz,
                               uint32_t workGroupSize = 256)
        : BasePerfusionReconstructor(pdimx, pdimy, pdimz, vdimx, vdimy, vdimz, workGroupSize)
    {
    }

    int reconstruct(uint32_t maxIterations = 100, float errCondition = 0.01, bool blocking = false);
};

} // namespace KCT
