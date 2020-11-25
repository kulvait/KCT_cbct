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

namespace CTL {

class PerfusionOperator : public BasePerfusionReconstructor
{
public:
    /**
     * Initialize GLSQRPerfusionReconstructor
     *
     * @param volume Pointer to volume file
     * @param vdimx Volume x dimension
     * @param vdimy Volume y dimension
     * @param vdimz Volume z dimension
     * @param xpath Path of cl kernel files
     * @param debug Should debugging be used by suppliing source and -g as options
     */
    PerfusionOperator(uint32_t pdimx,
                                uint32_t pdimy,
                                uint32_t pdimz,
                                uint32_t vdimx,
                                uint32_t vdimy,
                                uint32_t vdimz,
                                uint32_t workGroupSize = 256)
        : BasePerfusionReconstructor(pdimx, pdimy, pdimz, vdimx, vdimy, vdimz, workGroupSize)
    {
    }

    int reconstruct(uint32_t maxIterations = 100, float errCondition = 0.01, bool blocking = false)
    {
        LOGD << "This function is not implemented in PerfusionOperator class.";
        return 1;
    }
    int project(bool blocking = false);
    int backproject(bool blocking = false);
};

} // namespace CTL
