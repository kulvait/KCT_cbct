#pragma once

// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <CL/cl.hpp>
#include <chrono>
#include <ctime>
#include <iostream>

// Internal libraries
#include "BasePBCT2DReconstructor.hpp"
#include "DEN/DenProjectionMatrixReader.hpp"
#include "GEOMETRY/Geometry3DParallel.hpp"
#include "GEOMETRY/Geometry3DParallelI.hpp"
#include "OPENCL/OpenCLManager.hpp"
#include "rawop.h"
#include "stringFormatter.h"

namespace KCT {

class PDHGPBCT2DReconstructor : virtual public BasePBCT2DReconstructor
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
    PDHGPBCT2DReconstructor(uint32_t pdimx,
                            uint32_t pdimy,
                            uint32_t pdimz,
                            uint32_t vdimx,
                            uint32_t vdimy,
                            uint32_t vdimz,
                            uint32_t workGroupSize = 256)
        : BasePBCT2DOperator(pdimx, pdimy, pdimz, vdimx, vdimy, vdimz, workGroupSize)
        , BasePBCT2DReconstructor(pdimx, pdimy, pdimz, vdimx, vdimy, vdimz, workGroupSize)

    {
    }

    virtual int reconstruct(uint32_t maxIterations = 100, float errCondition = 0.01);

    int reconstruct(float lambda,
                    float tau,
                    float sigma,
                    float theta,
                    uint32_t maxPDHGIterations = 100,
                    float errConditionPDHG = 0.01,
                    uint32_t maxCGLSIterations = 100,
                    float errConditionCGLS = 0.01);

private:
    // CGLS solver
    int proximalOperatorCGLS(cl::Buffer x0proxIN_xbuf,
                             cl::Buffer xbOUT_xbuf,
                             float tau,
                             uint32_t maxCGLSIterations = 100,
                             float errConditionCGLS = 0.01,
                             uint32_t outerIterationIndex = 0);
    // Add global buffers for CGLS
    std::shared_ptr<cl::Buffer> directionVector_xbuf, residualVector_xbuf,
        residualVector_xbuf_L2add, discrepancy_bbuf_xpart_L2, AdirectionVector_bbuf_xpart_L2;
    std::shared_ptr<cl::Buffer> discrepancy_bbuf, AdirectionVector_bbuf;
    double NB0;
};

} // namespace KCT
