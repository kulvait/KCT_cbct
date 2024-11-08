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
    /**
     * CGLS proximal operator solving problem
     * min_x 1 / (2 * tau) ||x-x_prox||_2^2 + ||Ax - b||_2^2 by means of CGLS based approximation.
     * The pointers might point to the same buffer or three different buffers. It is also admissible
     * that xbufIN_x_prox and xbufIN_x0 are not equal and xbufOUT is one of them. The procedure will
     * handle these situations.
     *
     * @param xbufIN_x_prox Proximal vector
     * @param xbufIN_x0 Initial vector
     * @param xbufOUT Output vector
     * @param tau Regularization parameter for proximal operator
     * @param maxCGLSIterations Maximum number of CGLS iterations
     * @param errConditionCGLS Error condition for CGLS
     * @param outerIterationIndex Index of the outer iteration for logging purposes
     *
     * @return
     */
    int proximalOperatorCGLS(std::shared_ptr<cl::Buffer> xbufIN_x_prox,
                             std::shared_ptr<cl::Buffer> xbufIN_x0,
                             std::shared_ptr<cl::Buffer> xbufOUT,
                             float tau,
                             uint32_t maxCGLSIterations = 100,
                             float errConditionCGLS = 0.01,
                             uint32_t outerIterationIndex = 0);
    // Add global buffers for CGLS
    std::shared_ptr<cl::Buffer> directionVector_xbuf, residualVector_xbuf,
        residualVector_xbuf_L2add, discrepancy_bbuf_xpart_L2, AdirectionVector_bbuf_xpart_L2;
    std::shared_ptr<cl::Buffer> discrepancy_bbuf, AdirectionVector_bbuf;
    std::array<double, 3> computeSolutionNorms(std::shared_ptr<cl::Buffer> x_vector,
                                               std::shared_ptr<cl::Buffer> x_vector_dx,
                                               std::shared_ptr<cl::Buffer> x_vector_dy,
                                               std::shared_ptr<cl::Buffer> x_0 = nullptr,
                                               bool computeDiscrepancy = true);
    double NB0, NATB0;
    bool proximalOperatorVerbose = true;
};

} // namespace KCT
