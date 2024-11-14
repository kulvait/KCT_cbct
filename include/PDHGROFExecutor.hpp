#pragma once

// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <CL/cl.hpp>
#include <chrono>
#include <ctime>
#include <iostream>

// Internal libraries
#include "BaseROFOperator.hpp"
#include "rawop.h"
#include "stringFormatter.h"

namespace KCT {

class PDHGROFExecutor : virtual public BaseROFOperator
{
public:
    /**
     * Initialize object
     *
     * @param vdimx Volume x dimension
     * @param vdimy Volume y dimension
     * @param vdimz Volume z dimension
     * @param xpath Path of cl kernel files
     * @param debug Should debugging be used by suppliing source and -g as options
     */
    /**
     * @brief Construct a new PDHGROFExecutor object
     *
     * @param pdimx
     * @param pdimy
     * @param pdimz
     * @param vdimx
     * @param vdimy
     * @param vdimz
     * @param workGroupSize
     */
    PDHGROFExecutor(uint32_t vdimx, uint32_t vdimy, uint32_t vdimz, uint32_t workGroupSize = 256)
        : BaseROFOperator(vdimx, vdimy, vdimz, workGroupSize)

    {
    }

    virtual int reconstruct(float mu,
                            float tau,
                            float sigma,
                            float theta,
                            uint32_t maxPDHGIterations = 100,
                            float errConditionPDHG = 0.01) override;

private:
    std::shared_ptr<cl::Buffer> directionVector_xbuf, residualVector_xbuf,
        residualVector_xbuf_L2add, discrepancy_bbuf_xpart_L2, AdirectionVector_bbuf_xpart_L2;
    std::array<double, 2> computeSolutionNorms(std::shared_ptr<cl::Buffer> x_vector,
                                               std::shared_ptr<cl::Buffer> x_vector_dx,
                                               std::shared_ptr<cl::Buffer> x_vector_dy,
                                               std::shared_ptr<cl::Buffer> x_0);
    bool proximalOperatorVerbose = true;
};

} // namespace KCT
