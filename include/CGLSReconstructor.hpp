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

namespace KCT {

class CGLSReconstructor : public BaseReconstructor
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
    CGLSReconstructor(uint32_t pdimx,
                      uint32_t pdimy,
                      uint32_t pdimz,
                      uint32_t vdimx,
                      uint32_t vdimy,
                      uint32_t vdimz,
                      uint32_t workGroupSize = 256,
                      cl::NDRange projectorLocalNDRange = cl::NDRange(),
                      cl::NDRange backprojectorLocalNDRange = cl::NDRange())
        : BaseReconstructor(pdimx,
                            pdimy,
                            pdimz,
                            vdimx,
                            vdimy,
                            vdimz,
                            workGroupSize,
                            projectorLocalNDRange,
                            backprojectorLocalNDRange)
    {
        removeTikhonovRegularization();
        useBoundaryReflection(false);
    }

    virtual int reconstruct(uint32_t maxIterations = 100, float errCondition = 0.01);

    int reconstruct_experimental(uint32_t maxIterations = 100, float errCondition = 0.01);

    int reconstructTikhonov(uint32_t maxIterations = 100, float errCondition = 0.01);

    int reconstructDiagonalPreconditioner(std::shared_ptr<cl::Buffer> invertedpreconditioner_xbuf,
                                          uint32_t maxIterations = 100,
                                          float errCondition = 0.01);

    int reconstructDiagonalPreconditioner(float* invertedpreconditioner,
                                          uint32_t maxIterations = 100,
                                          float errCondition = 0.01);

    int reconstructJacobi(uint32_t maxIterations = 100, float errCondition = 0.01);

    void precomputeJacobiPreconditioner(std::shared_ptr<cl::Buffer> X);

    void addTikhonovRegularization(float L2, float V2, float Laplace);

    void useBoundaryReflection(bool boundaryReflection);

    void removeTikhonovRegularization();

private:
    void tikhonovMatrixActionToAdirectionAndScale(cl::Buffer XIN);
    void tikhonovMatrixActionToDiscrepancyAndScale(cl::Buffer XIN);
    void tikhonovMatrixActionOnDiscrepancyToUpdateResidualVector(cl::Buffer residualVector);
    void tikhonov_discrepancy_equals_discrepancy_minus_alphaAdirection(double alpha);
    void tikhonovZeroDiscrepancyBuffers();
    void tikhonovSetRegularizingBuffersNull();
    double tikhonovSumOfAdirectionNorms2();

    std::shared_ptr<cl::Buffer> residualVector_xbuf_L2add, residualVector_xbuf_V2xadd,
        residualVector_xbuf_V2yadd, residualVector_xbuf_V2zadd,
        residualVector_xbuf_Laplaceadd; // X buffers
    std::shared_ptr<cl::Buffer> discrepancy_bbuf_xpart_L2, discrepancy_bbuf_xpart_V2x,
        discrepancy_bbuf_xpart_V2y, discrepancy_bbuf_xpart_V2z,
        discrepancy_bbuf_xpart_Laplace; // X buffers
    std::shared_ptr<cl::Buffer> AdirectionVector_bbuf_xpart_L2, AdirectionVector_bbuf_xpart_V2x,
        AdirectionVector_bbuf_xpart_V2y, AdirectionVector_bbuf_xpart_V2z,
        AdirectionVector_bbuf_xpart_Laplace; // X buffers
    bool tikhonovRegularization;
    bool tikhonovRegularizationL2;
    bool tikhonovRegularizationV2;
    bool tikhonovRegularizationLaplace;
    bool boundaryReflection;
    float effectSizeL2, effectSizeV2, effectSizeLaplace;
};

} // namespace KCT
