#pragma once

// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <CL/cl.hpp>
#include <chrono>
#include <ctime>
#include <iostream>

// Internal libraries
#include "BasePBCTReconstructor.hpp"
#include "DEN/DenProjectionMatrixReader.hpp"
#include "OPENCL/OpenCLManager.hpp"
#include "GEOMETRY/Geometry3DParallel.hpp"
#include "GEOMETRY/Geometry3DParallelI.hpp"
#include "rawop.h"
#include "stringFormatter.h"

namespace KCT {

class CGLSPBCTReconstructor : virtual public BasePBCTReconstructor
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
    CGLSPBCTReconstructor(uint32_t pdimx,
                          uint32_t pdimy,
                          uint32_t pdimz,
                          uint32_t vdimx,
                          uint32_t vdimy,
                          uint32_t vdimz,
                          uint32_t workGroupSize = 256)
        : BasePBCTOperator(pdimx,
                           pdimy,
                           pdimz,
                           vdimx,
                           vdimy,
                           vdimz,
                           workGroupSize)
        , BasePBCTReconstructor(pdimx,
                                pdimy,
                                pdimz,
                                vdimx,
                                vdimy,
                                vdimz,
                                workGroupSize)

    {
        removeTikhonovRegularization();
        useGradient3D(true);
        useLaplace3D(true);
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

    void useGradient3D(bool gradient3D);
    void useLaplace3D(bool laplace3D);

    void removeTikhonovRegularization();

private:
    void tikhonovMatrixActionToAdirectionAndScale(cl::Buffer XIN);
    void tikhonovMatrixActionToDiscrepancyAndScale(cl::Buffer XIN);
    void tikhonovMatrixActionOnDiscrepancyToUpdateResidualVector(cl::Buffer residualVector);
    void tikhonov_discrepancy_equals_discrepancy_minus_alphaAdirection(double alpha);
    void tikhonovZeroDiscrepancyBuffers();
    void tikhonovSetRegularizingBuffersNull();
    int weightedLeastSquares(float* weights);
    int preconditionnedLeastSquares(float* preconditionerXDIM);
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
    std::shared_ptr<cl::Buffer> weighting_bbuf = nullptr;
    std::shared_ptr<cl::Buffer> preconditioning_xbuf = nullptr;
    bool tikhonovRegularization;
    bool tikhonovRegularizationL2;
    bool tikhonovRegularizationV2;
    bool tikhonovRegularizationLaplace;
    bool laplace3D;
    bool gradient3D;
    float effectSizeL2, effectSizeV2, effectSizeLaplace;
};

} // namespace KCT
