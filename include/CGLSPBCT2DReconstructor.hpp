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

class CGLSPBCT2DReconstructor : virtual public BasePBCT2DReconstructor
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
    CGLSPBCT2DReconstructor(uint32_t pdimx,
                            uint32_t pdimy,
                            uint32_t pdimz,
                            uint32_t vdimx,
                            uint32_t vdimy,
                            uint32_t vdimz,
                            uint32_t workGroupSize = 256)
        : BasePBCT2DOperator(pdimx, pdimy, pdimz, vdimx, vdimy, vdimz, workGroupSize)
        , BasePBCT2DReconstructor(pdimx, pdimy, pdimz, vdimx, vdimy, vdimz, workGroupSize)

    {
        removeTikhonovRegularization();
        useGradient3D(true);
        useLaplace3D(true);
    }

    virtual int reconstruct(uint32_t maxIterations = 100, float errCondition = 0.01);

    int reconstruct_experimental(uint32_t maxIterations = 100, float errCondition = 0.01);

    /**
     * Tikhonov regularized tomographic reconstruction using CGLS. Sometimes the problem is also
     * referred as ridge regression, see https://en.wikipedia.org/wiki/Ridge_regression
     *
     * This procedure can be used also for preconditioning. In this case we compute the problem with
     * two diagonal matrices C and W.
     *
     *
     * @param maxIterations
     * @param errCondition
     * @param invCpr_xbuf Right preconditioner.
     * @param leftPreconditioner_bbuf Left preconditioner W. We are solving min_x || W (b - Ax)||^2,
     * wehre W is the diagonal matrix constructed out of left preconditioner. In weighted least
     * squares, this matrix is often named W^{1/2} and can represent the reciprocal of the standard
     * deviation of the measurements. In the out of center scan, it can be also used as square root
     * of the ray weight, reciprocal of the number of measurements for given ray. In SIRT, the
     * weighting scheme is chosen such that W is the square root of the row sums of the system
     * matrix, this scheme is adapted for CGLS in function reconstructSumPreconditioning.
     *
     * @return 0 if success, 1 if error
     */
    int reconstructTikhonov(uint32_t maxIterations = 100,
                            float errCondition = 0.01,
                            std::shared_ptr<cl::Buffer> invCpr_xbuf = nullptr,
                            std::shared_ptr<cl::Buffer> invSqrtRpr_bbuf = nullptr);

    int reconstructDiagonalPreconditioner(std::shared_ptr<cl::Buffer> invertedpreconditioner_xbuf,
                                          uint32_t maxIterations = 100,
                                          float errCondition = 0.01);

    int reconstructDiagonalPreconditioner(float* invertedpreconditioner,
                                          uint32_t maxIterations = 100,
                                          float errCondition = 0.01);

    int reconstructJacobi(uint32_t maxIterations = 100, float errCondition = 0.01);

    int reconstructSumPreconditioning(uint32_t maxIterations = 100, float errCondition = 0.01);

    int reconstructWLS(uint32_t maxIterations = 100,
                       float errCondition = 0.01,
                       float* weighs_BDIM = nullptr);

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
