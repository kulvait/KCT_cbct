#include "CGLSPBCTReconstructor.hpp"

namespace KCT {

int CGLSPBCTReconstructor::reconstruct(uint32_t maxIterations, float errCondition)
{
    if(tikhonovRegularization)
    {
        return reconstructTikhonov(maxIterations, errCondition);
    }
    LOGD << printTime("WELCOME TO CGLS, init", false, true);
    uint32_t iteration = 1;

    // Initialization
    double norm, residualNorm2_old, residualNorm2_now, AdirectionNorm2, alpha, beta;
    double NB0 = std::sqrt(normBBuffer_barrier_double(*b_buf));
    double NR0, NX;
    LOGI << io::xprintf("||b||=%f", NB0);
    std::shared_ptr<cl::Buffer> directionVector_xbuf, residualVector_xbuf; // X buffers
    allocateXBuffers(2);
    directionVector_xbuf = getXBuffer(0);
    residualVector_xbuf = getXBuffer(1);
    allocateBBuffers(2);
    std::shared_ptr<cl::Buffer> discrepancy_bbuf, AdirectionVector_bbuf; // B buffers
    discrepancy_bbuf = getBBuffer(0);
    AdirectionVector_bbuf = getBBuffer(1);
    // discrepancy_bbuf stores initial discrepancy
    algFLOATvector_copy(*b_buf, *discrepancy_bbuf, BDIM);
    if(useVolumeAsInitialX0)
    {
        project(*x_buf, *AdirectionVector_bbuf);
        algFLOATvector_A_equals_A_plus_cB(*discrepancy_bbuf, *AdirectionVector_bbuf, -1.0f, BDIM);
        reportTime("Projection x0", false, true);
    } else
    {
        Q[0]->enqueueFillBuffer<cl_float>(*x_buf, FLOATZERO, 0, XDIM * sizeof(float));
    }
    backproject(*discrepancy_bbuf, *residualVector_xbuf);
    // writeVolume(*residualVector_xbuf, "/tmp/initialBackprojection");
    algFLOATvector_copy(*residualVector_xbuf, *directionVector_xbuf, XDIM);
    residualNorm2_old = normXBuffer_barrier_double(*residualVector_xbuf);
    reportTime("Backprojection 0", false, true);
    NR0 = std::sqrt(residualNorm2_old);
    project(*directionVector_xbuf, *AdirectionVector_bbuf);
    // writeProjections(*AdirectionVector_bbuf, "/tmp/initialProjection");
    AdirectionNorm2 = normBBuffer_barrier_double(*AdirectionVector_bbuf);
    reportTime("Projection 1", false, true);
    alpha = residualNorm2_old / AdirectionNorm2;
    algFLOATvector_A_equals_A_plus_cB(*x_buf, *directionVector_xbuf, alpha, XDIM);
    algFLOATvector_A_equals_A_plus_cB(*discrepancy_bbuf, *AdirectionVector_bbuf, -alpha, BDIM);
    norm = std::sqrt(normBBuffer_barrier_double(*discrepancy_bbuf));
    while(norm / NB0 > errCondition && iteration < maxIterations)
    {
        if(reportKthIteration > 0 && iteration % reportKthIteration == 0)
        {
            LOGD << io::xprintf("Writing file %sx_it%02d.den", intermediatePrefix.c_str(),
                                iteration);
            BasePBCTReconstructor::writeVolume(
                *x_buf, io::xprintf("%sx_it%02d.den", intermediatePrefix.c_str(), iteration));
        }
        // DEBUG
        if(iteration % 1000 == 0)
        {
            project(*x_buf, *discrepancy_bbuf);
            algFLOATvector_A_equals_Ac_plus_B(*discrepancy_bbuf, *b_buf, -1.0, BDIM);
            double norm_ = std::sqrt(normBBuffer_barrier_double(*discrepancy_bbuf));
            reportTime(io::xprintf("Reothrogonalization projection %d", iteration), false, true);
            LOGI << io::xprintf_green("\nReorthogonalization in iteration %d: |Ax-b|=%0.1f "
                                      "representing %0.2f%% of |b|, loss of orthogonality %f%%.",
                                      iteration, norm_, 100.0 * norm_ / NB0,
                                      100 * std::abs(norm - norm_) / norm);
        }
        // DEBUG
        backproject(*discrepancy_bbuf, *residualVector_xbuf);
        residualNorm2_now = normXBuffer_barrier_double(*residualVector_xbuf);
        reportTime(io::xprintf("Backprojection %d", iteration), false, true);
        // Delayed update of residual vector
        beta = residualNorm2_now / residualNorm2_old;
        NX = std::sqrt(residualNorm2_now);
        LOGI << io::xprintf_green("\nIteration %d: |Ax-b|=%0.1f representing %0.2f%% of |b|, "
                                  "|AT(Ax-b)|=%0.2f representing %0.3f%% of "
                                  "|AT(Ax0-b)|.",
                                  iteration, norm, 100.0 * norm / NB0, NX, 100 * NX / NR0);
        algFLOATvector_A_equals_Ac_plus_B(*directionVector_xbuf, *residualVector_xbuf, beta, XDIM);
        // Delayed update of direction vector
        iteration = iteration + 1;
        residualNorm2_old = residualNorm2_now;
        project(*directionVector_xbuf, *AdirectionVector_bbuf);
        AdirectionNorm2 = normBBuffer_barrier_double(*AdirectionVector_bbuf);
        reportTime(io::xprintf("Projection %d", iteration), false, true);
        alpha = residualNorm2_old / AdirectionNorm2;
        algFLOATvector_A_equals_A_plus_cB(*x_buf, *directionVector_xbuf, alpha, XDIM);
        algFLOATvector_A_equals_A_plus_cB(*discrepancy_bbuf, *AdirectionVector_bbuf, -alpha, BDIM);
        norm = std::sqrt(normBBuffer_barrier_double(*discrepancy_bbuf));
    }
    LOGI << io::xprintf_green("\nIteration %d: |Ax-b|=%0.1f representing %0.2f%% of |b|.",
                              iteration, norm, 100.0 * norm / NB0);
    Q[0]->enqueueReadBuffer(*x_buf, CL_TRUE, 0, sizeof(float) * XDIM, x);
    return 0;
}

int CGLSPBCTReconstructor::weightedLeastSquares(float* weightsBDIM)
{
    cl_int err;
    weighting_bbuf
        = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                       sizeof(float) * BDIM, (void*)weightsBDIM, &err);
    if(err != CL_SUCCESS)
    {
        return 0;
    } else
    {
        return 1;
    }
}

int CGLSPBCTReconstructor::preconditionnedLeastSquares(float* preconditionerXDIM)
{
    cl_int err;
    preconditioning_xbuf
        = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                       sizeof(float) * XDIM, (void*)preconditionerXDIM, &err);
    if(err != CL_SUCCESS)
    {
        return 0;
    } else
    {
        return 1;
    }
}

void CGLSPBCTReconstructor::addTikhonovRegularization(float L2, float V2, float Laplace)
{
    this->tikhonovRegularization = false;
    if(!std::isnan(L2) && L2 != 0.0f)
    {
        this->tikhonovRegularization = true;
        this->tikhonovRegularizationL2 = true;
        this->effectSizeL2 = L2;
    } else
    {
        this->tikhonovRegularizationL2 = false;
    }
    if(!std::isnan(V2) && V2 != 0.0f)
    {
        this->tikhonovRegularization = true;
        this->tikhonovRegularizationV2 = true;
        this->effectSizeV2 = V2;
    } else
    {
        this->tikhonovRegularizationV2 = false;
    }
    if(!std::isnan(Laplace) && Laplace != 0.0f)
    {
        this->tikhonovRegularization = true;
        this->tikhonovRegularizationLaplace = true;
        this->effectSizeLaplace = Laplace;
    } else
    {
        this->tikhonovRegularizationLaplace = false;
    }
}

void CGLSPBCTReconstructor::useGradient3D(bool gradient3D) { this->gradient3D = gradient3D; }
void CGLSPBCTReconstructor::useLaplace3D(bool laplace3D) { this->laplace3D = laplace3D; }

void CGLSPBCTReconstructor::removeTikhonovRegularization() { this->tikhonovRegularization = false; }

void CGLSPBCTReconstructor::tikhonovMatrixActionToAdirectionAndScale(cl::Buffer XIN)
{
    cl::NDRange globalRange(vdimx, vdimy, vdimz);
    cl::NDRange localRange = projectorLocalNDRange;
    cl_float3 voxelSizesF = { (float)voxelSizes.x, (float)voxelSizes.y, (float)voxelSizes.z };
    if(tikhonovRegularizationL2)
    {
        algFLOATvector_copy(XIN, *AdirectionVector_bbuf_xpart_L2,
                            XDIM); // discrepancy_bbuf stores initial discrepancy
        algFLOATvector_scale(*AdirectionVector_bbuf_xpart_L2, effectSizeL2, XDIM);
    }
    if(tikhonovRegularizationV2)
    {
        if(gradient3D)
        {
            algFLOATvector_3DisotropicGradient(
                XIN, *AdirectionVector_bbuf_xpart_V2x, *AdirectionVector_bbuf_xpart_V2y,
                *AdirectionVector_bbuf_xpart_V2z, vdims, voxelSizesF, globalRange, localRange);
        } else
        {
            algFLOATvector_2DisotropicGradient(XIN, *AdirectionVector_bbuf_xpart_V2x,
                                               *AdirectionVector_bbuf_xpart_V2y, vdims, voxelSizesF,
                                               globalRange, localRange);
        }
        algFLOATvector_scale(*AdirectionVector_bbuf_xpart_V2x, effectSizeV2, XDIM);
        algFLOATvector_scale(*AdirectionVector_bbuf_xpart_V2y, effectSizeV2, XDIM);
        algFLOATvector_scale(*AdirectionVector_bbuf_xpart_V2z, effectSizeV2, XDIM);
    }
    if(tikhonovRegularizationLaplace)
    {

        if(laplace3D)
        {
            algFLOATvector_3DconvolutionLaplaceZeroBoundary(
                XIN, *AdirectionVector_bbuf_xpart_Laplace, vdims, voxelSizesF, globalRange,
                localRange);
        } else
        {
            cl_float16 convolutionKernel = { 0.25f, 0.5f, 0.25f, 0.5f, -3.0f, 0.5f, 0.25f, 0.5f,
                                             0.25f, 0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f };
            algFLOATvector_2Dconvolution3x3ZeroBoundary(XIN, *AdirectionVector_bbuf_xpart_Laplace, vdims,
                                            convolutionKernel, globalRange, localRange);
        }
        algFLOATvector_scale(*AdirectionVector_bbuf_xpart_Laplace, effectSizeLaplace, XDIM);
    }
}

void CGLSPBCTReconstructor::tikhonovMatrixActionToDiscrepancyAndScale(cl::Buffer XIN)
{
    cl::NDRange globalRange(vdimx, vdimy, vdimz);
    cl::NDRange localRange = projectorLocalNDRange;
    cl_float3 voxelSizesF = { (float)voxelSizes.x, (float)voxelSizes.y, (float)voxelSizes.z };
    if(tikhonovRegularizationL2)
    {
        algFLOATvector_copy(XIN, *discrepancy_bbuf_xpart_L2,
                            XDIM); // discrepancy_bbuf stores initial discrepancy
        algFLOATvector_scale(*discrepancy_bbuf_xpart_L2, effectSizeL2, XDIM);
    }
    if(tikhonovRegularizationV2)
    {
        if(gradient3D)
        {
            algFLOATvector_3DisotropicGradient(
                XIN, *discrepancy_bbuf_xpart_V2x, *discrepancy_bbuf_xpart_V2y,
                *discrepancy_bbuf_xpart_V2z, vdims, voxelSizesF, globalRange, localRange);
        } else
        {
            algFLOATvector_2DisotropicGradient(

                XIN, *discrepancy_bbuf_xpart_V2x, *discrepancy_bbuf_xpart_V2y, vdims, voxelSizesF,
                globalRange, localRange);
        }
        algFLOATvector_scale(*discrepancy_bbuf_xpart_V2x, effectSizeV2, XDIM);
        algFLOATvector_scale(*discrepancy_bbuf_xpart_V2y, effectSizeV2, XDIM);
        algFLOATvector_scale(*discrepancy_bbuf_xpart_V2z, effectSizeV2, XDIM);
    }
    if(tikhonovRegularizationLaplace)
    {
        if(laplace3D)
        {
            algFLOATvector_3DconvolutionLaplaceZeroBoundary(
                XIN, *discrepancy_bbuf_xpart_Laplace, vdims, voxelSizesF, globalRange, localRange);
        } else
        {
            cl_float16 convolutionKernel = { 0.25f, 0.5f, 0.25f, 0.5f, -3.0f, 0.5f, 0.25f, 0.5f,
                                             0.25f, 0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f };
            algFLOATvector_2Dconvolution3x3ZeroBoundary(XIN, *discrepancy_bbuf_xpart_Laplace, vdims,
                                            convolutionKernel, globalRange, localRange);
        }
        algFLOATvector_scale(*discrepancy_bbuf_xpart_Laplace, effectSizeLaplace, XDIM);
    }
}

void CGLSPBCTReconstructor::tikhonovMatrixActionOnDiscrepancyToUpdateResidualVector(
    cl::Buffer residualVector)
{
    cl::NDRange globalRange(vdimx, vdimy, vdimz);
    cl::NDRange localRange = projectorLocalNDRange;
    cl_float3 voxelSizesF = { (float)voxelSizes.x, (float)voxelSizes.y, (float)voxelSizes.z };
    if(tikhonovRegularizationL2)
    {
        algFLOATvector_copy(*discrepancy_bbuf_xpart_L2, *residualVector_xbuf_L2add, XDIM);
        algFLOATvector_A_equals_A_plus_cB(residualVector, *residualVector_xbuf_L2add, effectSizeL2,
                                          XDIM);
    }
    if(tikhonovRegularizationV2)
    {
        // Here backprojection needs three calls
        // We need to multiply with -1
        // This shall be action of transposed regularizing matrix, it is with - sign but it will not
        // be exact with reflection conditions
        if(gradient3D)
        {
            algFLOATvector_isotropicBackDx(*discrepancy_bbuf_xpart_V2x, *residualVector_xbuf_V2xadd,
                                           vdims, voxelSizesF, globalRange, localRange);
            algFLOATvector_isotropicBackDy(*discrepancy_bbuf_xpart_V2y, *residualVector_xbuf_V2yadd,
                                           vdims, voxelSizesF, globalRange, localRange);
            algFLOATvector_isotropicBackDz(*discrepancy_bbuf_xpart_V2z, *residualVector_xbuf_V2zadd,
                                           vdims, voxelSizesF, globalRange, localRange);
            algFLOATvector_A_equals_A_plus_cB(residualVector, *residualVector_xbuf_V2xadd,
                                              effectSizeV2, XDIM);
            algFLOATvector_A_equals_A_plus_cB(residualVector, *residualVector_xbuf_V2yadd,
                                              effectSizeV2, XDIM);
            algFLOATvector_A_equals_A_plus_cB(residualVector, *residualVector_xbuf_V2zadd,
                                              effectSizeV2, XDIM);

        } else
        {
            algFLOATvector_isotropicBackDx(*discrepancy_bbuf_xpart_V2x, *residualVector_xbuf_V2xadd,
                                           vdims, voxelSizesF, globalRange, localRange);
            algFLOATvector_isotropicBackDy(*discrepancy_bbuf_xpart_V2y, *residualVector_xbuf_V2yadd,
                                           vdims, voxelSizesF, globalRange, localRange);
            algFLOATvector_A_equals_A_plus_cB(residualVector, *residualVector_xbuf_V2xadd,
                                              effectSizeV2, XDIM);
            algFLOATvector_A_equals_A_plus_cB(residualVector, *residualVector_xbuf_V2yadd,
                                              effectSizeV2, XDIM);
        }
    }
    if(tikhonovRegularizationLaplace)
    {
        if(laplace3D)
        {
            algFLOATvector_3DconvolutionLaplaceZeroBoundary(*discrepancy_bbuf_xpart_Laplace,
                                                            *residualVector_xbuf_Laplaceadd, vdims,
                                                            voxelSizesF, globalRange, localRange);
        } else
        {
            cl_float16 convolutionKernel = { 0.25f, 0.5f, 0.25f, 0.5f, -3.0f, 0.5f, 0.25f, 0.5f,
                                             0.25f, 0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f };
            algFLOATvector_2Dconvolution3x3ZeroBoundary(*discrepancy_bbuf_xpart_Laplace,
                                            *residualVector_xbuf_Laplaceadd, vdims,
                                            convolutionKernel, globalRange, localRange);
        }
        algFLOATvector_A_equals_A_plus_cB(residualVector, *residualVector_xbuf_Laplaceadd,
                                          effectSizeLaplace, XDIM);
    }
}

void CGLSPBCTReconstructor::tikhonov_discrepancy_equals_discrepancy_minus_alphaAdirection(
    double alpha)
{
    if(tikhonovRegularizationL2)
    {
        algFLOATvector_A_equals_A_plus_cB(*discrepancy_bbuf_xpart_L2,
                                          *AdirectionVector_bbuf_xpart_L2, -alpha, XDIM);
    }
    if(tikhonovRegularizationV2)
    {
        algFLOATvector_A_equals_A_plus_cB(*discrepancy_bbuf_xpart_V2x,
                                          *AdirectionVector_bbuf_xpart_V2x, -alpha, XDIM);
        algFLOATvector_A_equals_A_plus_cB(*discrepancy_bbuf_xpart_V2y,
                                          *AdirectionVector_bbuf_xpart_V2y, -alpha, XDIM);
        if(gradient3D)
        {
            algFLOATvector_A_equals_A_plus_cB(*discrepancy_bbuf_xpart_V2z,
                                              *AdirectionVector_bbuf_xpart_V2z, -alpha, XDIM);
        }
    }
    if(tikhonovRegularizationLaplace)
    {
        algFLOATvector_A_equals_A_plus_cB(*discrepancy_bbuf_xpart_Laplace,
                                          *AdirectionVector_bbuf_xpart_Laplace, -alpha, XDIM);
    }
}

double CGLSPBCTReconstructor::tikhonovSumOfAdirectionNorms2()
{
    double AdirectionNorms2Xpart = 0.0;
    if(tikhonovRegularizationL2)
    {
        AdirectionNorms2Xpart += normXBuffer_barrier_double(*AdirectionVector_bbuf_xpart_L2);
    }
    if(tikhonovRegularizationV2)
    {
        AdirectionNorms2Xpart += normXBuffer_barrier_double(*AdirectionVector_bbuf_xpart_V2x);
        AdirectionNorms2Xpart += normXBuffer_barrier_double(*AdirectionVector_bbuf_xpart_V2y);
        if(gradient3D)
        {
            AdirectionNorms2Xpart += normXBuffer_barrier_double(*AdirectionVector_bbuf_xpart_V2z);
        }
    }
    if(tikhonovRegularizationLaplace)
    {
        AdirectionNorms2Xpart += normXBuffer_barrier_double(*AdirectionVector_bbuf_xpart_Laplace);
    }
    return AdirectionNorms2Xpart;
}

void CGLSPBCTReconstructor::tikhonovZeroDiscrepancyBuffers()
{
    if(tikhonovRegularizationL2)
    {
        Q[0]->enqueueFillBuffer<cl_float>(*discrepancy_bbuf_xpart_L2, FLOATZERO, 0,
                                          XDIM * sizeof(float));
    }
    if(tikhonovRegularizationV2)
    {
        Q[0]->enqueueFillBuffer<cl_float>(*discrepancy_bbuf_xpart_V2x, FLOATZERO, 0,
                                          XDIM * sizeof(float));
        Q[0]->enqueueFillBuffer<cl_float>(*discrepancy_bbuf_xpart_V2y, FLOATZERO, 0,
                                          XDIM * sizeof(float));
        Q[0]->enqueueFillBuffer<cl_float>(*discrepancy_bbuf_xpart_V2z, FLOATZERO, 0,
                                          XDIM * sizeof(float));
    }
    if(tikhonovRegularizationLaplace)
    {
        Q[0]->enqueueFillBuffer<cl_float>(*discrepancy_bbuf_xpart_Laplace, FLOATZERO, 0,
                                          XDIM * sizeof(float));
    }
}

void CGLSPBCTReconstructor::tikhonovSetRegularizingBuffersNull()
{
    residualVector_xbuf_L2add = nullptr;
    residualVector_xbuf_V2xadd = nullptr;
    residualVector_xbuf_V2yadd = nullptr;
    residualVector_xbuf_V2zadd = nullptr;
    residualVector_xbuf_Laplaceadd = nullptr;
    discrepancy_bbuf_xpart_L2 = nullptr;
    discrepancy_bbuf_xpart_V2x = nullptr;
    discrepancy_bbuf_xpart_V2y = nullptr;
    discrepancy_bbuf_xpart_V2z = nullptr;
    discrepancy_bbuf_xpart_Laplace = nullptr;
    AdirectionVector_bbuf_xpart_L2 = nullptr;
    AdirectionVector_bbuf_xpart_V2x = nullptr;
    AdirectionVector_bbuf_xpart_V2y = nullptr;
    AdirectionVector_bbuf_xpart_V2z = nullptr;
    AdirectionVector_bbuf_xpart_Laplace = nullptr;
}

int CGLSPBCTReconstructor::reconstructTikhonov(uint32_t maxIterations, float errCondition)
{
    std::string INFO;
    INFO = "WELCOME TO CGLS WITH TIKHONOV, ";
    uint32_t additionalRegularizationVectors = 0; // Additional vector allocation
    if(tikhonovRegularizationL2)
    {
        INFO = io::xprintf("%sL2=%0.2f, ", effectSizeL2, INFO.c_str());
        additionalRegularizationVectors++;
    }
    if(tikhonovRegularizationV2)
    {
        INFO = io::xprintf("%sV2=%0.2f, ", effectSizeV2, INFO.c_str());
        additionalRegularizationVectors += 3;
    }
    if(tikhonovRegularizationLaplace)
    {
        INFO = io::xprintf("%sLaplace=%0.2f, ", effectSizeLaplace, INFO.c_str());
        additionalRegularizationVectors++;
    }
    INFO = io::xprintf("%sinit", effectSizeLaplace, INFO.c_str());
    tikhonovSetRegularizingBuffersNull();
    allocateXBuffers(2 + 3 * additionalRegularizationVectors); // We neeed three new X buffers per
                                                               // one regularization vector
    allocateBBuffers(2);
    LOGD << printTime(INFO, false, true);
    uint32_t iteration = 1;
    // Initialization
    double norm, normDiscrepancy2 = 0.0, normL22 = 0.0, normV2x2 = 0.0, normV2y2 = 0.0,
                 normV2z2 = 0.0, normLaplace2 = 0.0;

    double residualNorm2_old, residualNorm2_now, AdirectionNorm2, AdirectionNorm2Xpart;
    double alpha, beta;
    double NB0 = std::sqrt(normBBuffer_barrier_double(*b_buf));
    double NR0, NX;
    LOGI << io::xprintf("||b||=%f", NB0);
    // Allocating vectors representing x
    std::shared_ptr<cl::Buffer> directionVector_xbuf, residualVector_xbuf; // X buffers
    std::shared_ptr<cl::Buffer> discrepancy_bbuf, AdirectionVector_bbuf; // B buffers
    directionVector_xbuf = getXBuffer(0);
    residualVector_xbuf = getXBuffer(1);
    discrepancy_bbuf = getBBuffer(0);
    AdirectionVector_bbuf = getBBuffer(1);
    // Tikhonov buffers initialization
    uint32_t xBufferIndex = 1;
    if(tikhonovRegularizationL2)
    {
        residualVector_xbuf_L2add = getXBuffer(xBufferIndex + 1);
        discrepancy_bbuf_xpart_L2 = getXBuffer(xBufferIndex + 2);
        AdirectionVector_bbuf_xpart_L2 = getXBuffer(xBufferIndex + 3);
        xBufferIndex = xBufferIndex + 3;
    }
    if(tikhonovRegularizationV2)
    {
        residualVector_xbuf_V2xadd = getXBuffer(xBufferIndex + 1);
        residualVector_xbuf_V2yadd = getXBuffer(xBufferIndex + 2);
        residualVector_xbuf_V2zadd = getXBuffer(xBufferIndex + 3);
        discrepancy_bbuf_xpart_V2x = getXBuffer(xBufferIndex + 4);
        discrepancy_bbuf_xpart_V2y = getXBuffer(xBufferIndex + 5);
        discrepancy_bbuf_xpart_V2z = getXBuffer(xBufferIndex + 6);
        AdirectionVector_bbuf_xpart_V2x = getXBuffer(xBufferIndex + 7);
        AdirectionVector_bbuf_xpart_V2y = getXBuffer(xBufferIndex + 8);
        AdirectionVector_bbuf_xpart_V2z = getXBuffer(xBufferIndex + 9);
        xBufferIndex = xBufferIndex + 9;
    }
    if(tikhonovRegularizationLaplace)
    {
        residualVector_xbuf_Laplaceadd = getXBuffer(xBufferIndex + 1);
        discrepancy_bbuf_xpart_Laplace = getXBuffer(xBufferIndex + 2);
        AdirectionVector_bbuf_xpart_Laplace = getXBuffer(xBufferIndex + 3);
        xBufferIndex = xBufferIndex + 3;
    }
    // discrepancy_bbuf stores initial discrepancy
    algFLOATvector_copy(*b_buf, *discrepancy_bbuf, BDIM);
    tikhonovZeroDiscrepancyBuffers();
    if(useVolumeAsInitialX0)
    {
        project(*x_buf, *AdirectionVector_bbuf);
        tikhonovMatrixActionToAdirectionAndScale(*x_buf);
        algFLOATvector_A_equals_A_plus_cB(*discrepancy_bbuf, *AdirectionVector_bbuf, -1.0f, BDIM);
        tikhonov_discrepancy_equals_discrepancy_minus_alphaAdirection(1.0);
        reportTime("Projection x0", false, true);
    } else
    {
        Q[0]->enqueueFillBuffer<cl_float>(*x_buf, FLOATZERO, 0, XDIM * sizeof(float));
    }
    backproject(*discrepancy_bbuf, *residualVector_xbuf);
    tikhonovMatrixActionOnDiscrepancyToUpdateResidualVector(*residualVector_xbuf);
    algFLOATvector_copy(*residualVector_xbuf, *directionVector_xbuf, XDIM);
    residualNorm2_old = normXBuffer_barrier_double(*residualVector_xbuf);
    reportTime("Backprojection 0", false, true);
    NR0 = std::sqrt(residualNorm2_old);
    project(*directionVector_xbuf, *AdirectionVector_bbuf);
    tikhonovMatrixActionToAdirectionAndScale(*directionVector_xbuf);
    AdirectionNorm2 = normBBuffer_barrier_double(*AdirectionVector_bbuf);
    AdirectionNorm2Xpart = tikhonovSumOfAdirectionNorms2();
    reportTime("Projection 1", false, true);
    alpha = residualNorm2_old / (AdirectionNorm2 + AdirectionNorm2Xpart);
    algFLOATvector_A_equals_A_plus_cB(*x_buf, *directionVector_xbuf, alpha, XDIM);
    algFLOATvector_A_equals_A_plus_cB(*discrepancy_bbuf, *AdirectionVector_bbuf, -alpha, BDIM);
    tikhonov_discrepancy_equals_discrepancy_minus_alphaAdirection(alpha);
    normDiscrepancy2 = normBBuffer_barrier_double(*discrepancy_bbuf);
    if(tikhonovRegularizationL2)
    {
        normL22 = normXBuffer_barrier_double(*discrepancy_bbuf_xpart_L2);
    }
    if(tikhonovRegularizationV2)
    {
        normV2x2 = normXBuffer_barrier_double(*discrepancy_bbuf_xpart_V2x);
        normV2y2 = normXBuffer_barrier_double(*discrepancy_bbuf_xpart_V2y);
        normV2z2 = normXBuffer_barrier_double(*discrepancy_bbuf_xpart_V2z);
    }
    if(tikhonovRegularizationLaplace)
    {
        normLaplace2 = normXBuffer_barrier_double(*discrepancy_bbuf_xpart_Laplace);
    }
    norm = std::sqrt(normDiscrepancy2 + normL22 + normV2x2 + normV2y2 + normV2z2 + normLaplace2);
    while(norm / NB0 > errCondition && iteration < maxIterations)
    {
        if(reportKthIteration > 0 && iteration % reportKthIteration == 0)
        {
            LOGD << io::xprintf("Writing file %sx_it%02d.den", intermediatePrefix.c_str(),
                                iteration);
            writeVolume(*x_buf,
                        io::xprintf("%sx_it%02d.den", intermediatePrefix.c_str(), iteration));
        }
        // DEBUG
        if(iteration % 1000 == 0)
        {
            project(*x_buf, *discrepancy_bbuf);
            tikhonovMatrixActionToDiscrepancyAndScale(*x_buf);
            algFLOATvector_A_equals_Ac_plus_B(*discrepancy_bbuf, *b_buf, -1.0, BDIM);
            // Comparing the following with alredy computed to estimate LOO
            double normDiscrepancy2_, normL22_ = 0.0, normV2x2_ = 0.0, normV2y2_ = 0.0,
                                      normV2z2_ = 0.0, normLaplace2_ = 0.0, norm_;
            normDiscrepancy2_ = normBBuffer_barrier_double(*discrepancy_bbuf);
            if(tikhonovRegularizationL2)
            {
                normL22_ = normXBuffer_barrier_double(*discrepancy_bbuf_xpart_L2);
            }
            if(tikhonovRegularizationV2)
            {
                normV2x2_ = normXBuffer_barrier_double(*discrepancy_bbuf_xpart_V2x);
                normV2y2_ = normXBuffer_barrier_double(*discrepancy_bbuf_xpart_V2y);
                normV2z2_ = normXBuffer_barrier_double(*discrepancy_bbuf_xpart_V2z);
            }
            if(tikhonovRegularizationLaplace)
            {
                normLaplace2_ = normXBuffer_barrier_double(*discrepancy_bbuf_xpart_Laplace);
            }
            norm_ = std::sqrt(normDiscrepancy2_ + normL22_ + normV2x2_ + normV2y2_ + normV2z2_
                              + normLaplace2_);
            reportTime(io::xprintf("Reothrogonalization projection %d", iteration), false, true);
            LOGI << io::xprintf_green(
                "\nReorthogonalization in iteration %d: sqrt(|Ax-b|^2+|Tx|^2)=%0.1f representing "
                "%0.2f%% of |b|, "
                "|Ax-b|=%0.1f representing %0.2f%% of |b|, "
                "|Ax-b|=%0.1f, |L2|=%0.1f, |V2|=%0.1f, |V2x|=%0.1f, |V2y|=%0.1f, |V2z|=%0.1f, "
                "|Laplace|=%0.1f, loss of orthogonality %f%%.",
                iteration, norm_, 100.0 * norm_ / NB0, std::sqrt(normDiscrepancy2_),
                100.0 * std::sqrt(normDiscrepancy2_) / NB0, std::sqrt(normDiscrepancy2_),
                std::sqrt(normL22_), std::sqrt(normV2x2_ + normV2y2_ + normV2z2_),
                std::sqrt(normV2x2_), std::sqrt(normV2y2_), std::sqrt(normV2z2_),
                std::sqrt(normLaplace2_), 100 * std::abs(norm - norm_) / norm);
        }
        // DEBUG
        backproject(*discrepancy_bbuf, *residualVector_xbuf);
        tikhonovMatrixActionOnDiscrepancyToUpdateResidualVector(*residualVector_xbuf);
        residualNorm2_now = normXBuffer_barrier_double(*residualVector_xbuf);
        reportTime(io::xprintf("Backprojection %d", iteration), false, true);
        // Delayed update of residual vector
        beta = residualNorm2_now / residualNorm2_old;
        NX = std::sqrt(residualNorm2_now);
        LOGI << io::xprintf_green(
            "\nIteration %d: sqrt(|Ax-b|^2+|Tx|^2)=%0.1f representing %0.2f%% of |b|, "
            "|Ax-b|=%0.1f representing %0.2f%% of |b|, "
            "sqrt(|AT(Ax-b)|^2 + |TTTx|^2)=%0.2f representing %0.3f%% of "
            "sqrt(|AT(Ax0-b)|^2+|TTTx0|^2), "
            "|Ax-b|=%0.1f, |L2|=%0.1f, |V2|=%0.1f, |V2x|=%0.1f, |V2y|=%0.1f, |V2z|=%0.1f, "
            "|Laplace|=%0.1f.",
            iteration, norm, 100.0 * norm / NB0, std::sqrt(normDiscrepancy2),
            100.0 * std::sqrt(normDiscrepancy2) / NB0, NX, 100 * NX / NR0,
            std::sqrt(normDiscrepancy2), std::sqrt(normL22),
            std::sqrt(normV2x2 + normV2y2 + normV2z2), std::sqrt(normV2x2), std::sqrt(normV2y2),
            std::sqrt(normV2z2), std::sqrt(normLaplace2));
        algFLOATvector_A_equals_Ac_plus_B(*directionVector_xbuf, *residualVector_xbuf, beta, XDIM);
        // Delayed update of direction vector
        iteration = iteration + 1;
        residualNorm2_old = residualNorm2_now;
        project(*directionVector_xbuf, *AdirectionVector_bbuf);
        tikhonovMatrixActionToAdirectionAndScale(*directionVector_xbuf);
        AdirectionNorm2 = normBBuffer_barrier_double(*AdirectionVector_bbuf);
        AdirectionNorm2Xpart = tikhonovSumOfAdirectionNorms2();
        reportTime(io::xprintf("Projection %d", iteration), false, true);
        alpha = residualNorm2_old / (AdirectionNorm2 + AdirectionNorm2Xpart);
        algFLOATvector_A_equals_A_plus_cB(*x_buf, *directionVector_xbuf, alpha, XDIM);
        algFLOATvector_A_equals_A_plus_cB(*discrepancy_bbuf, *AdirectionVector_bbuf, -alpha, BDIM);
        tikhonov_discrepancy_equals_discrepancy_minus_alphaAdirection(alpha);
        normDiscrepancy2 = normBBuffer_barrier_double(*discrepancy_bbuf);
        if(tikhonovRegularizationL2)
        {
            normL22 = normXBuffer_barrier_double(*discrepancy_bbuf_xpart_L2);
        }
        if(tikhonovRegularizationV2)
        {
            normV2x2 = normXBuffer_barrier_double(*discrepancy_bbuf_xpart_V2x);
            normV2y2 = normXBuffer_barrier_double(*discrepancy_bbuf_xpart_V2y);
            normV2z2 = normXBuffer_barrier_double(*discrepancy_bbuf_xpart_V2z);
        }
        if(tikhonovRegularizationLaplace)
        {
            normLaplace2 = normXBuffer_barrier_double(*discrepancy_bbuf_xpart_Laplace);
        }
        norm
            = std::sqrt(normDiscrepancy2 + normL22 + normV2x2 + normV2y2 + normV2z2 + normLaplace2);
    }
    LOGI << io::xprintf_green(
        "\nIteration %d: sqrt(|Ax-b|^2+|Tx|^2)=%0.1f representing %0.2f%% of |b|, "
        "|Ax-b|=%0.1f representing %0.2f%% of |b|, "
        "|Ax-b|=%0.1f, |L2|=%0.1f, |V2|=%0.1f, |V2x|=%0.1f, |V2y|=%0.1f, |V2z|=%0.1f, "
        "|Laplace|=%0.1f.",
        iteration, norm, 100.0 * norm / NB0, std::sqrt(normDiscrepancy2),
        100.0 * std::sqrt(normDiscrepancy2) / NB0, std::sqrt(normDiscrepancy2), std::sqrt(normL22),
        std::sqrt(normV2x2 + normV2y2 + normV2z2), std::sqrt(normV2x2), std::sqrt(normV2y2),
        std::sqrt(normV2z2), std::sqrt(normLaplace2));
    Q[0]->enqueueReadBuffer(*x_buf, CL_TRUE, 0, sizeof(float) * XDIM, x);
    return 0;
}

int CGLSPBCTReconstructor::reconstructDiagonalPreconditioner(
    std::shared_ptr<cl::Buffer> invertedpreconditioner_xbuf,
    uint32_t maxIterations,
    float errCondition)
{
    bool blockingReport = true;
    reportTime("WELCOME TO CGLS WITH PRECONDITIONING, init", blockingReport, true);
    LOGI << io::xprintf("Maximum number of iterations is set to %d and errCondition %f",
                        maxIterations, errCondition);
    uint32_t iteration = 1;

    // Initialization
    double norm, residualNorm2_old, residualNorm2_now, AdirectionNorm2, alpha, beta;
    double NB0 = std::sqrt(normBBuffer_barrier_double(*b_buf));
    double NR0, NX;
    LOGI << io::xprintf("||b||=%f", NB0);
    std::shared_ptr<cl::Buffer> directionVector_xbuf, residualVector_xbuf,
        preconditionedResidualVector_xbuf; // X buffers
    allocateXBuffers(3);
    directionVector_xbuf = getXBuffer(0);
    residualVector_xbuf = getXBuffer(1);
    preconditionedResidualVector_xbuf = getXBuffer(2);
    allocateBBuffers(2);
    std::shared_ptr<cl::Buffer> discrepancy_bbuf, AdirectionVector_bbuf; // B buffers
    discrepancy_bbuf = getBBuffer(0);
    AdirectionVector_bbuf = getBBuffer(1);

    // INITIALIZATION x_0 is initialized typically by zeros but in general by supplied array
    // c_0 is filled by b
    // v_0=w_0=BACKPROJECT(c_0)
    // writeProjections(*discrepancy_bbuf, io::xprintf("/tmp/cgls/c_0.den"));
    algFLOATvector_copy(*b_buf, *discrepancy_bbuf,
                        BDIM); // discrepancy_bbuf stores initial discrepancy
    if(useVolumeAsInitialX0)
    {
        setTimestamp(blockingReport);
        project(*x_buf, *AdirectionVector_bbuf);
        reportTime("Projection x0", blockingReport, true);
        algFLOATvector_A_equals_A_plus_cB(*discrepancy_bbuf, *AdirectionVector_bbuf, -1.0, BDIM);
    } else
    {
        Q[0]->enqueueFillBuffer<cl_float>(*x_buf, FLOATZERO, 0, XDIM * sizeof(float));
    }
    setTimestamp(blockingReport);
    backproject(*discrepancy_bbuf, *residualVector_xbuf);
    reportTime("Backprojection 0", blockingReport, true);
    algFLOATvector_copy(*residualVector_xbuf, *directionVector_xbuf, XDIM);
    algFLOATvector_A_equals_A_times_B(*directionVector_xbuf, *invertedpreconditioner_xbuf, XDIM);
    algFLOATvector_copy(*directionVector_xbuf, *preconditionedResidualVector_xbuf, XDIM);
    residualNorm2_old = scalarProductXBuffer_barrier_double(*residualVector_xbuf,
                                                            *preconditionedResidualVector_xbuf);
    NR0 = std::sqrt(residualNorm2_old);
    // DEBUG
    NR0 = std::sqrt(normXBuffer_barrier_double(*residualVector_xbuf));
    setTimestamp(blockingReport);
    project(*directionVector_xbuf, *AdirectionVector_bbuf);
    reportTime("Projection 1", blockingReport, true);
    AdirectionNorm2 = normBBuffer_barrier_double(*AdirectionVector_bbuf);
    alpha = residualNorm2_old / AdirectionNorm2;
    algFLOATvector_A_equals_A_plus_cB(*x_buf, *directionVector_xbuf, alpha, XDIM);
    algFLOATvector_A_equals_A_plus_cB(*discrepancy_bbuf, *AdirectionVector_bbuf, -alpha, BDIM);
    norm = std::sqrt(normBBuffer_barrier_double(*discrepancy_bbuf));
    while(norm / NB0 > errCondition && iteration < maxIterations)
    {
        if(reportKthIteration > 0 && iteration % reportKthIteration == 0)
        {
            LOGD << io::xprintf("Writing file %sx_it%02d.den", intermediatePrefix.c_str(),
                                iteration);
            writeVolume(*x_buf,
                        io::xprintf("%sx_it%02d.den", intermediatePrefix.c_str(), iteration));
        }
        // DEBUG
        if(iteration % 10 == 0)
        {
            setTimestamp(blockingReport);
            project(*x_buf, *discrepancy_bbuf);
            reportTime(io::xprintf("Reothrogonalization projection %d", iteration), blockingReport,
                       true);
            algFLOATvector_A_equals_Ac_plus_B(*discrepancy_bbuf, *b_buf, -1.0, BDIM);
            double norm2 = std::sqrt(normBBuffer_barrier_double(*discrepancy_bbuf));

            LOGI << io::xprintf_green(
                "Iteration %d, the norm of |Ax-b| is %f that is %0.2f%% of |b|, norms "
                "loss of orthogonality %f%%.",
                iteration, norm2, 100.0 * norm2 / NB0, 100 * (norm2 - norm) / norm);
        }
        // DEBUG
        setTimestamp(blockingReport);
        backproject(*discrepancy_bbuf, *residualVector_xbuf);
        algFLOATvector_C_equals_A_times_B(*residualVector_xbuf, *invertedpreconditioner_xbuf,
                                          *preconditionedResidualVector_xbuf, XDIM);
        //        writeVolume(
        //            *residualVector_xbuf,
        //            io::xprintf("%sresidualVector_xbuf_it%02d.den", intermediatePrefix.c_str(),
        //            iteration));
        //        writeVolume(*preconditionedResidualVector_xbuf,
        //                    io::xprintf("%spreconditionedResidualVector_xbuf_it%02d.den",
        //                                intermediatePrefix.c_str(), iteration));
        reportTime(io::xprintf("Backprojection %d", iteration), blockingReport, true);
        residualNorm2_now = scalarProductXBuffer_barrier_double(*residualVector_xbuf,
                                                                *preconditionedResidualVector_xbuf);
        // Delayed update of residual vector
        beta = residualNorm2_now / residualNorm2_old;
        LOGI << io::xprintf("Beta=%f/%f is %f", residualNorm2_now, residualNorm2_old, beta);
        NX = std::sqrt(residualNorm2_now);
        // DEBUG
        NX = std::sqrt(normXBuffer_barrier_double(*residualVector_xbuf));
        // DEBUG
        LOGI << io::xprintf_green(
            "Iteration %d: |Ax-b|=%0.1f that is %0.2f%% of |b|, |AT(Ax-b)|=%0.2f "
            "that is %0.3f%% of |AT(Ax0-b)|.",
            iteration, norm, 100.0 * norm / NB0, NX, 100 * NX / NR0);
        algFLOATvector_A_equals_Ac_plus_B(*directionVector_xbuf, *preconditionedResidualVector_xbuf,
                                          beta, XDIM);
        // Delayed update of direction vector
        iteration = iteration + 1;
        residualNorm2_old = residualNorm2_now;
        setTimestamp(blockingReport);
        project(*directionVector_xbuf, *AdirectionVector_bbuf);
        reportTime(io::xprintf("Projection %d", iteration), blockingReport, true);
        AdirectionNorm2 = normBBuffer_barrier_double(*AdirectionVector_bbuf);
        alpha = residualNorm2_old / AdirectionNorm2;
        algFLOATvector_A_equals_A_plus_cB(*x_buf, *directionVector_xbuf, alpha, XDIM);
        algFLOATvector_A_equals_A_plus_cB(*discrepancy_bbuf, *AdirectionVector_bbuf, -alpha, BDIM);
        norm = std::sqrt(normBBuffer_barrier_double(*discrepancy_bbuf));
    }
    LOGI << io::xprintf_green("Iteration %d, the norm of |Ax-b| is %f that is %0.2f%% of |b|.",
                              iteration, norm, 100.0 * norm / NB0);
    Q[0]->enqueueReadBuffer(*x_buf, CL_TRUE, 0, sizeof(float) * XDIM, x);
    return 0;
}

int CGLSPBCTReconstructor::reconstruct_experimental(uint32_t maxIterations, float errCondition)
{
    LOGI << io::xprintf(
        "WELCOME TO Experimental implementation CGLS using nonstandard cone beam operator");
    std::shared_ptr<cl::Buffer> v_buf, w_buf; // X buffers
    allocateXBuffers(2);
    v_buf = getXBuffer(0);
    w_buf = getXBuffer(1);
    allocateBBuffers(2);
    std::shared_ptr<cl::Buffer> c_buf, d_buf; // B buffers
    c_buf = getBBuffer(0);
    d_buf = getBBuffer(1);
    reportTime("CGLS INIT", false, true);
    if(verbose)
    {
        // writeProjections(*b_buf, io::xprintf("%sb.den", intermediatePrefix.c_str()));
        // writeVolume(*x_buf, io::xprintf("%sx_0.den", intermediatePrefix.c_str()));
    }
    uint32_t iteration = 0;
    double norm, vnorm2_old, vnorm2_now, dnorm2_old, alpha, beta;
    double NB0 = std::sqrt(normBBuffer_barrier_double(*b_buf));
    norm = NB0;
    project(*x_buf, *tmp_b_buf);
    reportTime("X_0 projection", false, true);
    algFLOATvector_A_equals_A_plus_cB(*tmp_b_buf, *b_buf, -1.0, BDIM);
    norm = std::sqrt(normBBuffer_barrier_double(*tmp_b_buf));
    // Experimental
    cl_int err;
    std::shared_ptr<cl::Buffer> v_proj = std::make_shared<cl::Buffer>(
        *context, CL_MEM_READ_WRITE, sizeof(float) * pdimx * pdimy, nullptr, &err);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
        return -1;
    }
    std::shared_ptr<cl::Buffer> w_proj = std::make_shared<cl::Buffer>(
        *context, CL_MEM_READ_WRITE, sizeof(float) * pdimx * pdimy, nullptr, &err);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
        return -1;
    }
    std::shared_ptr<cl::Buffer> x_proj = std::make_shared<cl::Buffer>(
        *context, CL_MEM_READ_WRITE, sizeof(float) * pdimx * pdimy, nullptr, &err);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
        return -1;
    }
    Q[0]->enqueueFillBuffer<cl_float>(*v_proj, FLOATZERO, 0,
                                      uint32_t(pdimx) * pdimy * sizeof(float));
    Q[0]->enqueueFillBuffer<cl_float>(*w_proj, FLOATZERO, 0,
                                      uint32_t(pdimx) * pdimy * sizeof(float));
    Q[0]->enqueueFillBuffer<cl_float>(*x_proj, FLOATZERO, 0,
                                      uint32_t(pdimx) * pdimy * sizeof(float));
    // Experimental
    LOGI << io::xprintf("Initial norm of b is %f and initial |Ax-b| is %f.", NB0, norm);
    // INITIALIZATION x_0 is initialized typically by zeros but in general by supplied array
    // c_0 is filled by b
    // v_0=w_0=BACKPROJECT(c_0)
    // writeProjections(*c_buf, io::xprintf("/tmp/cgls/c_0.den"));
    backproject(*c_buf, *v_buf);
    // Experimental
    LOGI << "Backprojection correction vector";
    Q[0]->enqueueFillBuffer<cl_float>(*v_proj, FLOATZERO, 0, pdimx * pdimy * sizeof(float));
    for(unsigned int i = 0; i != pdimz; i++)
    {
        algFLOATvector_B_equals_A_plus_B_offsets(*c_buf, i * pdimx * pdimy, *v_proj, 0,
                                                 uint32_t(pdimx) * pdimy);
    }
    // Experimental
    reportTime("v_0 backprojection", false, true);
    if(verbose)
    {
        // writeVolume(*v_buf, io::xprintf("%sv_0.den", intermediatePrefix.c_str()));
    }
    vnorm2_old = normXBuffer_barrier_double(*v_buf);
    // EXPERIMENTAL
    double sum;
    uint32_t framesize = pdimx * pdimy;
    cl::EnqueueArgs eargs1(*Q[0], cl::NDRange(1));
    (*vector_NormSquarePartial)(eargs1, *v_proj, *tmp_red1[0], framesize).wait();
    Q[0]->enqueueReadBuffer(*tmp_red1[0], CL_TRUE, 0, sizeof(double), &sum);
    vnorm2_old += sum;
    // EXPERIMENTAL
    algFLOATvector_copy(*v_buf, *w_buf, XDIM);
    // EXPERIMENTAL
    algFLOATvector_copy(*v_proj, *w_proj, pdimx * pdimy);
    // EXPERIMENTAL
    if(verbose)
    {
        // writeVolume(*w_buf, io::xprintf("%sw_0.den", intermediatePrefix.c_str()));
    }
    project(*w_buf, *d_buf);
    // Experimental
    for(unsigned int i = 0; i != pdimz; i++)
    {
        algFLOATvector_B_equals_A_plus_B_offsets(*w_proj, 0, *d_buf, i * pdimx * pdimy,
                                                 pdimx * pdimy);
    }
    // Experimental
    reportTime("d_0 projection", false, true);
    if(verbose)
    {
        // writeProjections(*d_buf, io::xprintf("%sd_0.den", intermediatePrefix.c_str()));
    }
    dnorm2_old = normBBuffer_barrier_double(*d_buf);
    while(norm / NB0 > errCondition && iteration < maxIterations)
    {
        // Iteration
        iteration = iteration + 1;
        alpha = vnorm2_old / dnorm2_old;
        LOGI << io::xprintf("After iteration %d, |v|^2=%E, |d|^2=%E, alpha=%E", iteration - 1,
                            vnorm2_old, dnorm2_old, float(alpha));
        algFLOATvector_A_equals_A_plus_cB(*x_buf, *w_buf, alpha, XDIM);
        // EXPERIMENTAL
        algFLOATvector_A_equals_A_plus_cB(*x_proj, *w_proj, alpha, pdimx * pdimy);
        // EXPERIMENTAL
        if(verbose)
        {
            writeVolume(*x_buf, io::xprintf("%sx_%d.den", intermediatePrefix.c_str(), iteration));
        }
        algFLOATvector_A_equals_A_plus_cB(*c_buf, *d_buf, -alpha, BDIM);
        if(verbose)
        {
            //    writeProjections(*c_buf,
            //                     io::xprintf("%sc_%d.den", intermediatePrefix.c_str(),
            //                     iteration));
        }
        backproject(*c_buf, *v_buf);
        // Experimental
        Q[0]->enqueueFillBuffer<cl_float>(*v_proj, FLOATZERO, 0, pdimx * pdimy * sizeof(float));
        for(unsigned int i = 0; i != pdimz; i++)
        {
            algFLOATvector_B_equals_A_plus_B_offsets(*c_buf, i * pdimx * pdimy, *v_proj, 0,
                                                     pdimx * pdimy);
        }
        // Experimental
        reportTime(io::xprintf("v_%d backprojection", iteration), false, true);

        if(verbose)
        {
            //    writeVolume(*v_buf, io::xprintf("%sv_%d.den", intermediatePrefix.c_str(),
            //    iteration));
        }
        vnorm2_now = normXBuffer_barrier_double(*v_buf);
        // EXPERIMENTAL
        cl::EnqueueArgs eargs1(*Q[0], cl::NDRange(1));
        (*vector_NormSquarePartial)(eargs1, *v_proj, *tmp_red1[0], framesize).wait();
        Q[0]->enqueueReadBuffer(*tmp_red1[0], CL_TRUE, 0, sizeof(double), &sum);
        vnorm2_now += sum;
        // EXPERIMENTAL
        beta = vnorm2_now / vnorm2_old;
        LOGI << io::xprintf("In iteration %d, |v_now|^2=%E, |v_old|^2=%E, beta=%0.2f", iteration,
                            vnorm2_now, vnorm2_old, beta);
        vnorm2_old = vnorm2_now;
        algFLOATvector_A_equals_Ac_plus_B(*w_buf, *v_buf, beta, XDIM);
        // EXPERIMENTAL
        algFLOATvector_A_equals_Ac_plus_B(*w_proj, *v_proj, beta, pdimx * pdimy);
        // EXPERIMENTAL
        if(verbose)
        {
            //    writeVolume(*w_buf, io::xprintf("%sw_%d.den", intermediatePrefix.c_str(),
            //    iteration));
        }
        algFLOATvector_A_equals_A_plus_cB(*tmp_b_buf, *d_buf, alpha, BDIM);
        project(*w_buf, *d_buf);
        // EXPERIMENTAL
        for(unsigned int i = 0; i != pdimz; i++)
        {
            algFLOATvector_B_equals_A_plus_B_offsets(*w_proj, 0, *d_buf, i * pdimx * pdimy,
                                                     pdimx * pdimy);
        }
        // EXPERIMENTAL
        reportTime(io::xprintf("d_%d projection", iteration), false, true);
        if(verbose)
        {
            //    writeProjections(*d_buf,
            //                     io::xprintf("%sd_%d.den", intermediatePrefix.c_str(),
            //                     iteration));
        }
        dnorm2_old = normBBuffer_barrier_double(*d_buf);

        norm = std::sqrt(normBBuffer_barrier_double(*tmp_b_buf));
        LOGI << io::xprintf_green(
            "After iteration %d, the norm of |Ax-b| is %f that is %0.2f%% of NB0.", iteration, norm,
            100.0 * norm / NB0);
    }
    // Optionally write even more converged solution
    alpha = vnorm2_old / dnorm2_old;
    LOGI << io::xprintf("Finally |v|^2=%E, |d|^2=%E, alpha=%E", iteration - 1, vnorm2_old,
                        dnorm2_old, float(alpha));
    algFLOATvector_A_equals_A_plus_cB(*x_buf, *w_buf, alpha, XDIM);
    // EXPERIMENTAL
    algFLOATvector_A_equals_A_plus_cB(*x_proj, *w_proj, alpha, pdimx * pdimy);
    // EXPERIMENTAL
    Q[0]->enqueueReadBuffer(*x_buf, CL_TRUE, 0, sizeof(float) * XDIM, x);
    return 0;
}

int CGLSPBCTReconstructor::reconstructJacobi(uint32_t maxIterations, float errCondition)
{
    bool minmaxfiltering = true;
    std::shared_ptr<cl::Buffer> preconditioner_xbuf; // X buffers
    allocateXBuffers(4 + minmaxfiltering);
    preconditioner_xbuf = getXBuffer(3);
    precomputeJacobiPreconditioner(preconditioner_xbuf);
    algFLOATvector_invert_except_zero(*preconditioner_xbuf, XDIM);
    if(minmaxfiltering)
    {
        KCTERR("minmaxfiltering is not implemented for parallel ray geometry.")
        std::shared_ptr<cl::Buffer> minmax_xbuf = getXBuffer(4);
        // backproject_minmax(*b_buf, *minmax_xbuf);
        // TODO: Is the following line neccessarry?
        // Without this problematic
        algFLOATvector_substitute_greater_than(*minmax_xbuf, 0.0, 1.0, XDIM);
        writeVolume(*minmax_xbuf, io::xprintf("%sx_minmax.den", intermediatePrefix.c_str()));
        algFLOATvector_A_equals_A_times_B(*preconditioner_xbuf, *minmax_xbuf, XDIM);
    }
    LOGD << io::xprintf("Writing file %s_preconditioner.inv", intermediatePrefix.c_str());
    writeVolume(*preconditioner_xbuf,
                io::xprintf("%sx_preconditioner.inv", intermediatePrefix.c_str()));
    reconstructDiagonalPreconditioner(preconditioner_xbuf, maxIterations, errCondition);
    return 0;
}

void CGLSPBCTReconstructor::precomputeJacobiPreconditioner(std::shared_ptr<cl::Buffer> X)
{
    KCTERR("Not implemented for parallel ray geometry.");
}

int CGLSPBCTReconstructor::reconstructDiagonalPreconditioner(float* invertedpreconditioner,
                                                             uint32_t maxIterations,
                                                             float errCondition)
{
    std::shared_ptr<cl::Buffer> preconditioner_xbuf; // X buffers
    allocateXBuffers(4);
    preconditioner_xbuf = getXBuffer(3);
    arrayIntoBuffer(invertedpreconditioner, *preconditioner_xbuf, XDIM);
    double norm = std::sqrt(normXBuffer_barrier_double(*preconditioner_xbuf));
    LOGI << io::xprintf_green("Norm %f", norm);
    reconstructDiagonalPreconditioner(preconditioner_xbuf, maxIterations, errCondition);
    return 0;
}

} // namespace KCT
