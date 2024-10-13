#include "PDHGPBCT2DReconstructor.hpp"

namespace KCT {

int PDHGPBCT2DReconstructor::reconstruct(uint32_t maxIterations, float errCondition)
{
    LOGI << "Virtual method PDHGPBCT2DReconstructor::reconstruct(maxIterations, errCondition) not "
            "implemented for PHDGPBCT2DReconstructor, call int "
            "PDHGPBCT2DReconstructor::reconstruct(float mu, float tau, float sigma, float "
            "theta, uint32_t maxPDHGIterations, float errConditionPDHG, uint32_t "
            "maxCGLSIterations, float errConditionCGLS) instead.";
    return 0;
}

int PDHGPBCT2DReconstructor::proximalOperatorCGLS(cl::Buffer x0prox_xbufIN,
                                                  cl::Buffer x_xbufOUT,
                                                  float tau,
                                                  uint32_t maxCGLSIterations,
                                                  float errConditionCGLS,
                                                  uint32_t outerIterationIndex)
// Solve the problem 1 / (2 * tau) ||x-x0prox||_2^2 + ||Ax - b||_2^2 by means of CGLS
{
    float effectSize;
    float sqrtEffectSize;
    if(tau == 0.0f)
    {
        effectSize = 0.0f;
        sqrtEffectSize = 0.0f;
    } else
    {
        effectSize = 0.5f / tau;
        sqrtEffectSize = std::sqrt(effectSize);
    }
    uint32_t iteration = 1;
    double norm, residualNorm2_old, residualNorm2_now, AdirectionNorm2, alpha, beta;
    double NX = normXBuffer_barrier_double(x0prox_xbufIN);
    double NF0 = std::sqrt(NB0 * NB0 + effectSize * NX);
    double NR0;
    std::string str = io::xprintf(
        "PDHG %d iteration, Proximal CGLS using tau=%0.2f and |b| + 1/sqrt(2*tau) "
        "|x0prox|=%0.2f, 1/sqrt(2*tau) |x0prox| = %0.2f effectSize=%0.2f, sqrtEffectSize=%0.2f",
        outerIterationIndex, tau, NF0, NX * sqrtEffectSize, effectSize, sqrtEffectSize);
    LOGD << printTime(str, false, true);
    std::string filename
        = io::xprintf("%s_x0prox_it%02d.den", intermediatePrefix.c_str(), outerIterationIndex);
    LOGD << io::xprintf("Writing file %s", filename.c_str());
    BasePBCT2DReconstructor::writeVolume(x0prox_xbufIN, filename);
    Q[0]->enqueueFillBuffer<cl_float>(*discrepancy_bbuf_xpart_L2, FLOATZERO, 0,
                                      XDIM * sizeof(float));

    algFLOATvector_copy(*b_buf, *discrepancy_bbuf, BDIM);
    algFLOATvector_copy(x0prox_xbufIN, x_xbufOUT, XDIM);
    // Use x0prox_xbufIN as initial vector
    project(x_xbufOUT, *AdirectionVector_bbuf);
    algFLOATvector_A_equals_A_plus_cB(*discrepancy_bbuf, *AdirectionVector_bbuf, -1.0f,
                                      BDIM); // Ax - b
    backproject(*discrepancy_bbuf, *residualVector_xbuf); // A^T(Ax - b)
    // Because of the setting x_0 = x0prox, the residual is initially just residual of tomographic
    // reconstruction A^T(Ax - b)

    algFLOATvector_copy(*residualVector_xbuf, *directionVector_xbuf, XDIM);
    residualNorm2_old = normXBuffer_barrier_double(*residualVector_xbuf);
    NR0 = std::sqrt(residualNorm2_old);
    project(*directionVector_xbuf, *AdirectionVector_bbuf);
    AdirectionNorm2 = normBBuffer_barrier_double(*AdirectionVector_bbuf)
        + normXBuffer_barrier_double(*directionVector_xbuf) * effectSize;
    alpha = residualNorm2_old / AdirectionNorm2;

    algFLOATvector_A_equals_A_plus_cB(x_xbufOUT, *directionVector_xbuf, alpha, XDIM);
    algFLOATvector_A_equals_A_plus_cB(*discrepancy_bbuf, *AdirectionVector_bbuf, -alpha, BDIM);
    // Update discrepancy related to proximal part of the right hand side
    // AdirectionVector_bbuf_xpart_L2 = sqrt(effectsSize) * directionVector_xbuf
    algFLOATvector_A_equals_A_plus_cB(*discrepancy_bbuf_xpart_L2, *directionVector_xbuf,
                                      -alpha * sqrtEffectSize, XDIM);
    norm = std::sqrt(normBBuffer_barrier_double(*discrepancy_bbuf)
                     + normXBuffer_barrier_double(*discrepancy_bbuf_xpart_L2));

    while(norm / NF0 > errConditionCGLS && iteration < maxCGLSIterations)
    {
        if(reportKthIteration > 0 && iteration % reportKthIteration == 0)
        {
            LOGD << io::xprintf("Writing file %sx_it%02d.den", intermediatePrefix.c_str(),
                                iteration);
            BasePBCT2DReconstructor::writeVolume(
                x_xbufOUT, io::xprintf("%sx_it%02d.den", intermediatePrefix.c_str(), iteration));
        }
        backproject(*discrepancy_bbuf, *residualVector_xbuf);
        // I have to update the residual vector with proximal part
        // residualVector_xbuf_L2add = sqrtEffectSize * discrepancy_bbuf_xpart_L2
        algFLOATvector_A_equals_A_plus_cB(*residualVector_xbuf, *discrepancy_bbuf_xpart_L2,
                                          sqrtEffectSize, XDIM);
        residualNorm2_now = normXBuffer_barrier_double(*residualVector_xbuf);
        reportTime(io::xprintf("Backprojection %d", iteration), false, true);
        // Delayed update of residual vector
        beta = residualNorm2_now / residualNorm2_old;
        NX = std::sqrt(residualNorm2_now);
        LOGI << io::xprintf_green("\nIteration %d: |Ax-b|=%0.1f representing %0.2f%% of |b|, "
                                  "|AT(Ax-b)|=%0.2f representing %0.3f%% of "
                                  "|AT(Ax0-b)|.",
                                  iteration, norm, 100.0 * norm / NF0, NX, 100 * NX / NR0);
        algFLOATvector_A_equals_Ac_plus_B(*directionVector_xbuf, *residualVector_xbuf, beta, XDIM);
        // Delayed update of direction vector
        iteration = iteration + 1;
        residualNorm2_old = residualNorm2_now;
        project(*directionVector_xbuf, *AdirectionVector_bbuf);
        // AdirectionVector_bbuf_xpart_L2 = sqrt(effectsSize) * directionVector_xbuf
        AdirectionNorm2 = normBBuffer_barrier_double(*AdirectionVector_bbuf)
            + normXBuffer_barrier_double(*directionVector_xbuf) * effectSize;
        reportTime(io::xprintf("Projection %d", iteration), false, true);
        alpha = residualNorm2_old / AdirectionNorm2;
        algFLOATvector_A_equals_A_plus_cB(x_xbufOUT, *directionVector_xbuf, alpha, XDIM);
        algFLOATvector_A_equals_A_plus_cB(*discrepancy_bbuf, *AdirectionVector_bbuf, -alpha, BDIM);
        // Update discrepancy related to proximal part of the right hand side
        algFLOATvector_A_equals_A_plus_cB(*discrepancy_bbuf_xpart_L2, *directionVector_xbuf,
                                          -alpha * sqrtEffectSize, XDIM);
        norm = std::sqrt(normBBuffer_barrier_double(*discrepancy_bbuf)
                         + normXBuffer_barrier_double(*discrepancy_bbuf_xpart_L2));
        LOGD << io::xprintf("Iteration %d: alpha = %f, beta = %f, |Ax-b| = %f, |AT(Ax-b)| = %f",
                            iteration, alpha, beta, norm, NX);
        //    BasePBCT2DReconstructor::writeVolume(*x_buf,
        //                                         io::xprintf("PDHG_x_xbufOUT_it%02d.den",
        //                                         iteration));
    }
    LOGI << io::xprintf_green("\nIteration %d: |Ax-b|=%0.1f representing %0.2f%% of |b|.",
                              iteration, norm, 100.0 * norm / NF0);
    /*    BasePBCT2DReconstructor::writeVolume(
            x_xbufOUT,
            io::xprintf("%sx_cgls_it%02d.den", intermediatePrefix.c_str(), outerIterationIndex));*/
    return 0;
}

int PDHGPBCT2DReconstructor::reconstruct(float mu,
                                         float tau,
                                         float sigma,
                                         float theta,
                                         uint32_t maxPDHGIterations,
                                         float errConditionPDHG,
                                         uint32_t maxCGLSIterations,
                                         float errConditionCGLS)
{
    uint32_t iteration = 1;

    // Initialization of primal and dual variables
    double norm;
    NB0 = std::sqrt(normBBuffer_barrier_double(*b_buf));
    double hsquare = voxelSizesF.x * voxelSizesF.x;
    std::string str
        = io::xprintf("WELCOME TO Chambolle-Pock 2D, PDHGPBCT2DReconstructor, "
                      "mu=%0.2fh^2, tau=%0.2f, sigma=%0.2fh^2, theta=%0.2f ||b||_2=%0.2f",
                      mu / hsquare, tau, sigma / hsquare, theta, NB0);
    LOGD << io::xprintf_blue(printTime(str, false, true));

    std::shared_ptr<cl::Buffer> primal_xbuf, primal_xbuf_prime, primal_xbuf_dx, primal_xbuf_dy,
        dual_xbuf_x, dual_xbuf_y, divergence_xbuf;
    // Primal and dual buffers
    allocateXBuffers(11);
    primal_xbuf = getXBuffer(0); // Primal variable (x)
    primal_xbuf_dx = getXBuffer(1); // Primal variable derivative (x)
    primal_xbuf_dy = getXBuffer(2); // Primal variable derivative (y)
    dual_xbuf_x = getXBuffer(3); // Dual variable (x)
    dual_xbuf_y = getXBuffer(4); // Dual variable (y)
    divergence_xbuf = getXBuffer(5); // For divergence of the dual variable
    // Prealocate CGLS vectors
    directionVector_xbuf = getXBuffer(6); // For direction vector
    residualVector_xbuf = getXBuffer(7); // For residual vector
    residualVector_xbuf_L2add = getXBuffer(8); // For residual vector
    discrepancy_bbuf_xpart_L2 = getXBuffer(9); // For residual vector
    AdirectionVector_bbuf_xpart_L2 = getXBuffer(10); // For residual vector

    allocateBBuffers(2);
    discrepancy_bbuf = getBBuffer(0);
    AdirectionVector_bbuf = getBBuffer(1);

    // Initial setup of primal variable (x)
    if(useVolumeAsInitialX0)
    {
        algFLOATvector_copy(*x_buf, *primal_xbuf, XDIM); // Start with x = x0
    } else
    {
        Q[0]->enqueueFillBuffer<cl_float>(*primal_xbuf, FLOATZERO, 0,
                                          XDIM * sizeof(float)); // Start with x = 0
    }
    std::shared_ptr<cl::Buffer> primal_xbuf_new
        = divergence_xbuf; // Divergence is to be used as a temporary buffer since it is fully
                           // initialized in second step
    proximalOperatorCGLS(*primal_xbuf, *primal_xbuf_new, 0.0, maxCGLSIterations, errConditionCGLS,
                         0);
    algFLOATvector_copy(*primal_xbuf_new, *primal_xbuf, XDIM);
    BasePBCT2DReconstructor::writeVolume(
        *primal_xbuf, io::xprintf("%sx0_it%02d.den", intermediatePrefix.c_str(), iteration));

    // Dual variable initialization
    Q[0]->enqueueFillBuffer<cl_float>(*dual_xbuf_x, FLOATZERO, 0,
                                      XDIM * sizeof(float)); // Dual variable = 0
    Q[0]->enqueueFillBuffer<cl_float>(*dual_xbuf_y, FLOATZERO, 0,
                                      XDIM * sizeof(float)); // Dual variable = 0

    // Initial discrepancy (Ax - b) and backprojection
    // project(*primal_xbuf, *discrepancy_bbuf);
    // backproject(*discrepancy_bbuf, *divergence_xbuf); // This will initialize the residual (x0)

    // Main Chambolle-Pock iteration loop
    cl::NDRange globalRangeGradient(vdimx, vdimy, vdimz);
    cl::NDRange localRangeGradient = cl::NullRange;
    primal_xbuf_prime = primal_xbuf; // At the beginnig they are the same
    bool useROF = true;
    std::shared_ptr<cl::Buffer> rof_u0_xbuf;
    if(useROF)
    {
        rof_u0_xbuf = AdirectionVector_bbuf_xpart_L2;
        algFLOATvector_copy(*primal_xbuf, *rof_u0_xbuf, XDIM);
    }
    while(iteration < maxPDHGIterations)
    {
        // Step 1: Update Dual Variable (p^k+1 = Proj_Dual(p^k + sigma * Grad(x^k)))
        algFLOATvector_2DisotropicGradient(
            *primal_xbuf_prime, *primal_xbuf_dx, *primal_xbuf_dy, vdims, voxelSizesF,
            globalRangeGradient,
            localRangeGradient); // Compute gradient of x (dual variable p update)
        // p = p + sigma * Grad(x)
        algFLOATvector_A_equals_A_plus_cB(*dual_xbuf_x, *primal_xbuf_dx, sigma, XDIM);
        algFLOATvector_A_equals_A_plus_cB(*dual_xbuf_y, *primal_xbuf_dy, sigma, XDIM);
        // Here I need to calculate dual_xbuf_x * dual_xbuf_x + dual_xbuf_y * dual_xbuf_y
        // Then find a pointwise maximum
        // Then compare this by mu and if it is greater than mu, then multiply the dual
        // variable by max/mu - this is L1 proximal operator
        /*
                BasePBCT2DReconstructor::writeVolume(
                    *dual_xbuf_x,
                    io::xprintf("%s_dual_xbuf_x_bef_it%02d.den", intermediatePrefix.c_str(),
           iteration));*/
        algFLOATvector_infProjectionToLambda2DBall(*dual_xbuf_x, *dual_xbuf_y, mu, XDIM);
        /*        BasePBCT2DReconstructor::writeVolume(
                    *dual_xbuf_x,
                    io::xprintf("%s_dual_xbuf_x_aft_it%02d.den", intermediatePrefix.c_str(),
           iteration));*/

        // Step 2: Update Primal Variable (x^k+1 = Prox_F(x^k - tau * Div(p^k+1)))
        algFLOATvector_isotropicBackDivergence2D(*dual_xbuf_x, *dual_xbuf_y, *divergence_xbuf,
                                                 vdims, voxelSizesF, globalRangeGradient,
                                                 localRangeGradient); // Compute divergence of p
        BasePBCT2DReconstructor::writeVolume(
            *divergence_xbuf,
            io::xprintf("%s_divergence_it%02d.den", intermediatePrefix.c_str(), iteration));
        // I need primal_xbuf for update in step 3 so I put argument for primal proximal operator to
        // dirergence_xbuf, which is fully computed in step 2
        std::shared_ptr<cl::Buffer> proximal_arg_xbuf = divergence_xbuf;
        algFLOATvector_A_equals_Ac_plus_B(*proximal_arg_xbuf, *primal_xbuf, -tau,
                                          XDIM); // arg = x - tau * (-Div)(p)

        // Step 2.1: apply proximal operator related to |Ax-b|_2^2
        BasePBCT2DReconstructor::writeVolume(
            *proximal_arg_xbuf,
            io::xprintf("%s_proximal_arg_it%02d.den", intermediatePrefix.c_str(), iteration));
        std::shared_ptr<cl::Buffer> primal_xbuf_new
            = primal_xbuf_dx; // primal_xbuf_dx is to be used as a temporary buffer since it is
                              // fully initialized in the first step
        if(useROF)
        {
            // ROF proximal operator (1/(2*tau)) * ||u - rof_u0||_2^2 +  ||u -
            // proximal_arg_xbuf||_2^2 primal_xbuf_new = (1/(2*tau))/(1/(2*tau) + 1) * rof_u0 +
            // (1/(1/(2*tau) + 1)) * proximal_arg_xbuf
            float rof_prefactor = 1.0f / (1.0f / (2.0f * tau) + 1.0f);
            float arg_prefactor = 1.0f - rof_prefactor;
            LOGD << io::xprintf("ROF proximal operator: arg_prefactor=%f, rof_prefactor=%f",
                                arg_prefactor, rof_prefactor);
            algFLOATvector_C_equals_Ad_plus_Be(*rof_u0_xbuf, *proximal_arg_xbuf, *primal_xbuf_new,
                                               rof_prefactor, arg_prefactor, XDIM);
        } else
        {
            proximalOperatorCGLS(*proximal_arg_xbuf, *primal_xbuf_new, tau, maxCGLSIterations,
                                 errConditionCGLS, iteration);
        }

        // Step 3: Over-relaxation (x̄ = x^k+1 + θ(x^k+1 - x^k))
        if(theta > 0.0f)
        {
            primal_xbuf_prime = residualVector_xbuf; // It is not used right now
            algFLOATvector_C_equals_Ad_plus_Be(*primal_xbuf, *primal_xbuf_new, *primal_xbuf_prime,
                                               -theta, 1.0f + theta, XDIM);
        } else
        {
            primal_xbuf_prime = primal_xbuf_new; // They are the same
        }

        // Swap buffers
        // hack to avoid copying buffers
        std::shared_ptr<cl::Buffer> temp = primal_xbuf;
        primal_xbuf = primal_xbuf_new;
        primal_xbuf_dx = temp;

        BasePBCT2DReconstructor::writeVolume(
            *primal_xbuf, io::xprintf("%sx_it%02d.den", intermediatePrefix.c_str(), iteration));
        BasePBCT2DReconstructor::writeVolume(
            *primal_xbuf,
            io::xprintf("%sx_prime_it%02d.den", intermediatePrefix.c_str(), iteration));
        // Projection and error condition check
        project(*primal_xbuf, *discrepancy_bbuf); // Forward projection of x
        algFLOATvector_A_equals_A_plus_cB(*discrepancy_bbuf, *b_buf, -1.0f, BDIM); // Ax - b
        norm = std::sqrt(normBBuffer_barrier_double(*discrepancy_bbuf)); // Norm of Ax - b

        if(norm / NB0 < errConditionPDHG)
        {
            break; // Exit if error condition is met
        }

        iteration++;
        LOGI << io::xprintf_green("Iteration %d: |Ax-b|=%0.1f representing %0.2f%% of |b|.",
                                  iteration, norm, 100.0 * norm / NB0);
    }

    LOGI << io::xprintf_green("Finished at iteration %d: |Ax-b|=%0.1f.", iteration, norm);
    Q[0]->enqueueReadBuffer(*primal_xbuf, CL_TRUE, 0, sizeof(float) * XDIM, x);
    return 0;
}

} // namespace KCT
