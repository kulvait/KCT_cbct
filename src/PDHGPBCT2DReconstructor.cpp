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

std::array<double, 3>
PDHGPBCT2DReconstructor::computeSolutionNorms(std::shared_ptr<cl::Buffer> x_vector,
                                              std::shared_ptr<cl::Buffer> x_vector_dx,
                                              std::shared_ptr<cl::Buffer> x_vector_dy,
                                              std::shared_ptr<cl::Buffer> x_0,
                                              bool computeDiscrepancy)
{
    double TVNorm = isotropicTVNormXBuffer_barrier_double(*x_vector_dx, *x_vector_dy);
    double DiscrepancyNorm = 0.0;
    if(computeDiscrepancy)
    {
        project(*x_vector, *discrepancy_bbuf); // Forward projection of x
        algFLOATvector_A_equals_A_plus_cB(*discrepancy_bbuf, *b_buf, -1.0f, BDIM); // Ax - b
        DiscrepancyNorm
            = std::sqrt(normBBuffer_barrier_double(*discrepancy_bbuf)); // Norm of Ax - b
    }
    double DifferenceNorm = 0.0;
    if(x_0 != nullptr)
    {
        std::shared_ptr<cl::Buffer> differenceVector_xbuf = discrepancy_bbuf_xpart_L2;
        algFLOATvector_C_equals_Ad_plus_Be(*x_vector, *x_0, *differenceVector_xbuf, -1.0f, 1.0f,
                                           XDIM);
        DifferenceNorm = normXBuffer_barrier_double(*differenceVector_xbuf);
    }
    return { DiscrepancyNorm, TVNorm, DifferenceNorm };
}

// The pointers might point to the same buffer or three different buffers. It is also admissible
// that xbufIN_x_prox and xbufIN_x0 are not equal and xbufOUT is one of them. The procedure will
// handle these situations.
int PDHGPBCT2DReconstructor::proximalOperatorCGLS(std::shared_ptr<cl::Buffer> xbufIN_x_prox,
                                                  std::shared_ptr<cl::Buffer> xbufIN_x0,
                                                  std::shared_ptr<cl::Buffer> xbufOUT,
                                                  float tau,
                                                  uint32_t maxCGLSIterations,
                                                  float errConditionCGLS,
                                                  uint32_t outerIterationIndex)
// Solve the problem 1 / (2 * tau) ||x-x_prox||_2^2 + ||Ax - b||_2^2 by means of CGLS
{
    float effectSize;
    float sqrtEffectSize;
    if(tau == 0.0f)
    {
        effectSize = 0.0f;
        sqrtEffectSize = 0.0f;
        if(proximalOperatorVerbose)
        {
            LOGI << io::xprintf(
                "PDHG %d iteration, CGLS minimizing ||Ax-b||_2^2 ||b||=%0.2f %s %s %s",
                outerIterationIndex, NB0,
                xbufIN_x_prox == xbufIN_x0 ? "x_prox == x0" : "x_prox != x0",
                xbufIN_x_prox == xbufOUT ? "x_prox == x_out" : "x_prox != x_out",
                xbufIN_x0 == xbufOUT ? "x0 == x_out" : "x0 != x_out");
        }
    } else
    {
        effectSize = 0.5f / tau;
        sqrtEffectSize = std::sqrt(effectSize);
        if(proximalOperatorVerbose)
        {
            LOGI << io::xprintf(
                "PDHG %d iteration, CGLS proximal operator ||Ax-b||_2^2 + 1/(2*tau) "
                "||x-x_prox||_2^2, |b|=%0.2f, |ATb|=%0.2f, %d iterations, tau=%0.2f "
                "1/(2*tau)=%0.2f, 1/sqrt(2*tau)=%0.2f %s %s %s",
                outerIterationIndex, NB0, NATB0, maxCGLSIterations, tau, 0.5f / tau,
                std::sqrt(0.5f / tau), xbufIN_x_prox == xbufIN_x0 ? "x_prox == x0" : "x_prox != x0",
                xbufIN_x_prox == xbufOUT ? "x_prox == x_out" : "x_prox != x_out",
                xbufIN_x0 == xbufOUT ? "x0 == x_out" : "x0 != x_out");
        }
    }
    uint32_t iteration = 1;
    double residualNorm2_old, residualNorm2_now, AdirectionNorm2, alpha, beta;
    double NXP2 = normXBuffer_barrier_double(*xbufIN_x_prox); // Squared L2 norm of x_prox
    double NX02; // Squared L2 norm of x0
    // Discrepancy norms
    double NDRHSB2, NDRHSP2,
        NDRHS; // ||Ax - b||_2^2, 1/(2*tau) ||x-x_prox||_2^2 and norm of discrepancy of the right
               // hand side, which is sqrt(NDRHSB2 + NDRHSP2)
    double NDRHSB2_START, NDRHSP2_START, NDRHS_START; // Previous quantities for initial x = x0
    double NDRHSB2_ZEROX, NDRHSP2_ZEROX, NDRHS_ZEROX; // Previous quantities for x = 0
    double NRHSB2, NRHSP2, NR2; // ||AT(Ax - b)||_2^2, 1/(2*tau)^2 ||x-x_prox||_2^2 and norm of
                                // residual  right hand side, which is NRHSB2 + NRHSP2
    double NRHSB2_START, NRHSP2_START, NR2_START; // Previous quantities for initial x = x0
    double NRHSB2_ZEROX, NRHSP2_ZEROX, NR2_ZEROX; // Previous quantities for x = 0

    NDRHSP2_ZEROX = effectSize * NXP2;
    // Set corectly discrepancy_bbuf and discrepancy_bbuf_xpart_L2 for x = xbufIN_x0
    if(xbufIN_x_prox == xbufIN_x0)
    {
        NX02 = NXP2;
        Q[0]->enqueueFillBuffer<cl_float>(*discrepancy_bbuf_xpart_L2, FLOATZERO, 0,
                                          XDIM * sizeof(float));
        NDRHSP2_START = 0.0;
    } else
    {
        NX02 = normXBuffer_barrier_double(*xbufIN_x0);
        algFLOATvector_C_equals_Ad_plus_Be(*xbufIN_x_prox, *xbufIN_x0, *discrepancy_bbuf_xpart_L2,
                                           sqrtEffectSize, -sqrtEffectSize, XDIM);
        NDRHSP2_START = normXBuffer_barrier_double(*discrepancy_bbuf_xpart_L2);
    }
    project(*xbufIN_x0, *AdirectionVector_bbuf);
    algFLOATvector_C_equals_Ad_plus_Be(*b_buf, *AdirectionVector_bbuf, *discrepancy_bbuf, 1.0f,
                                       -1.0f, BDIM); // Ax - b
    NDRHSB2_START = normBBuffer_barrier_double(*discrepancy_bbuf);
    NDRHS_START = std::sqrt(NDRHSB2_START + NDRHSP2_START);
    NDRHSB2_ZEROX = NB0 * NB0;
    NDRHS_ZEROX = std::sqrt(NDRHSB2_ZEROX + NDRHSP2_ZEROX);
    NDRHSB2 = NDRHSB2_START;
    NDRHSP2 = NDRHSP2_START;
    NDRHS = NDRHS_START;
    backproject(*discrepancy_bbuf, *residualVector_xbuf); // A^T(Ax - b)
    NRHSB2_START = normXBuffer_barrier_double(*residualVector_xbuf);
    if(xbufIN_x_prox != xbufIN_x0)
    {
        // Here I have to account for updating residulaVector_xbuf with proximal part
        algFLOATvector_A_equals_A_plus_cB(*residualVector_xbuf, *discrepancy_bbuf_xpart_L2,
                                          sqrtEffectSize, XDIM);
    }
    NRHSP2_START = effectSize * NDRHSP2_START;
    NR2_START = NRHSB2_START + NRHSP2_START;
    NRHSB2_ZEROX = NATB0 * NATB0;
    NRHSP2_ZEROX = effectSize * NDRHSP2_ZEROX;
    NR2_ZEROX = NRHSB2_ZEROX + NRHSP2_ZEROX;
    NRHSB2 = NRHSB2_START;
    NRHSP2 = NRHSP2_START;
    NR2 = NR2_START;
    // After initial norms are computed, I can inform about the initial state
    std::string str = io::xprintf(
        "Initialization 1/sqrt(2*tau)||x_prox||=%0.2f and |(b,  1/sqrt(2*tau) "
        "|x0prox|)|=%0.2f.\n Initial |x_prox| = %f and |Ax0-b|=%0.2f, "
        "1/sqrt(2*tau)||x_prox-x0||=%0.2f and |(Ax0-b, 1/sqrt(2*tau) |x_prox-x0||)=%0.2f. Initial "
        "residuals |ATb|=%f |AT(Ax0-b)|=%0.2f and 1/(2*tau)||x0-x_prox||=%0.2f, |AT(Ax0-b) + "
        "1/(2*tau)(x0-x_prox)|=%0.2f.",
        std::sqrt(NDRHSP2_ZEROX), NDRHS_ZEROX, std::sqrt(NX02), std::sqrt(NDRHSB2_START),
        std::sqrt(NDRHSP2_START), NDRHS_START, std::sqrt(NRHSB2_ZEROX), std::sqrt(NRHSB2_START),
        std::sqrt(NRHSP2_START), std::sqrt(NR2_START));
    if(proximalOperatorVerbose)
    {
        LOGD << printTime(str, false, true);
    }
    algFLOATvector_copy(*residualVector_xbuf, *directionVector_xbuf, XDIM);
    residualNorm2_old = NR2;
    project(*directionVector_xbuf,
            *AdirectionVector_bbuf); // There is proximal part of AdirectionVector of the form
                                     // sqrtEffectSize * directionVector_xbuf
    // Since directionVector_xbuf = residualVector_xbuf, the square of the L2 norm of the proximal
    // part is effectSize * NR2
    AdirectionNorm2 = normBBuffer_barrier_double(*AdirectionVector_bbuf) + effectSize * NR2;
    alpha = residualNorm2_old / AdirectionNorm2;
    // x_new = x_old + alpha * directionVector_xbuf
    if(xbufOUT == xbufIN_x0)
    {
        algFLOATvector_A_equals_A_plus_cB(*xbufOUT, *directionVector_xbuf, alpha, XDIM);
    } else
    {
        algFLOATvector_C_equals_Ad_plus_Be(*xbufIN_x0, *directionVector_xbuf, *xbufOUT, 1.0f, alpha,
                                           XDIM);
    }
    algFLOATvector_A_equals_A_plus_cB(*discrepancy_bbuf, *AdirectionVector_bbuf, -alpha, BDIM);
    NDRHSB2 = normBBuffer_barrier_double(*discrepancy_bbuf);

    // Update discrepancy related to proximal part of the right hand side
    // AdirectionVector_bbuf_xpart_L2 = sqrt(effectsSize) * directionVector_xbuf
    algFLOATvector_A_equals_A_plus_cB(*discrepancy_bbuf_xpart_L2, *directionVector_xbuf,
                                      -alpha * sqrtEffectSize, XDIM);
    NDRHSP2 = normXBuffer_barrier_double(*discrepancy_bbuf_xpart_L2);
    NDRHS = std::sqrt(NDRHSB2 + NDRHSP2);

    while(NDRHS / NDRHS_ZEROX > errConditionCGLS && iteration < maxCGLSIterations)
    {
        if(reportKthIteration > 0 && iteration % reportKthIteration == 0)
        {
            LOGD << io::xprintf("Writing file %sx_it%02d.den", intermediatePrefix.c_str(),
                                iteration);
            BasePBCT2DReconstructor::writeVolume(
                *xbufOUT, io::xprintf("%sx_it%02d.den", intermediatePrefix.c_str(), iteration));
        }
        backproject(*discrepancy_bbuf, *residualVector_xbuf);
        NRHSB2 = normBBuffer_barrier_double(*residualVector_xbuf);
        NRHSP2 = effectSize * NDRHSP2;
        // I have to update the residual vector with proximal part
        // residualVector_xbuf_L2add = sqrtEffectSize * discrepancy_bbuf_xpart_L2
        algFLOATvector_A_equals_A_plus_cB(*residualVector_xbuf, *discrepancy_bbuf_xpart_L2,
                                          sqrtEffectSize, XDIM);
        NR2 = NRHSB2 + NRHSP2;
        residualNorm2_now = normXBuffer_barrier_double(*residualVector_xbuf);
        // Shall be OK to put residualNorm2_now = NR2 but for sure I will check it, remove in
        // production
        if(std::abs(residualNorm2_now - NR2) < 1e-6)
        {
            LOGE << io::xprintf(
                "Residual norm computation gave different results NR2=%e and residualNorm2_now=%e",
                NR2, residualNorm2_now);
            break;
        }
        reportTime(io::xprintf("Backprojection %d", iteration), false, true);
        // Delayed update of residual vector
        beta = residualNorm2_now / residualNorm2_old;
        if(proximalOperatorVerbose)
        {
            LOGI << io::xprintf_green(
                "\nIteration %d:\n|Ax-b|=%.2f, %.2f%% |b| and %.2f%% of |Ax0-b|,\n"
                "1/sqrt(2*tau)|x-x_prox|=%.2f, %.2f%% of 1/sqrt(2*tau)|x_prox| and %.2f%% of "
                "1/sqrt(2*tau)|x_0-x_prox|,\n"
                "|(Ax-b, 1/sqrt(2*tau)(x-x_prox))|=%.2f %.2f%% of x=0 expression and %.2f%% of "
                "x=x0 "
                "expression,\n"
                "|AT(Ax-b)|=%.2f %.2f%% of x=0 expression and %.2f%% of x=x0 expression,\n "
                "1/(2*tau)|x-x_prox|=%.2f, %.2f%% of x=0 expression and %.2f%% of x=x0 "
                "expression,\n"
                "|(AT(Ax-b)|+ 1/(2*tau)|x-x_prox)|=%.2f %.2f%% of x=0 expression and %.2f%% of "
                "x=x0 "
                "expression.",
                iteration, std::sqrt(NDRHSB2), 100.0 * std::sqrt(NDRHSB2 / NDRHSB2_ZEROX),
                100.0 * std::sqrt(NDRHSB2 / NDRHSB2_START), std::sqrt(NDRHSP2),
                100.0 * std::sqrt(NDRHSP2 / NDRHSP2_ZEROX),
                100.0 * std::sqrt(NDRHSP2 / NDRHSP2_START), NDRHS, 100.0 * NDRHS / NDRHS_ZEROX,
                100.0 * NDRHS / NDRHS_START, std::sqrt(NRHSB2),
                100.0 * std::sqrt(NRHSB2 / NRHSB2_ZEROX), 100.0 * std::sqrt(NRHSB2 / NRHSB2_START),
                std::sqrt(NRHSP2), 100.0 * std::sqrt(NRHSP2 / NRHSP2_ZEROX),
                100.0 * std::sqrt(NRHSP2 / NRHSP2_START), std::sqrt(NR2),
                100.0 * std::sqrt(NR2 / NR2_ZEROX), 100.0 * std::sqrt(NR2 / NR2_START));
        }
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
        algFLOATvector_A_equals_A_plus_cB(*xbufOUT, *directionVector_xbuf, alpha, XDIM);
        algFLOATvector_A_equals_A_plus_cB(*discrepancy_bbuf, *AdirectionVector_bbuf, -alpha, BDIM);
        NDRHSB2 = normBBuffer_barrier_double(*discrepancy_bbuf);
        // Update discrepancy related to proximal part of the right hand side
        algFLOATvector_A_equals_A_plus_cB(*discrepancy_bbuf_xpart_L2, *directionVector_xbuf,
                                          -alpha * sqrtEffectSize, XDIM);
        NDRHSP2 = normXBuffer_barrier_double(*discrepancy_bbuf_xpart_L2);
        NDRHS = std::sqrt(NDRHSB2 + NDRHSP2);
        if(proximalOperatorVerbose)
        {
            LOGD << io::xprintf(
                "Iteration %d: alpha = %f, beta = %f, |DISCREPANCY| = %f, |RESIDUALOLD| = %e",
                iteration, alpha, beta, NDRHS, std::sqrt(NR2));
        }
        //    BasePBCT2DReconstructor::writeVolume(*x_buf,
        //                                         io::xprintf("PDHG_x_xbufOUT_it%02d.den",
        //                                         iteration));
    }
    LOGD << io::xprintf(
        "Iteration %d: |DISCREPANCY| = %f %.2f%% of zero expression |Ax-b|=%f %.2f%% of |b|",
        iteration, NDRHS, 100.0 * NDRHS / NDRHS_ZEROX, std::sqrt(NDRHSB2),
        100.0 * std::sqrt(NDRHSB2 / NDRHSB2_START));
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
    double norm = 0.0, TVNorm = 0.0, DifferenceNorm = 0.0;
    NB0 = std::sqrt(normBBuffer_barrier_double(*b_buf));
    double hsquare = voxelSizesF.x * voxelSizesF.x;
    std::string str
        = io::xprintf("WELCOME TO Chambolle-Pock 2D, PDHGPBCT2DReconstructor, "
                      "mu=%0.2fh^2, tau=%0.2f, sigma=%0.2fh^2, theta=%0.2f ||b||_2=%0.2f",
                      mu / hsquare, tau, sigma / hsquare, theta, NB0);
    LOGD << io::xprintf_green("\n%s", printTime(str, false, true).c_str());

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
    // Just computing ||A^T(b)||_2 for convergence estimates
    backproject(*b_buf, *primal_xbuf); // A^T( b)
    NATB0 = std::sqrt(normXBuffer_barrier_double(*primal_xbuf));

    // Initial setup of primal variable (x)
    if(useVolumeAsInitialX0)
    {
        algFLOATvector_copy(*x_buf, *primal_xbuf, XDIM); // Start with x = x0
    } else
    {
        Q[0]->enqueueFillBuffer<cl_float>(*primal_xbuf, FLOATZERO, 0,
                                          XDIM * sizeof(float)); // Start with x = 0
        std::shared_ptr<cl::Buffer> primal_xbuf_new
            = divergence_xbuf; // Divergence is to be used as a temporary buffer since it is fully
                               // initialized in second step
        proximalOperatorCGLS(primal_xbuf, primal_xbuf, primal_xbuf_new, 0.0, maxCGLSIterations,
                             errConditionCGLS, 0);
        algFLOATvector_copy(*primal_xbuf_new, *primal_xbuf, XDIM);
    }
    // BasePBCT2DReconstructor::writeVolume(
    //     *primal_xbuf, io::xprintf("%sx0_it%02d.den", intermediatePrefix.c_str(), iteration));

    // Dual variable initialization
    Q[0]->enqueueFillBuffer<cl_float>(*dual_xbuf_x, FLOATZERO, 0,
                                      XDIM * sizeof(float)); // Dual variable = 0
    Q[0]->enqueueFillBuffer<cl_float>(*dual_xbuf_y, FLOATZERO, 0,
                                      XDIM * sizeof(float)); // Dual variable = 0

    // Initial discrepancy (Ax - b) and backprojection
    // project(*primal_xbuf, *discrepancy_bbuf);
    // backproject(*discrepancy_bbuf, *divergence_xbuf); // This will initialize the residual (x0)

    // Main Chambolle-Pock iteration loop
    // cl::NDRange globalRangeGradient(vdimx, vdimy, vdimz);
    // cl::NDRange localRangeGradient = cl::NullRange;
    primal_xbuf_prime = primal_xbuf; // At the beginnig they are the same
    bool useROF = false;
    bool L1ROF = false;
    bool computeDiscrepancy = false;
    if(!useROF)
    {
        computeDiscrepancy = true;
    }
    std::shared_ptr<cl::Buffer> rof_u0_xbuf;
    if(useROF)
    {
        rof_u0_xbuf = AdirectionVector_bbuf_xpart_L2;
        algFLOATvector_copy(*primal_xbuf, *rof_u0_xbuf, XDIM);
    } else
    {
        rof_u0_xbuf = nullptr;
    }
    while(iteration < maxPDHGIterations)
    {
        // INFO As primal_xbuf_dx and primal_xbuf_dy are computed I can report norms
        volume_gradient2D(*primal_xbuf, *primal_xbuf_dx, *primal_xbuf_dy);
        if(iteration == 1)
        {
            BasePBCT2DReconstructor::writeVolume(
                *primal_xbuf_dx,
                io::xprintf("%s_xbuf_x_it%02d.den", intermediatePrefix.c_str(), iteration));
            BasePBCT2DReconstructor::writeVolume(
                *primal_xbuf_dy,
                io::xprintf("%s_xbuf_y_it%02d.den", intermediatePrefix.c_str(), iteration));
        }
        std::array<double, 3> norms = computeSolutionNorms(
            primal_xbuf, primal_xbuf_dx, primal_xbuf_dy, rof_u0_xbuf, computeDiscrepancy);
        norm = norms[0];
        TVNorm = norms[1];
        DifferenceNorm = norms[2];
        // norm = std::sqrt(normBBuffer_barrier_double(*discrepancy_bbuf)); // Norm of Ax - b
        if(useROF)
        {
            LOGI << io::xprintf_green(
                "Iteration %d: |Ax-b|=%0.1f representing %0.2f%% of |b|, mu "
                "TV(x)=%0.2e, |x-x_0|=%0.2e, mu TV(x)/|x-x_0|=%f |x-x_0| + mu*TV(x)=%e",
                iteration - 1, norm, 100.0 * norm / NB0, mu * TVNorm, DifferenceNorm,
                mu * TVNorm / DifferenceNorm, mu * TVNorm + DifferenceNorm);
        } else
        {
            LOGI << io::xprintf_red(
                "\nIteration %d: |Ax-b|=%0.1f representing %0.2f%% of |b|, |Ax-b|_2^2=%0.2f mu "
                "TV(x)=%0.2f, mu TV(x)/|Ax-b|=%0.2f, |Ax-b| + mu TV(x)=%0.2f, |Ax-b|_2^2 + mu "
                "TV(x)=%0.2f",
                iteration - 1, norm, 100.0 * norm / NB0, norm * norm, mu * TVNorm,
                mu * TVNorm / norm, mu * TVNorm + norm, norm * norm + mu * TVNorm);
        }
        if(computeDiscrepancy)
        {
            if(norm / NB0 < errConditionPDHG)
            {
                break; // Exit if error condition is met
            }
        }
        // INFO
        // Step 1: Update Dual Variable (p^k+1 = Proj_Dual(p^k + sigma * Grad(x^k)))
        volume_gradient2D(*primal_xbuf_prime, *primal_xbuf_dx, *primal_xbuf_dy);

        // OLD implementation
        // algFLOATvector_2DisotropicGradient(
        //     *primal_xbuf_prime, *primal_xbuf_dx, *primal_xbuf_dy, vdims, voxelSizesF,
        //     globalRangeGradient,
        //     localRangeGradient); // Compute gradient of x (dual variable p update)
        //  p = p + sigma * Grad(x)
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
        volume_gradient2D_adjoint(*dual_xbuf_x, *dual_xbuf_y, *divergence_xbuf);
        // algFLOATvector_isotropicBackDivergence2D(*dual_xbuf_x, *dual_xbuf_y, *divergence_xbuf,
        //                                         vdims, voxelSizesF, globalRangeGradient,
        //                                         localRangeGradient); // Compute divergence of p
        // BasePBCT2DReconstructor::writeVolume(
        //    *divergence_xbuf,
        //    io::xprintf("%s_divergence_it%02d.den", intermediatePrefix.c_str(), iteration));
        // I need primal_xbuf for update in step 3 so I put argument for primal proximal operator to
        // dirergence_xbuf, which is fully computed in step 2
        std::shared_ptr<cl::Buffer> proximal_arg_xbuf = divergence_xbuf;
        algFLOATvector_A_equals_Ac_plus_B(*proximal_arg_xbuf, *primal_xbuf, -tau,
                                          XDIM); // arg = x - tau * (-Div)(p)

        // Step 2.1: apply proximal operator related to |Ax-b|_2^2
        // BasePBCT2DReconstructor::writeVolume(
        //     *proximal_arg_xbuf,
        //     io::xprintf("%s_proximal_arg_it%02d.den", intermediatePrefix.c_str(), iteration));
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
            if(!L1ROF)
            {
                algFLOATvector_C_equals_Ad_plus_Be(*rof_u0_xbuf, *proximal_arg_xbuf,
                                                   *primal_xbuf_new, rof_prefactor, arg_prefactor,
                                                   XDIM);
            } else
            {
                LOGE << io::xprintf("algFLOATvector_distL1ProxSoftThreasholding with tau=%f", tau);
                algFLOATvector_distL1ProxSoftThreasholding(*rof_u0_xbuf, *proximal_arg_xbuf, tau,
                                                           XDIM);
                algFLOATvector_copy(*proximal_arg_xbuf, *primal_xbuf_new,
                                    XDIM); // Start with x = x0
                /*
                std::shared_ptr<cl::Buffer> temp = primal_xbuf_new;
                primal_xbuf_new = proximal_arg_xbuf;
                primal_xbuf_dx = primal_xbuf_new;
                proximal_arg_xbuf = temp;
                divergence_xbuf = temp;
        */
            }
        } else
        {
            // proximalOperatorCGLS(proximal_arg_xbuf, proximal_arg_xbuf, primal_xbuf_new, tau,
            //                      maxCGLSIterations, errConditionCGLS, iteration);
            // I. Does not work, let's try to initiaize by 0
            // Q[0]->enqueueFillBuffer<cl_float>(*primal_xbuf_new, FLOATZERO, 0,
            //                                  XDIM * sizeof(float)); // Start with x = 0

            // proximalOperatorCGLS(proximal_arg_xbuf, primal_xbuf_new, primal_xbuf_new, tau,
            //                      maxCGLSIterations, errConditionCGLS, iteration);
            // II. Let's try to initialize by primal_xbuf
            //        proximalOperatorCGLS(proximal_arg_xbuf, primal_xbuf, primal_xbuf_new, tau,
            //                             maxCGLSIterations, errConditionCGLS, iteration);
            // III. experiment

            proximalOperatorVerbose = false;
            maxCGLSIterations = 3;
            proximalOperatorCGLS(proximal_arg_xbuf, primal_xbuf, primal_xbuf_new, tau,
                                 maxCGLSIterations, errConditionCGLS, iteration);
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

        if(iteration % 50 == 0)
        {
            BasePBCT2DReconstructor::writeVolume(
                *primal_xbuf, io::xprintf("%sx_it%02d.den", intermediatePrefix.c_str(), iteration));
        }
        //    BasePBCT2DReconstructor::writeVolume(
        //        *primal_xbuf_prime,
        //        io::xprintf("%sx_prime_it%02d.den", intermediatePrefix.c_str(), iteration));
        //  Projection and error condition check
        //  project(*primal_xbuf, *discrepancy_bbuf); // Forward projection of x
        //  algFLOATvector_A_equals_A_plus_cB(*discrepancy_bbuf, *b_buf, -1.0f, BDIM); // Ax - b

        iteration++;
    }
    // INFO As primal_xbuf_dx and primal_xbuf_dy are computed I can report norms
    volume_gradient2D(*primal_xbuf, *primal_xbuf_dx, *primal_xbuf_dy);
    std::array<double, 3> norms
        = computeSolutionNorms(primal_xbuf, primal_xbuf_dx, primal_xbuf_dy, rof_u0_xbuf);
    norm = norms[0];
    TVNorm = norms[1];
    DifferenceNorm = norms[2];
    // norm = std::sqrt(normBBuffer_barrier_double(*discrepancy_bbuf)); // Norm of Ax - b
    if(useROF)
    {
        LOGI << io::xprintf_green(
            "Finished iteration %d: |Ax-b|=%0.1f representing %0.2f%% of |b|, mu "
            "TV(x)=%0.2e, |x-x_0|=%0.2e, mu TV(x)/|x-x_0|=%f |x-x_0| + mu*TV(x)=%e",
            iteration, norm, 100.0 * norm / NB0, mu * TVNorm, DifferenceNorm,
            mu * TVNorm / DifferenceNorm, mu * TVNorm + DifferenceNorm);
    } else
    {
        LOGW << io::xprintf_green(
            "Finished iteration %d: |Ax-b|=%0.1f representing %0.2f%% of |b|, mu "
            "TV(x)=%0.2f, mu TV(x)/|Ax-b|=%f, |Ax-b| + mu TV(x)=%e",
            iteration, norm, 100.0 * norm / NB0, mu * TVNorm, mu * TVNorm / norm,
            mu * TVNorm + norm);
    }
    // INFO
    Q[0]->enqueueReadBuffer(*primal_xbuf, CL_TRUE, 0, sizeof(float) * XDIM, x);
    return 0;
}

} // namespace KCT
