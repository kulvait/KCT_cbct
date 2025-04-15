#include "PDHGROFExecutor.hpp"

namespace KCT {

std::array<double, 2> PDHGROFExecutor::computeSolutionNorms(std::shared_ptr<cl::Buffer> x_vector,
                                                            std::shared_ptr<cl::Buffer> x_vector_dx,
                                                            std::shared_ptr<cl::Buffer> x_vector_dy,
                                                            std::shared_ptr<cl::Buffer> x_0)
{
    double TVNorm = isotropicTVNormXBuffer_barrier_double(*x_vector_dx, *x_vector_dy);
    double DifferenceNorm = 0.0;
    if(x_0 != nullptr)
    {
        std::shared_ptr<cl::Buffer> differenceVector_xbuf = discrepancy_bbuf_xpart_L2;
        algFLOATvector_C_equals_Ad_plus_Be(*x_vector, *x_0, *differenceVector_xbuf, -1.0f, 1.0f, XDIM);
        DifferenceNorm = normXBuffer_barrier_double(*differenceVector_xbuf);
    }
    return { TVNorm, DifferenceNorm };
}

int PDHGROFExecutor::reconstruct(float mu, float tau, float sigma, float theta, uint32_t maxPDHGIterations, float errConditionPDHG)
{
    uint32_t iteration = 1;
    // Initialization of primal and dual variables
    double TVNorm = 0.0, DifferenceNorm = 0.0;
    double hsquare = voxelSizesF.x * voxelSizesF.x;

    std::shared_ptr<cl::Buffer> primal_xbuf, primal_xbuf_prime, primal_xbuf_dx, primal_xbuf_dy, dual_xbuf_x, dual_xbuf_y, divergence_xbuf;
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

    // Just computing ||x||_2 for convergence estimates
    algFLOATvector_copy(*x_buf, *primal_xbuf, XDIM); // Start with x = x0
    double NX0 = std::sqrt(normXBuffer_barrier_double(*primal_xbuf));
    std::string str = io::xprintf("WELCOME TO ROF Chambolle-Pock 2D, PDHGROFExecutor, "
                                  "mu=%0.2fh^2, tau=%0.2f, sigma=%0.2fh^2, theta=%0.2f, |x|=%0.2f",
                                  mu / hsquare, tau, sigma / hsquare, theta, NX0);
    LOGD << io::xprintf_green("\n%s", printTime(str, false, true).c_str());

    // BaseROFOperator::writeVolume(
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
    bool L1ROF = false;
    std::shared_ptr<cl::Buffer> rof_u0_xbuf;
    rof_u0_xbuf = AdirectionVector_bbuf_xpart_L2;
    algFLOATvector_copy(*primal_xbuf, *rof_u0_xbuf, XDIM);
    while(iteration < maxPDHGIterations)
    {
        // INFO As primal_xbuf_dx and primal_xbuf_dy are computed I can report norms
        volume_gradient2D(*primal_xbuf, *primal_xbuf_dx, *primal_xbuf_dy);
        if(iteration == 1 && proximalOperatorVerbose)
        {
            BaseROFOperator::writeVolume(*primal_xbuf_dx, io::xprintf("%s_xbuf_x_it%02d.den", intermediatePrefix.c_str(), iteration));
            BaseROFOperator::writeVolume(*primal_xbuf_dy, io::xprintf("%s_xbuf_y_it%02d.den", intermediatePrefix.c_str(), iteration));
        }
        std::array<double, 2> norms = computeSolutionNorms(primal_xbuf, primal_xbuf_dx, primal_xbuf_dy, rof_u0_xbuf);
        TVNorm = norms[0];
        DifferenceNorm = norms[1];
        LOGI << io::xprintf_green("Iteration %d: mu TV(x)=%0.2e, |x-x_0|=%0.2e, mu "
                                  "TV(x)/|x-x_0|=%f |x-x_0| + mu*TV(x)=%e",
                                  iteration - 1, mu * TVNorm, DifferenceNorm, mu * TVNorm / DifferenceNorm, mu * TVNorm + DifferenceNorm);
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
                BaseROFOperator::writeVolume(
                    *dual_xbuf_x,
                    io::xprintf("%s_dual_xbuf_x_bef_it%02d.den", intermediatePrefix.c_str(),
           iteration));*/
        algFLOATvector_infProjectionToLambda2DBall(*dual_xbuf_x, *dual_xbuf_y, mu, XDIM);
        /*        BaseROFOperator::writeVolume(
                    *dual_xbuf_x,
                    io::xprintf("%s_dual_xbuf_x_aft_it%02d.den", intermediatePrefix.c_str(),
           iteration));*/

        // Step 2: Update Primal Variable (x^k+1 = Prox_F(x^k - tau * Div(p^k+1)))
        volume_gradient2D_adjoint(*dual_xbuf_x, *dual_xbuf_y, *divergence_xbuf);
        // algFLOATvector_isotropicBackDivergence2D(*dual_xbuf_x, *dual_xbuf_y, *divergence_xbuf,
        //                                         vdims, voxelSizesF, globalRangeGradient,
        //                                         localRangeGradient); // Compute divergence of p
        // BaseROFOperator::writeVolume(
        //    *divergence_xbuf,
        //    io::xprintf("%s_divergence_it%02d.den", intermediatePrefix.c_str(), iteration));
        // I need primal_xbuf for update in step 3 so I put argument for primal proximal operator to
        // dirergence_xbuf, which is fully computed in step 2
        std::shared_ptr<cl::Buffer> proximal_arg_xbuf = divergence_xbuf;
        algFLOATvector_A_equals_Ac_plus_B(*proximal_arg_xbuf, *primal_xbuf, -tau,
                                          XDIM); // arg = x - tau * (-Div)(p)

        // Step 2.1: apply proximal operator related to |Ax-b|_2^2
        // BaseROFOperator::writeVolume(
        //     *proximal_arg_xbuf,
        //     io::xprintf("%s_proximal_arg_it%02d.den", intermediatePrefix.c_str(), iteration));
        std::shared_ptr<cl::Buffer> primal_xbuf_new = primal_xbuf_dx; // primal_xbuf_dx is to be used as a temporary buffer since it is
                                                                      // fully initialized in the first step
        // ROF proximal operator (1/(2*tau)) * ||u - rof_u0||_2^2 +  ||u -
        // proximal_arg_xbuf||_2^2 primal_xbuf_new = (1/(2*tau))/(1/(2*tau) + 1) * rof_u0 +
        // (1/(1/(2*tau) + 1)) * proximal_arg_xbuf
        float rof_prefactor = 1.0f / (1.0f / (2.0f * tau) + 1.0f);
        float arg_prefactor = 1.0f - rof_prefactor;
        LOGD << io::xprintf("ROF proximal operator: arg_prefactor=%f, rof_prefactor=%f", arg_prefactor, rof_prefactor);
        if(!L1ROF)
        {
            algFLOATvector_C_equals_Ad_plus_Be(*rof_u0_xbuf, *proximal_arg_xbuf, *primal_xbuf_new, rof_prefactor, arg_prefactor, XDIM);
        } else
        {
            LOGE << io::xprintf("algFLOATvector_distL1ProxSoftThreasholding with tau=%f", tau);
            algFLOATvector_distL1ProxSoftThreasholding(*rof_u0_xbuf, *proximal_arg_xbuf, tau, XDIM);
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

        // Step 3: Over-relaxation (x̄ = x^k+1 + θ(x^k+1 - x^k))
        if(theta > 0.0f)
        {
            primal_xbuf_prime = residualVector_xbuf; // It is not used right now
            algFLOATvector_C_equals_Ad_plus_Be(*primal_xbuf, *primal_xbuf_new, *primal_xbuf_prime, -theta, 1.0f + theta, XDIM);
        } else
        {
            primal_xbuf_prime = primal_xbuf_new; // They are the same
        }

        // Swap buffers
        // hack to avoid copying buffers
        std::shared_ptr<cl::Buffer> temp = primal_xbuf;
        primal_xbuf = primal_xbuf_new;
        primal_xbuf_dx = temp;

        if(iteration % 50 == 0 && proximalOperatorVerbose)
        {
            BaseROFOperator::writeVolume(*primal_xbuf, io::xprintf("%sx_it%02d.den", intermediatePrefix.c_str(), iteration));
        }
        //    BaseROFOperator::writeVolume(
        //        *primal_xbuf_prime,
        //        io::xprintf("%sx_prime_it%02d.den", intermediatePrefix.c_str(), iteration));
        //  Projection and error condition check
        //  project(*primal_xbuf, *discrepancy_bbuf); // Forward projection of x
        //  algFLOATvector_A_equals_A_plus_cB(*discrepancy_bbuf, *b_buf, -1.0f, BDIM); // Ax - b

        iteration++;
    }
    // INFO As primal_xbuf_dx and primal_xbuf_dy are computed I can report norms
    volume_gradient2D(*primal_xbuf, *primal_xbuf_dx, *primal_xbuf_dy);
    std::array<double, 2> norms = computeSolutionNorms(primal_xbuf, primal_xbuf_dx, primal_xbuf_dy, rof_u0_xbuf);
    TVNorm = norms[0];
    DifferenceNorm = norms[1];
    // norm = std::sqrt(normBBuffer_barrier_double(*discrepancy_bbuf)); // Norm of Ax - b
    LOGI << io::xprintf_green("Finished iteration %d: "
                              "TV(x)=%0.2e, |x-x_0|=%0.2e, mu TV(x)/|x-x_0|=%f |x-x_0| + mu*TV(x)=%e",
                              iteration, mu * TVNorm, DifferenceNorm, mu * TVNorm / DifferenceNorm, mu * TVNorm + DifferenceNorm);
    // INFO
    Q[0]->enqueueReadBuffer(*primal_xbuf, CL_TRUE, 0, sizeof(float) * XDIM, x);
    return 0;
}

} // namespace KCT
