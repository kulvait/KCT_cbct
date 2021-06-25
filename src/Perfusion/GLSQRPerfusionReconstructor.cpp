#include "Perfusion/GLSQRPerfusionReconstructor.hpp"

namespace CTL {

int GLSQRPerfusionReconstructor::reconstruct(uint32_t maxIterations,
                                             float errCondition,
                                             bool blocking)
{
    reportTime("WELCOME TO GLSQR, init perfusion", false, true);
    uint32_t iteration = 0;

    // Initialization
    double NB0 = std::sqrt(normBBuffer_barrier_double(b_buf));
    LOGI << io::xprintf("||b||=%f", NB0);
    std::vector<std::shared_ptr<cl::Buffer>> u_prev, u_cur, u_next;
    std::vector<std::shared_ptr<cl::Buffer>> v_prev, v_cur, v_next;
    std::vector<std::shared_ptr<cl::Buffer>> w_prev_prev, w_prev, w_cur;
    std::vector<std::shared_ptr<cl::Buffer>> x_cur;
    std::vector<std::shared_ptr<cl::Buffer>> XZ, BZ;
    allocateXBuffers(4);
    allocateTmpXBuffers(1);
    allocateBBuffers(2);
    allocateTmpBBuffers(1);

    // Anything might be supplied here, but we will do standard initialization first
    v_next = getXBuffers(0);
    zeroXBuffers(v_next);
    backproject(b_buf, v_next);
    //    LOGD << io::xprintf("Writing file v_init.den");
    //    writeVolume(*v_next, io::xprintf("v_init.den"));
    double vnextnorm = std::sqrt(normXBuffer_barrier_double(v_next));
    LOGI << io::xprintf("vnextnorm=%f", vnextnorm);
    scaleFloatVector(v_next, float(1.0 / vnextnorm), XDIM);
    bool initializedByScaledBackprojectedRightSide = true;

    double d = 0.0;

    u_cur = getBBuffers(0);
    zeroBBuffers(u_cur);

    v_cur = getXBuffers(1);
    zeroXBuffers(v_cur);

    double varphi_hat = NB0;

    u_next = getBBuffers(1);
    zeroBBuffers(u_next);
    addIntoFirstVectorSecondVectorScaled(u_next, b_buf, float(1.0 / varphi_hat), BDIM);

    x_cur = x_buf;
    zeroXBuffers(x_cur);

    w_cur = getXBuffers(2);
    zeroXBuffers(w_cur);

    w_prev = getXBuffers(3);
    zeroXBuffers(w_prev);

    double rho_cur = 1.0;
    double rho_prev = 1.0;
    double c_cur = -1.0;
    double c_prev = -1.0;
    double s_cur = 0.0;
    double s_prev = 0.0;

    XZ = getTmpXBuffers(0);
    BZ = getTmpBBuffers(0);
    double c_prev_prev;
    double s_prev_prev;
    double rho_prev_prev;
    double sigma_prev;
    double sigma_cur;
    double sigma_next;
    double sigma_tol = 0.001; // Based on the fact that numerical error is on the level ~0.0002
    double tau_cur, tau_prev, tau_next;
    double gamma;
    double varphi;
    double theta;
    double rho_hat;

    while(std::abs(varphi_hat) / NB0 > errCondition && iteration < maxIterations)
    {
        // Iteration
        iteration = iteration + 1;

        u_prev = u_cur;
        u_cur = u_next;
        // u_next = u_prev;  will be issued after u_prev is not used

        v_prev = v_cur;
        v_cur = v_next;
        // v_next=v_prev will be issued after v_prev is not used

        c_prev_prev = c_prev;
        c_prev = c_cur;

        s_prev_prev = s_prev;
        s_prev = s_cur;

        w_prev_prev = w_prev;
        w_prev = w_cur;
        // w_cur=w_prev_prev will be issued after it is possible

        rho_prev_prev = rho_prev;
        rho_prev = rho_cur;

        backproject(u_cur, XZ);
        sigma_prev = scalarProductXBuffer_barrier_double(XZ, v_prev);
        addIntoFirstVectorSecondVectorScaled(XZ, v_prev, float(-sigma_prev), XDIM);
        v_next = v_prev;
        LOGI << io::xprintf("sigma_prev=%f", sigma_prev);

        if(d == 0.0)
        {
            LOGI << "d=0.0";
            sigma_cur = scalarProductXBuffer_barrier_double(XZ, v_cur);
            addIntoFirstVectorSecondVectorScaled(XZ, v_cur, float(-sigma_cur), XDIM);

            sigma_next = std::sqrt(normXBuffer_barrier_double(XZ));
            LOGI << io::xprintf("sigma_next=%f", sigma_next);

            if(initializedByScaledBackprojectedRightSide)
            {
                LOGI << io::xprintf("Size of numerical error sigma_next=%f", sigma_next);
                sigma_next = 0;
            }

            if(sigma_next > sigma_tol)
            {
                algFLOATvector_A_equals_cB(v_next, XZ, float(1.0f / sigma_next), XDIM);
            } else
            {
                d = 1.0;
            }
        } else
        {
            LOGI << "d=1.0";
            sigma_cur = std::sqrt(normXBuffer_barrier_double(XZ));
            LOGI << io::xprintf("sigma_cur=%f", sigma_cur);

            if(sigma_cur > sigma_tol)
            {
                algFLOATvector_A_equals_cB(v_cur, XZ, float(1.0f / sigma_cur), XDIM);
            } else
            {
                LOGI << "Ending due to the convergence";
                break;
            }
        }

        project(v_cur, BZ);
        tau_prev = scalarProductBBuffer_barrier_double(BZ, u_prev);
        addIntoFirstVectorSecondVectorScaled(BZ, u_prev, float(-tau_prev), BDIM);
        u_next = u_prev;

        tau_cur = scalarProductBBuffer_barrier_double(BZ, u_cur);
        addIntoFirstVectorSecondVectorScaled(BZ, u_cur, float(-tau_cur), BDIM);
        tau_next = std::sqrt(normBBuffer_barrier_double(BZ));
        LOGE << io::xprintf("tau_prev=%f, tau_cur=%f, tau_next=%f", tau_prev, tau_cur, tau_next);

        if(tau_next != 0)
        {
            algFLOATvector_A_equals_cB(u_next, BZ, float(1.0f / tau_next), BDIM);
        }

        gamma = s_prev_prev * tau_prev;
        theta = -c_prev * c_prev_prev * tau_prev + s_prev * tau_cur;
        rho_hat = -s_prev * c_prev_prev * tau_prev - c_prev * tau_cur;
        LOGE << io::xprintf("gamma=%f, theta=%f, rho_hat=%f", gamma, theta, rho_hat);

        rho_cur = std::sqrt(rho_hat * rho_hat + tau_next * tau_next);
        c_cur = rho_hat / rho_cur;
        s_cur = tau_next / rho_cur;
        LOGE << io::xprintf("rho_cur=%f, s_cur=%f, c_cur=%f", rho_cur, s_cur, c_cur);
        // 24
        varphi = c_cur * varphi_hat;
        varphi_hat = s_cur * varphi_hat;
        // 25
        w_cur = w_prev_prev;
        addIntoFirstVectorScaledSecondVector(w_cur, v_cur, float(-gamma / rho_prev_prev), XDIM);
        addIntoFirstVectorSecondVectorScaled(w_cur, w_prev, float(-theta / rho_prev), XDIM);

        // 26
        addIntoFirstVectorSecondVectorScaled(x_cur, w_cur, float(varphi / rho_cur), XDIM);
        if(tau_next == 0)
        {
            LOGI << "Ending due to the convergence";
            break;
        }

        LOGW << io::xprintf("After iteration %d, the norm of |Ax-b| is %f that is %0.2f%% of NB0.",
                            iteration, std::abs(varphi_hat), 100.0 * std::abs(varphi_hat) / NB0);
        if(reportKthIteration > 0 && iteration % reportKthIteration == 0)
        {
            LOGD << io::xprintf("Writing file %sx_it%02d.den", progressPrefixPath.c_str(),
                                iteration);
            writeVolume(x_cur,
                        io::xprintf("%sx_it%02d.den", progressPrefixPath.c_str(), iteration));
        }
    }
    for(std::uint32_t vectorID = 0; vectorID != XVNUM; vectorID++)
    {
        Q[0]->enqueueReadBuffer(*x_cur[vectorID], CL_TRUE, 0, sizeof(float) * XDIM, x[vectorID]);
    }
    return 0;
}

} // namespace CTL
