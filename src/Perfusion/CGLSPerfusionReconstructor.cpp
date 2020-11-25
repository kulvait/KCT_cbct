#include "Perfusion/CGLSPerfusionReconstructor.hpp"

namespace CTL {

int CGLSPerfusionReconstructor::reconstruct(uint32_t maxIterations,
                                            float errCondition,
                                            bool blocking)
{
    reportTime("WELCOME TO CGLS, init perfusion", blocking, true);
    uint32_t iteration = 1;

    // Initialization
    double norm, residualNorm2_old, residualNorm2_now, AdirectionNorm2, alpha, beta;
    double NB0 = std::sqrt(normBBuffer_barier_double(b_buf));
    double NR0, NX;
    LOGI << io::xprintf("||b||=%f", NB0);
    std::vector<std::shared_ptr<cl::Buffer>> directionVector_xbuf, residualVector_xbuf; // X buffers
    allocateXBuffers(2);
    directionVector_xbuf = getXBuffers(0);
    residualVector_xbuf = getXBuffers(1);
    allocateBBuffers(2);
    std::vector<std::shared_ptr<cl::Buffer>> discrepancy_bbuf, AdirectionVector_bbuf; // B buffers
    discrepancy_bbuf = getBBuffers(0);
    AdirectionVector_bbuf = getBBuffers(1);

    // INITIALIZATION x_0 is initialized typically by zeros but in general by supplied array
    // c_0 is filled by b
    // v_0=w_0=BACKPROJECT(c_0)
    // writeProjections(*discrepancy_bbuf, io::xprintf("/tmp/cgls/c_0.den"));
    copyFloatVector(b_buf, discrepancy_bbuf, BDIM); // discrepancy_bbuf stores initial discrepancy
    if(useVolumeAsInitialX0)
    {
        setTimestamp(blocking);
        project(x_buf, AdirectionVector_bbuf);
        reportTime("Projection x0", blocking, true);
        addIntoFirstVectorSecondVectorScaled(discrepancy_bbuf, AdirectionVector_bbuf, -1.0, BDIM);
    } else
    {
        zeroXBuffers(x_buf);
    }
    setTimestamp(blocking);
    backproject(discrepancy_bbuf, residualVector_xbuf);
    reportTime("Backprojection 0", blocking, true);
    copyFloatVector(residualVector_xbuf, directionVector_xbuf, XDIM);
    residualNorm2_old = normXBuffer_barier_double(residualVector_xbuf);
    NR0 = std::sqrt(residualNorm2_old);
    setTimestamp(blocking);
    project(directionVector_xbuf, AdirectionVector_bbuf);
    reportTime("Projection 1", blocking, true);
    AdirectionNorm2 = normBBuffer_barier_double(AdirectionVector_bbuf);
    alpha = residualNorm2_old / AdirectionNorm2;
    addIntoFirstVectorSecondVectorScaled(x_buf, directionVector_xbuf, alpha, XDIM);
    addIntoFirstVectorSecondVectorScaled(discrepancy_bbuf, AdirectionVector_bbuf, -alpha, BDIM);
    norm = std::sqrt(normBBuffer_barier_double(discrepancy_bbuf));
    while(norm / NB0 > errCondition && iteration < maxIterations)
    {
        if(reportKthIteration > 0 && iteration % reportKthIteration == 0)
        {
            LOGD << io::xprintf("Writing file %sx_it%02d.den", progressPrefixPath.c_str(),
                                iteration);
            writeVolume(x_buf,
                        io::xprintf("%sx_it%02d.den", progressPrefixPath.c_str(), iteration));
        }
        // DEBUG
        if(iteration % 10 == 0)
        {
            setTimestamp(blocking);
            project(x_buf, discrepancy_bbuf);
            reportTime(io::xprintf("Reothrogonalization projection %d", iteration), blocking, true);
            addIntoFirstVectorScaledSecondVector(discrepancy_bbuf, b_buf, -1.0, BDIM);
            double norm2 = std::sqrt(normBBuffer_barier_double(discrepancy_bbuf));

            LOGE << io::xprintf(
                "Iteration %d, the norm of |Ax-b| is %f that is %0.2f%% of |b|, norms "
                "loss of orthogonality %f%%.",
                iteration, norm2, 100.0 * norm2 / NB0, 100 * (norm2 - norm) / norm);
        }
        // DEBUG
        setTimestamp(blocking);
        backproject(discrepancy_bbuf, residualVector_xbuf);
        reportTime(io::xprintf("Backprojection %d", iteration), blocking, true);
        residualNorm2_now = normXBuffer_barier_double(residualVector_xbuf);
        // Delayed update of residual vector
        beta = residualNorm2_now / residualNorm2_old;
        NX = std::sqrt(residualNorm2_now);
        LOGE << io::xprintf("Iteration %d: |Ax-b|=%0.1f that is %0.2f%% of |b|, |AT(Ax-b)|=%0.2f "
                            "that is %0.3f%% of |AT(Ax0-b)|.",
                            iteration, norm, 100.0 * norm / NB0, NX, 100 * NX / NR0);
        addIntoFirstVectorScaledSecondVector(directionVector_xbuf, residualVector_xbuf, beta, XDIM);
        // Delayed update of direction vector
        iteration = iteration + 1;
        residualNorm2_old = residualNorm2_now;
        setTimestamp(blocking);
        project(directionVector_xbuf, AdirectionVector_bbuf);
        reportTime(io::xprintf("Projection %d", iteration), blocking, true);
        AdirectionNorm2 = normBBuffer_barier_double(AdirectionVector_bbuf);
        alpha = residualNorm2_old / AdirectionNorm2;
        addIntoFirstVectorSecondVectorScaled(x_buf, directionVector_xbuf, alpha, XDIM);
        addIntoFirstVectorSecondVectorScaled(discrepancy_bbuf, AdirectionVector_bbuf, -alpha, BDIM);
        norm = std::sqrt(normBBuffer_barier_double(discrepancy_bbuf));
    }
    LOGE << io::xprintf("Iteration %d, the norm of |Ax-b| is %f that is %0.2f%% of |b|.", iteration,
                        norm, 100.0 * norm / NB0);
    for(uint32_t vectorID = 0; vectorID != XVNUM; vectorID++)
    {
        Q[0]->enqueueReadBuffer(*x_buf[vectorID], CL_TRUE, 0, sizeof(float) * XDIM, x[vectorID]);
    }
    return 0;
}

} // namespace CTL
