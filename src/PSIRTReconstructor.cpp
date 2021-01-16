#include "PSIRTReconstructor.hpp"

namespace CTL {

void PSIRTReconstructor::setup(double alpha) { this->alpha = alpha; }

int PSIRTReconstructor::reconstruct(uint32_t maxIterations, float errCondition)
{
    LOGD << printTime("WELCOME TO PSIRT, init", false, true);
    uint32_t iteration = 0;

    // Initialization
    allocateBBuffers(2);
    allocateXBuffers(2);
    std::shared_ptr<cl::Buffer> rowsums_bbuf, invrowsums_bbuf, discrepancy_bbuf; // B buffers
    std::shared_ptr<cl::Buffer> ones_xbuf, update_xbuf; // X buffers
    rowsums_bbuf = getBBuffer(0);
    discrepancy_bbuf = getBBuffer(1);
    ones_xbuf = getXBuffer(0);
    update_xbuf = getXBuffer(1);
    Q[0]->enqueueFillBuffer<cl_float>(*ones_xbuf, FLOATONE, 0, XDIM * sizeof(float));
    project(*ones_xbuf, *rowsums_bbuf);
    double A1norm = sumBBuffer_barier_double(*rowsums_bbuf);
    double p = 1.0 / A1norm;
    if(!useVolumeAsInitialX0)
    {
        Q[0]->enqueueFillBuffer<cl_float>(*x_buf, FLOATZERO, 0, XDIM * sizeof(float));
    }
    algFLOATvector_invert_except_zero(*rowsums_bbuf, XDIM);
    invrowsums_bbuf = rowsums_bbuf;

    double norm;
    double NB0 = std::sqrt(normBBuffer_barier_double(*b_buf));
    norm = NB0;
    LOGI << io::xprintf("||b||=%f", NB0);
    while(norm / NB0 > errCondition && iteration < maxIterations)
    {
	iteration++;
        project(*x_buf, *discrepancy_bbuf);
        algFLOATvector_A_equals_Ac_plus_B(*discrepancy_bbuf, *b_buf, -1.0, BDIM);
        norm = std::sqrt(normBBuffer_barier_double(*discrepancy_bbuf));
        algFLOATvector_A_equals_A_times_B(*discrepancy_bbuf, *invrowsums_bbuf, BDIM);
        backproject(*discrepancy_bbuf, *update_xbuf);
        algFLOATvector_A_equals_A_plus_cB(*x_buf, *update_xbuf, alpha*p, XDIM);
        if(reportKthIteration > 0 && iteration % reportKthIteration == 0)
        {
            LOGD << io::xprintf("Writing file %sx_it%02d.den", progressPrefixPath.c_str(),
                                iteration);
            writeVolume(*x_buf,
                        io::xprintf("%sx_it%02d.den", progressPrefixPath.c_str(), iteration));
        }
    }
    LOGE << io::xprintf("Iteration %d, the norm of |Ax-b| is %f that is %0.2f%% of |b|.", iteration,
                        norm, 100.0 * norm / NB0);
    Q[0]->enqueueReadBuffer(*x_buf, CL_TRUE, 0, sizeof(float) * XDIM, x);
    return 0;
}
} // namespace CTL
