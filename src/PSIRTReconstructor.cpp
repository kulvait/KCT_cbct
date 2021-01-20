#include "PSIRTReconstructor.hpp"

namespace CTL {

void PSIRTReconstructor::setup(double alpha) { this->alpha = alpha; }

int PSIRTReconstructor::reconstruct(uint32_t maxIterations, float errCondition)
{
    bool boxconditions = false;
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
    Q[0]->enqueueFillBuffer<cl_float>(*discrepancy_bbuf, FLOATONE, 0, BDIM * sizeof(float));
    project(*ones_xbuf, *rowsums_bbuf);
    // Ones used to store col sums
    backproject(*discrepancy_bbuf, *ones_xbuf);
    double A1norm = maxXBuffer_barier_float(*ones_xbuf);
    // double A2norm = std::sqrt(normBBuffer_barier_double(*rowsums_bbuf));
    // double p = 0.001 * BDIM / A1norm;
    double p = 1.0 / A1norm; // In 10.1109/TMI.2008.923696 authors probably mistakedly call the
                             // ||A||_1 what should be the largest col sum of A as they claim
                             // Since the largest row sum of AT in turn equals the largest column
                             // sum of A and thus the 1-norm of A. Tried here with
                             // ||A||_1 without success
    if(!useVolumeAsInitialX0)
    {
        Q[0]->enqueueFillBuffer<cl_float>(*x_buf, FLOATZERO, 0, XDIM * sizeof(float));
    }
    algFLOATvector_invert_except_zero(*rowsums_bbuf, BDIM);
    invrowsums_bbuf = rowsums_bbuf;
    double norm, normupdate, normx;
    double NB0 = std::sqrt(normBBuffer_barier_double(*b_buf));
    norm = NB0;
    LOGI << io::xprintf("||b||=%f, p=%f, A1norm=%f", NB0, p, A1norm);
    // LOGI << io::xprintf("||b||=%f, p=%f, A2norm=%f", NB0, p, A2norm);

    while(norm / NB0 > errCondition && iteration < maxIterations)
    {
        iteration++;
        project(*x_buf, *discrepancy_bbuf);
        algFLOATvector_A_equals_Ac_plus_B(*discrepancy_bbuf, *b_buf, -1.0, BDIM);
        norm = std::sqrt(normBBuffer_barier_double(*discrepancy_bbuf));
        LOGE << io::xprintf("Iteration %d, the norm of |Ax-b| is %f that is %0.2f%% of |b|.",
                            iteration, norm, 100.0 * norm / NB0);
        algFLOATvector_A_equals_A_times_B(*discrepancy_bbuf, *invrowsums_bbuf, BDIM);
        backproject(*discrepancy_bbuf, *update_xbuf);
        if(iteration == 1)
        {
            algFLOATvector_scale(*update_xbuf, p, XDIM);
        } else
        {
            algFLOATvector_scale(*update_xbuf, alpha * p, XDIM);
        }
        normupdate = std::sqrt(normXBuffer_barier_double(*update_xbuf));
        normx = std::sqrt(normXBuffer_barier_double(*x_buf));
        algFLOATvector_A_equals_A_plus_cB(*x_buf, *update_xbuf, 1.0, XDIM);
        if(boxconditions)
        {
            algFLOATvector_substitute_greater_than(*x_buf, 1.0, 1.0, XDIM);
            algFLOATvector_substitute_lower_than(*x_buf, 0.0, 0.0, XDIM);
        }
        LOGE << io::xprintf("Size of x=%f and size of the update=%f.", normx, normupdate);
        // algFLOATvector_A_equals_A_plus_cB(*x_buf, *update_xbuf, alpha * p, XDIM);
        // algFLOATvector_A_equals_A_plus_cB(*x_buf, *update_xbuf, alpha * p, XDIM);
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

int PSIRTReconstructor::reconstruct_sirt(uint32_t maxIterations, float errCondition)
{
    bool boxconditions = true;
    LOGD << printTime("WELCOME TO SIRT, init", false, true);
    uint32_t iteration = 0;

    // Initialization
    allocateBBuffers(2);
    allocateXBuffers(2);
    std::shared_ptr<cl::Buffer> rowsums_bbuf, invrowsums_bbuf, discrepancy_bbuf; // B buffers
    std::shared_ptr<cl::Buffer> invcolsums_xbuf, update_xbuf; // X buffers
    rowsums_bbuf = getBBuffer(0);
    discrepancy_bbuf = getBBuffer(1);
    invcolsums_xbuf = getXBuffer(0);
    update_xbuf = getXBuffer(1);
    Q[0]->enqueueFillBuffer<cl_float>(*invcolsums_xbuf, FLOATONE, 0, XDIM * sizeof(float));
    Q[0]->enqueueFillBuffer<cl_float>(*discrepancy_bbuf, FLOATONE, 0, BDIM * sizeof(float));
    project(*invcolsums_xbuf, *rowsums_bbuf);
    backproject(*discrepancy_bbuf, *invcolsums_xbuf);
    algFLOATvector_invert_except_zero(*rowsums_bbuf, BDIM);
    algFLOATvector_invert_except_zero(*invcolsums_xbuf, XDIM);
    invrowsums_bbuf = rowsums_bbuf;
    if(!useVolumeAsInitialX0)
    {
        Q[0]->enqueueFillBuffer<cl_float>(*x_buf, FLOATZERO, 0, XDIM * sizeof(float));
    }
    double norm, normupdate, normx;
    double NB0 = std::sqrt(normBBuffer_barier_double(*b_buf));
    norm = NB0;
    while(norm / NB0 > errCondition && iteration < maxIterations)
    {
        iteration++;
        project(*x_buf, *discrepancy_bbuf);
        algFLOATvector_A_equals_Ac_plus_B(*discrepancy_bbuf, *b_buf, -1.0, BDIM);
        norm = std::sqrt(normBBuffer_barier_double(*discrepancy_bbuf));
        LOGE << io::xprintf("Iteration %d, the norm of |Ax-b| is %f that is %0.2f%% of |b|.",
                            iteration, norm, 100.0 * norm / NB0);
        algFLOATvector_A_equals_A_times_B(*discrepancy_bbuf, *invrowsums_bbuf, BDIM);
        backproject(*discrepancy_bbuf, *update_xbuf);
        algFLOATvector_A_equals_A_times_B(*update_xbuf, *invcolsums_xbuf, XDIM);
        algFLOATvector_scale(*update_xbuf, alpha, XDIM);
        normupdate = std::sqrt(normXBuffer_barier_double(*update_xbuf));
        normx = std::sqrt(normXBuffer_barier_double(*x_buf));
        algFLOATvector_A_equals_A_plus_cB(*x_buf, *update_xbuf, 1.0, XDIM);
        if(boxconditions)
        {
            algFLOATvector_substitute_greater_than(*x_buf, 1.0, 1.0, XDIM);
            algFLOATvector_substitute_lower_than(*x_buf, 0.0, 0.0, XDIM);
        }
        LOGE << io::xprintf("Size of x=%f and size of the update=%f.", normx, normupdate);
        // algFLOATvector_A_equals_A_plus_cB(*x_buf, *update_xbuf, alpha * p, XDIM);
        // algFLOATvector_A_equals_A_plus_cB(*x_buf, *update_xbuf, alpha * p, XDIM);
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
