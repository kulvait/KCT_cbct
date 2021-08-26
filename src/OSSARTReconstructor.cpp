#include "OSSARTReconstructor.hpp"

namespace KCT {

void OSSARTReconstructor::setup(float relaxationParameter, uint32_t subsetCount)
{
    this->relaxationParameter = relaxationParameter;
    this->subsetCount = subsetCount;
}

void OSSARTReconstructor::addUpperBoxCondition(float upperBound, float upperBoundSubstitution)
{
    this->upperBoxCondition = true;
    this->upperBound = upperBound;
    this->upperBoundSubstitution = upperBoundSubstitution;
}

void OSSARTReconstructor::removeUpperBoxCondition() { this->upperBoxCondition = false; }

void OSSARTReconstructor::addLowerBoxCondition(float lowerBound, float lowerBoxSubstitution)
{
    this->lowerBoxCondition = true;
    this->lowerBound = lowerBound;
    this->lowerBoundSubstitution = lowerBoundSubstitution;
}

void OSSARTReconstructor::removeLowerBoxCondition() { this->lowerBoxCondition = false; }

int OSSARTReconstructor::reconstruct(uint32_t maxIterations, float errCondition)
{
    std::string welcomeString
        = io::xprintf("WELCOME TO OS SART with %d subsets and relaxation parameter %f.",
                      subsetCount, relaxationParameter);
    LOGD << printTime(welcomeString, false, true);
    uint32_t iteration = 0;

    // Initialization
    allocateBBuffers(3);
    allocateXBuffers(2);
    // First obtain poit-wise inverse of row sum
    std::shared_ptr<cl::Buffer> invrowsum_bbuf;
    std::shared_ptr<cl::Buffer> discrepancy_bbuf, partial_discrepancy_bbuf;
    std::shared_ptr<cl::Buffer> ones_bbuf;
    std::shared_ptr<cl::Buffer> ones_xbuf, update_xbuf;
    std::shared_ptr<cl::Buffer> partial_invcolsum_xbuf;
    invrowsum_bbuf = getBBuffer(0);
    ones_xbuf = getXBuffer(0);
    Q[0]->enqueueFillBuffer<cl_float>(*ones_xbuf, FLOATONE, 0, XDIM * sizeof(float));
    project(*ones_xbuf, *invrowsum_bbuf);
    update_xbuf = ones_xbuf;
    ones_xbuf = nullptr;
    algFLOATvector_invert_except_zero(*invrowsum_bbuf, BDIM);
    double norm, NB0 = std::sqrt(normBBuffer_barrier_double(*b_buf));
    // These will share single buffer
    discrepancy_bbuf = getBBuffer(1);
    partial_discrepancy_bbuf = getBBuffer(1);
    if(useVolumeAsInitialX0)
    {
        project(*x_buf, *discrepancy_bbuf);
        algFLOATvector_A_equals_Ac_plus_B(*discrepancy_bbuf, *b_buf, -1.0, BDIM);
        norm = std::sqrt(normBBuffer_barrier_double(*discrepancy_bbuf));
    } else
    {
        Q[0]->enqueueFillBuffer<cl_float>(*x_buf, FLOATZERO, 0, XDIM * sizeof(float));
        norm = NB0;
    }
    LOGI << io::xprintf_green("|Ax-b|=%0.1f that is %0.2f%% of |b|.", norm, 100.0 * norm / NB0);
    ones_bbuf = getBBuffer(2);
    Q[0]->enqueueFillBuffer<cl_float>(*ones_bbuf, FLOATONE, 0, BDIM * sizeof(float));
    partial_invcolsum_xbuf = getXBuffer(1);
    // x_buf now stores x0, discrepancy_bbuf initial discrepancy

    while(norm / NB0 > errCondition && iteration < maxIterations)
    {
        for(uint32_t subsetIndex = 0; subsetIndex != subsetCount; subsetIndex++)
        {
            backproject(*ones_bbuf, *partial_invcolsum_xbuf, subsetIndex, subsetCount);
            algFLOATvector_invert_except_zero(*partial_invcolsum_xbuf, XDIM);
            backproject(*partial_discrepancy_bbuf, *update_xbuf, subsetIndex, subsetCount);
            project(*x_buf, *partial_discrepancy_bbuf, subsetIndex, subsetCount);
            algFLOATvector_A_equals_Ac_plus_B(*partial_discrepancy_bbuf, *b_buf, -1.0, BDIM);
            algFLOATvector_A_equals_A_times_B(*partial_discrepancy_bbuf, *invrowsum_bbuf, BDIM);
            backproject(*partial_discrepancy_bbuf, *update_xbuf, subsetIndex, subsetCount);
            algFLOATvector_A_equals_A_times_B(*update_xbuf, *partial_invcolsum_xbuf, XDIM);
            algFLOATvector_A_equals_A_plus_cB(*x_buf, *update_xbuf, relaxationParameter, XDIM);
        }
        if(upperBoxCondition)
        {
            algFLOATvector_substitute_greater_than(*x_buf, upperBound, upperBoundSubstitution,
                                                   XDIM);
        }
        if(lowerBoxCondition)
        {
            algFLOATvector_substitute_lower_than(*x_buf, lowerBound, lowerBoundSubstitution, XDIM);
        }
        iteration++;
        reportTime(io::xprintf("Iteration %d", iteration), false, true);
        project(*x_buf, *discrepancy_bbuf);
        algFLOATvector_A_equals_Ac_plus_B(*discrepancy_bbuf, *b_buf, -1.0, BDIM);
        norm = std::sqrt(normBBuffer_barrier_double(*discrepancy_bbuf));
        LOGI << io::xprintf_green("Iteration %d, the norm of |Ax-b| is %f that is %0.2f%% of |b|.",
                                  iteration, norm, 100.0 * norm / NB0);
        if(reportKthIteration > 0 && iteration % reportKthIteration == 0)
        {
            LOGD << io::xprintf("Writing file %sx_it%02d.den", progressPrefixPath.c_str(),
                                iteration);
            writeVolume(*x_buf,
                        io::xprintf("%sx_it%02d.den", progressPrefixPath.c_str(), iteration));
        }
    }
    Q[0]->enqueueReadBuffer(*x_buf, CL_TRUE, 0, sizeof(float) * XDIM, x);
    return 0;
}

} // namespace KCT
