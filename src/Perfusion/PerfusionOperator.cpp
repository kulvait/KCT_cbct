#include "Perfusion/PerfusionOperator.hpp"

namespace CTL {

int PerfusionOperator::project(bool blocking)
{
    reportTime("PERFUSION PROJECTION", blocking, true);
    allocateBBuffers(1);//b_buf is not writeable as it contains only data for reconstruction
    std::vector<std::shared_ptr<cl::Buffer>> b_prj = getBBuffers(0);
    BasePerfusionReconstructor::project(x_buf, b_prj);
    for(uint32_t sweepID = 0; sweepID != BVNUM; sweepID++)
    {
        Q[0]->enqueueReadBuffer(*b_prj[sweepID], CL_TRUE, 0, sizeof(float) * BDIM, b[sweepID]);
    }
    reportTime("PERFUSION PROJECTION duration", blocking, true);
    return 0;
}

int PerfusionOperator::backproject(bool blocking)
{
    reportTime("PERFUSION BACKPROJECTION", blocking, true);
    BasePerfusionReconstructor::backproject(b_buf, x_buf);
    for(uint32_t vectorID = 0; vectorID != XVNUM; vectorID++)
    {
        Q[0]->enqueueReadBuffer(*x_buf[vectorID], CL_TRUE, 0, sizeof(float) * XDIM, x[vectorID]);
    }
    reportTime("PERFUSION BACKPROJECTION duration", blocking, true);
    return 0;
}

} // namespace CTL
