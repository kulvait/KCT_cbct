#include "ParallelBeamProjector.hpp"

namespace KCT {

int ParallelBeamProjector::arrayIntoBuffer(float* c_array, cl::Buffer cl_buffer, uint64_t size)
{
    uint64_t bufferSize = size * sizeof(float);
    cl_int err = CL_SUCCESS;
    if(c_array != nullptr)
    {
        err = Q[0]->enqueueWriteBuffer(cl_buffer, CL_TRUE, 0, bufferSize, (void*)c_array);
    } else
    {
        KCTERR("Null pointer exception!");
    }
    if(err != CL_SUCCESS)
    {
        std::string msg = io::xprintf("Failed arrayIntoBuffer with code %d!", err);
        KCTERR(msg);
    }
    return 0;
}

int ParallelBeamProjector::bufferIntoArray(cl::Buffer cl_buffer, float* c_array, uint64_t size)
{
    uint64_t bufferSize = size * sizeof(float);
    cl_int err = CL_SUCCESS;
    if(c_array != nullptr)
    {
        err = Q[0]->enqueueReadBuffer(cl_buffer, CL_TRUE, 0, bufferSize, (void*)c_array);
    } else
    {
        KCTERR("Null pointer exception!");
    }
    if(err != CL_SUCCESS)
    {
        std::string msg = io::xprintf("Failed bufferIntoArray with code %d!", err);
        KCTERR(msg);
    }
    return 0;
}

int ParallelBeamProjector::fillVolumeBufferByConstant(float constant)
{
    std::string msg;
    if(volumeBuffer == nullptr)
    {
        msg = io::xprintf(
            "Volume buffer is not yet initialized, call initializeVolumeBuffer first!");
        LOGE << msg;
        throw std::runtime_error(msg);
    }
    return Q[0]->enqueueFillBuffer<cl_float>(*volumeBuffer, constant, 0, totalVolumeBufferSize);
}

int ParallelBeamProjector::fillProjectionBufferByConstant(float constant)
{
    std::string msg;
    if(projectionBuffer == nullptr)
    {
        msg = io::xprintf(
            "Projection buffer is not yet initialized, call initializeProjectionBuffer first!");
        LOGE << msg;
        throw std::runtime_error(msg);
    }
    return Q[0]->enqueueFillBuffer<cl_float>(*projectionBuffer, constant, 0,
                                             totalProjectionBufferSize);
}

int ParallelBeamProjector::project(float* volume, float* projection)
{
    allocateXBuffers(1);
    allocateBBuffers(1);
    std::shared_ptr<cl::Buffer> x_buf, b_buf;
    x_buf = getXBuffer(0);
    b_buf = getBBuffer(0);
    arrayIntoBuffer(volume, *x_buf, XDIM);
    BasePBCTOperator::project(*x_buf, *b_buf);
    bufferIntoArray(*b_buf, projection, BDIM);
    return 0;
}
int ParallelBeamProjector::project_print_discrepancy(float* volume, float* projection, float* rhs)
{
    allocateXBuffers(1);
    allocateBBuffers(2);
    std::shared_ptr<cl::Buffer> x_buf, b_buf, bb_buf;
    x_buf = getXBuffer(0);
    b_buf = getBBuffer(0);
    bb_buf = getBBuffer(1);
    arrayIntoBuffer(volume, *x_buf, XDIM);
    arrayIntoBuffer(rhs, *bb_buf, BDIM);
    BasePBCTOperator::project(*x_buf, *b_buf);
    double normbb = std::sqrt(normBBuffer_barrier_double(*bb_buf));
    algFLOATvector_A_equals_A_plus_cB(*bb_buf, *b_buf, -1.0f, BDIM);
    double normDiscrepancy = std::sqrt(normBBuffer_barrier_double(*bb_buf));
    LOGI << io::xprintf("|Ax-rhs|=%f, |rhs|=%f, discrepancy is %f%% of the initial norm.",
                        normDiscrepancy, normbb, 100.0 * normDiscrepancy / normbb);
    bufferIntoArray(*b_buf, projection, BDIM);
    return 0;
}
int ParallelBeamProjector::backproject(float* projection, float* volume)
{

    allocateXBuffers(1);
    allocateBBuffers(1);
    std::shared_ptr<cl::Buffer> x_buf, b_buf;
    x_buf = getXBuffer(0);
    b_buf = getBBuffer(0);
    arrayIntoBuffer(projection, *b_buf, BDIM);
    BasePBCTOperator::backproject(*b_buf, *x_buf);
    bufferIntoArray(*x_buf, volume, XDIM);
    return 0;
}

double ParallelBeamProjector::normSquare(float* v, uint32_t pdimx, uint32_t pdimy)
{
    size_t vecsize = sizeof(float) * pdimx * pdimy;
    if(tmpBuffer == nullptr || vecsize != tmpBuffer_size)
    {
        tmpBuffer_size = vecsize;
        tmpBuffer = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                 tmpBuffer_size, (void*)v);
    } else
    {
        Q[0]->enqueueWriteBuffer(*tmpBuffer, CL_TRUE, 0, tmpBuffer_size, (void*)v);
    }
    cl::Buffer onedouble(*context, CL_MEM_READ_WRITE, sizeof(double), nullptr);
    double sum;
    uint32_t framesize = pdimx * pdimy;
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(1));
    (*vector_NormSquarePartial)(eargs, *tmpBuffer, onedouble, framesize).wait();
    Q[0]->enqueueReadBuffer(onedouble, CL_TRUE, 0, sizeof(double), &sum);
    return sum;
}

double ParallelBeamProjector::normSquareDifference(float* v, uint32_t pdimx, uint32_t pdimy)
{
    size_t vecsize = sizeof(float) * pdimx * pdimy;
    if(projectionBuffer == nullptr)
    {
        std::string msg = "Comparing to empty buffer is not possible.";
        KCTERR(msg);
    }
    if(vecsize != totalProjectionBufferSize)
    {
        std::string msg = "Can not compare buffers of incompatible dimensions.";
        KCTERR(msg);
    }

    if(tmpBuffer == nullptr || vecsize != tmpBuffer_size)
    {
        tmpBuffer_size = vecsize;
        tmpBuffer = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                 tmpBuffer_size, (void*)v);
    } else
    {
        Q[0]->enqueueWriteBuffer(*tmpBuffer, CL_TRUE, 0, tmpBuffer_size, (void*)v);
    }
    uint32_t framesize = pdimx * pdimy;
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(framesize));
    float factor = -1.0;
    (*FLOATvector_A_equals_Ac_plus_B)(eargs, *tmpBuffer, *projectionBuffer, factor).wait();
    cl::Buffer onedouble(*context, CL_MEM_READ_WRITE, sizeof(double), nullptr);
    double sum;
    cl::EnqueueArgs ear(*Q[0], cl::NDRange(1));
    (*vector_NormSquarePartial)(ear, *tmpBuffer, onedouble, framesize).wait();
    Q[0]->enqueueReadBuffer(onedouble, CL_TRUE, 0, sizeof(double), &sum);
    return sum;
}

} // namespace KCT
