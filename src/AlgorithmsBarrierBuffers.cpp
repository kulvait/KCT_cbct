#include "AlgorithmsBarrierBuffers.hpp"

namespace KCT {

/**
 * Funcion computes the norm of the Buffer that has vdimx * vdimy * vdimz elements.
 *
 * @param X
 *
 * @return
 */
double AlgorithmsBarrierBuffers::normXBuffer_frame_double(cl::Buffer& X)
{
    double sum;
    uint32_t partialFrameSize = vdimx * vdimy;
    uint32_t partialFrameCount = vdimz;
    algvector_NormSquarePartial(X, *tmp_x_red1, partialFrameSize, partialFrameCount);
    partialFrameSize = vdimz;
    partialFrameCount = 1;
    algvector_SumPartial(*tmp_x_red1, *tmp_x_red2, partialFrameSize, partialFrameCount);
    Q[0]->enqueueReadBuffer(*tmp_x_red2, CL_TRUE, 0, sizeof(double), &sum);
    return sum;
}

/**
 * Funcion computes the norm of the Buffer that has pdimx * pdimy * pdimz elements.
 *
 * @param X
 *
 * @return
 */
double AlgorithmsBarrierBuffers::normBBuffer_frame_double(cl::Buffer& B)
{ // Use workGroupSize that is private constant default to 256
    double sum;
    uint32_t partialFrameSize = pdimx * pdimy;
    uint32_t partialFrameCount = pdimz;
    algvector_NormSquarePartial(B, *tmp_b_red1, partialFrameSize, partialFrameCount);
    partialFrameSize = pdimz;
    partialFrameCount = 1;
    algvector_SumPartial(*tmp_b_red1, *tmp_b_red2, partialFrameSize, partialFrameCount);
    Q[0]->enqueueReadBuffer(*tmp_b_red2, CL_TRUE, 0, sizeof(double), &sum);
    return sum;
}

/**
 * Funcion computes the norm of the Buffer that has vdimx * vdimy * vdimz elements.
 *
 * @param X
 *
 * @return
 */
float AlgorithmsBarrierBuffers::sumXBuffer_barrier_float(cl::Buffer& X)
{
    float sum;
    cl::EnqueueArgs eargs_red1(*Q[0], cl::NDRange(XDIM_ALIGNED), cl::NDRange(workGroupSize));
    cl::LocalSpaceArg localsize = cl::Local(workGroupSize * sizeof(float));
    (*FLOATvector_SumPartial_barrier)(eargs_red1, X, *tmp_x_red1, localsize, XDIM);
    cl::EnqueueArgs eargs_red2(*Q[0], cl::NDRange(XDIM_REDUCED1_ALIGNED),
                               cl::NDRange(workGroupSize));
    (*FLOATvector_SumPartial_barrier)(eargs_red2, *tmp_x_red1, *tmp_x_red2, localsize,
                                      XDIM_REDUCED1);
    uint32_t partialFrameSize = XDIM_REDUCED2;
    uint32_t partialFrameCount = 1;
    algFLOATvector_SumPartial(*tmp_x_red2, *tmp_x_red1, partialFrameSize, partialFrameCount);
    Q[0]->enqueueReadBuffer(*tmp_x_red1, CL_TRUE, 0, sizeof(float), &sum);
    return sum;
}

float AlgorithmsBarrierBuffers::maxXBuffer_barrier_float(cl::Buffer& X)
{
    float sum;
    cl::EnqueueArgs eargs_red1(*Q[0], cl::NDRange(XDIM_ALIGNED), cl::NDRange(workGroupSize));
    cl::LocalSpaceArg localsize = cl::Local(workGroupSize * sizeof(float));
    (*FLOATvector_MaxPartial_barrier)(eargs_red1, X, *tmp_x_red1, localsize, XDIM);
    cl::EnqueueArgs eargs_red2(*Q[0], cl::NDRange(XDIM_REDUCED1_ALIGNED),
                               cl::NDRange(workGroupSize));
    (*FLOATvector_MaxPartial_barrier)(eargs_red2, *tmp_x_red1, *tmp_x_red2, localsize,
                                      XDIM_REDUCED1);
    uint32_t partialFrameSize = XDIM_REDUCED2;
    uint32_t partialFrameCount = 1;
    algFLOATvector_MaxPartial(*tmp_x_red2, *tmp_x_red1, partialFrameSize, partialFrameCount);
    Q[0]->enqueueReadBuffer(*tmp_x_red1, CL_TRUE, 0, sizeof(float), &sum);
    return sum;
}

/**
 * Funcion computes the norm of the Buffer that has vdimx * vdimy * vdimz elements.
 *
 * @param X
 *
 * @return
 */
double AlgorithmsBarrierBuffers::normXBuffer_barrier_double(cl::Buffer& X)
{
    double sum;
    cl::EnqueueArgs eargs_red1(*Q[0], cl::NDRange(XDIM_ALIGNED), cl::NDRange(workGroupSize));
    cl::LocalSpaceArg localsize = cl::Local(workGroupSize * sizeof(double));
    (*vector_NormSquarePartial_barrier)(eargs_red1, X, *tmp_x_red1, localsize, XDIM);
    cl::EnqueueArgs eargs_red2(*Q[0], cl::NDRange(XDIM_REDUCED1_ALIGNED),
                               cl::NDRange(workGroupSize));
    (*vector_SumPartial_barrier)(eargs_red2, *tmp_x_red1, *tmp_x_red2, localsize, XDIM_REDUCED1);
    uint32_t partialFrameSize = XDIM_REDUCED2;
    uint32_t partialFrameCount = 1;
    algvector_SumPartial(*tmp_x_red2, *tmp_x_red1, partialFrameSize, partialFrameCount);
    Q[0]->enqueueReadBuffer(*tmp_x_red1, CL_TRUE, 0, sizeof(double), &sum);
    return sum;
}

/**
 * Funcion computes the norm of the Buffer that has pdimx * pdimy * pdimz elements.
 *
 * @param X
 *
 * @return
 */
float AlgorithmsBarrierBuffers::sumBBuffer_barrier_float(cl::Buffer& B)
{ // Use workGroupSize that is private constant default to 256
    float sum;
    cl::EnqueueArgs eargs_red1(*Q[0], cl::NDRange(BDIM_ALIGNED), cl::NDRange(workGroupSize));
    cl::LocalSpaceArg localsize = cl::Local(workGroupSize * sizeof(float));
    (*FLOATvector_SumPartial_barrier)(eargs_red1, B, *tmp_b_red1, localsize, BDIM);
    cl::EnqueueArgs eargs_red2(*Q[0], cl::NDRange(BDIM_REDUCED1_ALIGNED),
                               cl::NDRange(workGroupSize));
    (*FLOATvector_SumPartial_barrier)(eargs_red2, *tmp_b_red1, *tmp_b_red2, localsize,
                                      BDIM_REDUCED1);
    uint32_t partialFrameSize = BDIM_REDUCED2;
    uint32_t partialFrameCount = 1;
    algFLOATvector_SumPartial(*tmp_b_red2, *tmp_b_red1, partialFrameSize, partialFrameCount);
    Q[0]->enqueueReadBuffer(*tmp_b_red1, CL_TRUE, 0, sizeof(float), &sum);
    return sum;
}

float AlgorithmsBarrierBuffers::maxBBuffer_barrier_float(cl::Buffer& B)
{ // Use workGroupSize that is private constant default to 256
    float sum;
    cl::EnqueueArgs eargs_red1(*Q[0], cl::NDRange(BDIM_ALIGNED), cl::NDRange(workGroupSize));
    cl::LocalSpaceArg localsize = cl::Local(workGroupSize * sizeof(float));
    (*FLOATvector_MaxPartial_barrier)(eargs_red1, B, *tmp_b_red1, localsize, BDIM);
    cl::EnqueueArgs eargs_red2(*Q[0], cl::NDRange(BDIM_REDUCED1_ALIGNED),
                               cl::NDRange(workGroupSize));
    (*FLOATvector_MaxPartial_barrier)(eargs_red2, *tmp_b_red1, *tmp_b_red2, localsize,
                                      BDIM_REDUCED1);
    uint32_t partialFrameSize = BDIM_REDUCED2;
    uint32_t partialFrameCount = 1;
    algFLOATvector_MaxPartial(*tmp_b_red2, *tmp_b_red1, partialFrameSize, partialFrameCount);
    Q[0]->enqueueReadBuffer(*tmp_b_red1, CL_TRUE, 0, sizeof(float), &sum);
    return sum;
}

/**
 * Funcion computes the norm of the Buffer that has pdimx * pdimy * pdimz elements.
 *
 * @param X
 *
 * @return
 */
double AlgorithmsBarrierBuffers::normBBuffer_barrier_double(cl::Buffer& B)
{ // Use workGroupSize that is private constant default to 256
    double sum;
    /*cl_int inf;
    std::string err;
    cl::EnqueueArgs eargs_red1(*Q[0], cl::NDRange(BDIM_ALIGNED), cl::NDRange(workGroupSize));
    cl::LocalSpaceArg localsize = cl::Local(workGroupSize * sizeof(double));
    auto exe = (*vector_NormSquarePartial_barrier)(eargs_red1, B, *tmp_b_red1, localsize, BDIM);
    exe.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS, &inf);
    if(inf != CL_COMPLETE && inf != CL_QUEUED && inf != CL_SUBMITTED && inf != CL_RUNNING)
    {
        err = io::xprintf("COMMAND_EXECUTION_STATUS is %d", inf);
        KCTERR(err);
    }*/
    algvector_NormSquarePartial_barrier(B, *tmp_b_red1, BDIM, BDIM_ALIGNED, workGroupSize);
    /*
        cl::EnqueueArgs eargs_red2(*Q[0], cl::NDRange(BDIM_REDUCED1_ALIGNED),
                                   cl::NDRange(workGroupSize));
        (*vector_SumPartial_barrier)(eargs_red2, *tmp_b_red1, *tmp_b_red2, localsize,
       BDIM_REDUCED1);
    */
    algvector_SumPartial_barrier(*tmp_b_red1, *tmp_b_red2, BDIM_REDUCED1, BDIM_REDUCED1_ALIGNED,
                                 workGroupSize);
    uint32_t partialFrameSize = BDIM_REDUCED2;
    uint32_t partialFrameCount = 1;
    algvector_SumPartial(*tmp_b_red2, *tmp_b_red1, partialFrameSize, partialFrameCount);
    Q[0]->enqueueReadBuffer(*tmp_b_red1, CL_TRUE, 0, sizeof(double), &sum);
    return sum;
}

/**
 * Funcion computes the scalar product of two Buffers that has vdimx * vdimy * vdimz elements.
 *
 * @param A CL buffer of the size vdimx * vdimy * vdimz
 * @param B CL buffer of the size vdimx * vdimy * vdimz
 *
 * @return
 */
double AlgorithmsBarrierBuffers::scalarProductXBuffer_barrier_double(cl::Buffer& A, cl::Buffer& B)
{

    double sum;
    cl::EnqueueArgs eargs_red1(*Q[0], cl::NDRange(XDIM_ALIGNED), cl::NDRange(workGroupSize));
    cl::LocalSpaceArg localsize = cl::Local(workGroupSize * sizeof(double));
    (*vector_ScalarProductPartial_barrier)(eargs_red1, A, B, *tmp_x_red1, localsize, XDIM);
    cl::EnqueueArgs eargs_red2(*Q[0], cl::NDRange(XDIM_REDUCED1_ALIGNED),
                               cl::NDRange(workGroupSize));
    (*vector_SumPartial_barrier)(eargs_red2, *tmp_x_red1, *tmp_x_red2, localsize, XDIM_REDUCED1);
    uint32_t partialFrameSize = XDIM_REDUCED2;
    uint32_t partialFrameCount = 1;
    algvector_SumPartial(*tmp_x_red2, *tmp_x_red1, partialFrameSize, partialFrameCount);
    Q[0]->enqueueReadBuffer(*tmp_x_red1, CL_TRUE, 0, sizeof(double), &sum);
    return sum;
}

/**
 * Funcion computes the scalar product of two Buffers that has pdimx * pdimy * pdimz elements.
 *
 * @param A CL buffer of the size pdimx * pdimy * pdimz
 * @param B CL buffer of the size pdimx * pdimy * pdimz
 *
 * @return
 */
double AlgorithmsBarrierBuffers::scalarProductBBuffer_barrier_double(cl::Buffer& A, cl::Buffer& B)
{

    double sum;
    cl::EnqueueArgs eargs_red1(*Q[0], cl::NDRange(BDIM_ALIGNED), cl::NDRange(workGroupSize));
    cl::LocalSpaceArg localsize = cl::Local(workGroupSize * sizeof(double));
    (*vector_ScalarProductPartial_barrier)(eargs_red1, A, B, *tmp_b_red1, localsize, BDIM);
    cl::EnqueueArgs eargs_red2(*Q[0], cl::NDRange(BDIM_REDUCED1_ALIGNED),
                               cl::NDRange(workGroupSize));
    (*vector_SumPartial_barrier)(eargs_red2, *tmp_b_red1, *tmp_b_red2, localsize, BDIM_REDUCED1);
    uint32_t partialFrameSize = BDIM_REDUCED2;
    uint32_t partialFrameCount = 1;
    algvector_SumPartial(*tmp_b_red2, *tmp_b_red1, partialFrameSize, partialFrameCount);
    Q[0]->enqueueReadBuffer(*tmp_b_red1, CL_TRUE, 0, sizeof(double), &sum);
    return sum;
}

/**
 * Funcion computes the norm of the Buffer that has vdimx * vdimy * vdimz elements.
 *
 * @param X
 *
 * @return
 */
float AlgorithmsBarrierBuffers::normXBuffer_frame(cl::Buffer& X)
{
    float sum;
    uint32_t partialFrameSize = vdimx * vdimy;
    uint32_t partialFrameCount = vdimz;
    algFLOATvector_NormSquarePartial(X, *tmp_x_red1, partialFrameSize, partialFrameCount);
    partialFrameSize = vdimz;
    partialFrameCount = 1;
    algFLOATvector_SumPartial(*tmp_x_red1, *tmp_x_red2, partialFrameSize, partialFrameCount);
    Q[0]->enqueueReadBuffer(*tmp_x_red2, CL_TRUE, 0, sizeof(float), &sum);
    return sum;
}

/**
 * Funcion computes the norm of the Buffer that has pdimx * pdimy * pdimz elements.
 *
 * @param X
 *
 * @return
 */
float AlgorithmsBarrierBuffers::normBBuffer_frame(cl::Buffer& B)
{ // Use workGroupSize that is private constant default to 256
    float sum;
    uint32_t partialFrameSize = pdimx * pdimy;
    uint32_t partialFrameCount = pdimz;
    algFLOATvector_NormSquarePartial(B, *tmp_b_red1, partialFrameSize, partialFrameCount);
    partialFrameSize = pdimz;
    partialFrameCount = 1;
    algFLOATvector_SumPartial(*tmp_b_red1, *tmp_b_red2, partialFrameSize, partialFrameCount);
    Q[0]->enqueueReadBuffer(*tmp_b_red2, CL_TRUE, 0, sizeof(float), &sum);
    return sum;
}

/**
 * Funcion computes the norm of the Buffer that has vdimx * vdimy * vdimz elements.
 *
 * @param X
 *
 * @return
 */
float AlgorithmsBarrierBuffers::normXBuffer_barrier(cl::Buffer& X)
{

    float sum;
    cl::EnqueueArgs eargs_red1(*Q[0], cl::NDRange(XDIM_ALIGNED), cl::NDRange(workGroupSize));
    cl::LocalSpaceArg localsize = cl::Local(workGroupSize * sizeof(float));
    (*FLOATvector_NormSquarePartial_barrier)(eargs_red1, X, *tmp_x_red1, localsize, XDIM);
    cl::EnqueueArgs eargs_red2(*Q[0], cl::NDRange(XDIM_REDUCED1_ALIGNED),
                               cl::NDRange(workGroupSize));
    (*FLOATvector_SumPartial_barrier)(eargs_red2, *tmp_x_red1, *tmp_x_red2, localsize,
                                      XDIM_REDUCED1);
    uint32_t partialFrameSize = XDIM_REDUCED2;
    uint32_t partialFrameCount = 1;
    algFLOATvector_SumPartial(*tmp_x_red2, *tmp_x_red1, partialFrameSize, partialFrameCount);
    Q[0]->enqueueReadBuffer(*tmp_x_red1, CL_TRUE, 0, sizeof(float), &sum);
    return sum;
}

/**
 * Funcion computes the norm of the Buffer that has pdimx * pdimy * pdimz elements.
 *
 * @param X
 *
 * @return
 */
float AlgorithmsBarrierBuffers::normBBuffer_barrier(cl::Buffer& B)
{ // Use workGroupSize that is private constant default to 256
    float sum;
    cl::EnqueueArgs eargs_red1(*Q[0], cl::NDRange(BDIM_ALIGNED), cl::NDRange(workGroupSize));
    cl::LocalSpaceArg localsize = cl::Local(workGroupSize * sizeof(float));
    (*FLOATvector_NormSquarePartial_barrier)(eargs_red1, B, *tmp_b_red1, localsize, BDIM);
    cl::EnqueueArgs eargs_red2(*Q[0], cl::NDRange(BDIM_REDUCED1_ALIGNED),
                               cl::NDRange(workGroupSize));
    (*FLOATvector_SumPartial_barrier)(eargs_red2, *tmp_b_red1, *tmp_b_red2, localsize,
                                      BDIM_REDUCED1);
    uint32_t partialFrameSize = BDIM_REDUCED2;
    uint32_t partialFrameCount = 1;
    algFLOATvector_SumPartial(*tmp_b_red2, *tmp_b_red1, partialFrameSize, partialFrameCount);
    Q[0]->enqueueReadBuffer(*tmp_b_red1, CL_TRUE, 0, sizeof(float), &sum);
    return sum;
}

int AlgorithmsBarrierBuffers::arrayIntoBuffer(float* c_array, cl::Buffer cl_buffer, uint64_t size)
{
    std::string msg;
    if(c_array != nullptr)
    {
        uint64_t arrayByteSize = size * sizeof(float);
        uint64_t bufferSize;
        cl_int err = CL_SUCCESS;
        cl_buffer.getInfo(CL_MEM_SIZE, &bufferSize);
        if(bufferSize >= arrayByteSize)
        {
            if(bufferSize != arrayByteSize)
            {
                msg = io::xprintf(
                    "The buffer of %lu bytes is larger than array of size %lu that is %lu bytes.",
                    bufferSize, size, arrayByteSize);
                LOGD << msg;
            }
            // Actual code here
            err = Q[0]->enqueueWriteBuffer(cl_buffer, CL_TRUE, 0, arrayByteSize, (void*)c_array);
            if(err != CL_SUCCESS)
            {
                msg = io::xprintf("Failed arrayIntoBuffer with code %d!", err);
                KCTERR(msg);
            }
        } else
        {
            msg = io::xprintf("The buffer of %lu bytes is too small to represent vector of size "
                              "%lu that is %lu bytes.",
                              bufferSize, size, arrayByteSize);
            KCTERR(msg);
        }
    } else
    {
        KCTERR("Null pointer exception!");
    }
    return 0;
}

int AlgorithmsBarrierBuffers::bufferIntoArray(cl::Buffer cl_buffer, float* c_array, uint64_t size)
{
    std::string msg;
    if(c_array != nullptr)
    {
        uint64_t arrayByteSize = size * sizeof(float);
        uint64_t bufferSize;
        cl_int err = CL_SUCCESS;
        cl_buffer.getInfo(CL_MEM_SIZE, &bufferSize);
        if(bufferSize <= arrayByteSize)
        {
            if(bufferSize != arrayByteSize)
            {
                msg = io::xprintf(
                    "The buffer of %lu bytes is smaller than array of size %lu that is %lu bytes.",
                    bufferSize, size, arrayByteSize);
                LOGD << msg;
            }
            // Actual code here
            err = Q[0]->enqueueReadBuffer(cl_buffer, CL_TRUE, 0, arrayByteSize, (void*)c_array);
            if(err != CL_SUCCESS)
            {
                std::string msg = io::xprintf("Failed bufferIntoArray with code %d!", err);
                KCTERR(msg);
            }
        } else
        {
            msg = io::xprintf(
                "The buffer of %lu bytes is too big to be written to the vector of size "
                "%lu that is %lu bytes.",
                bufferSize, size, arrayByteSize);
            KCTERR(msg);
        }
    } else
    {
        KCTERR("Null pointer exception!");
    }
    return 0;
}

int AlgorithmsBarrierBuffers::initializeAlgorithmsBuffers()
{
    cl_int err;
    if(algorithmsBuffersInitialized)
    {
        throw std::runtime_error("Buffers already initialized!");
    }
    if(XDIM > BDIM)
    {
        tmp_x_red1 = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                  sizeof(double) * XDIM_REDUCED1, nullptr, &err);
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
            return -1;
        }
        tmp_x_red2 = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                  sizeof(double) * XDIM_REDUCED2, nullptr, &err);
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
            return -1;
        }
        tmp_b_red1 = tmp_x_red1;
        tmp_b_red2 = tmp_x_red2;
    } else
    {

        tmp_b_red1 = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                  sizeof(double) * BDIM_REDUCED1, nullptr, &err);
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
            return -1;
        }
        tmp_b_red2 = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                  sizeof(double) * BDIM_REDUCED2, nullptr, &err);
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
            return -1;
        }
        tmp_x_red1 = tmp_b_red1;
        tmp_x_red2 = tmp_b_red2;
    }
    algorithmsBuffersInitialized = true;
    return 0;
}

} // namespace KCT
