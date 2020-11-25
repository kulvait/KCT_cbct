#include "AlgorithmsBarierBuffers.hpp"

namespace CTL {

/**
 * Funcion computes the norm of the Buffer that has vdimx * vdimy * vdimz elements.
 *
 * @param X
 *
 * @return
 */
double AlgorithmsBarierBuffers::normXBuffer_frame_double(cl::Buffer& X)
{
    double sum;
    uint32_t framesize = vdimx * vdimy;
    cl::EnqueueArgs eargs1(*Q[0], cl::NDRange(vdimz));
    (*vector_NormSquarePartial)(eargs1, X, *tmp_x_red1, framesize).wait();
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(1));
    unsigned int arg = vdimz;
    (*vector_SumPartial)(eargs, *tmp_x_red1, *tmp_x_red2, arg).wait();
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
double AlgorithmsBarierBuffers::normBBuffer_frame_double(cl::Buffer& B)
{ // Use workGroupSize that is private constant default to 256
    double sum;
    uint32_t framesize = pdimx * pdimy;
    cl::EnqueueArgs eargs1(*Q[0], cl::NDRange(pdimz));
    (*vector_NormSquarePartial)(eargs1, B, *tmp_b_red1, framesize).wait();
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(1));
    unsigned int arg = pdimz;
    (*vector_SumPartial)(eargs, *tmp_b_red1, *tmp_b_red2, arg).wait();
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
double AlgorithmsBarierBuffers::normXBuffer_barier_double(cl::Buffer& X)
{
    double sum;
    cl::EnqueueArgs eargs_red1(*Q[0], cl::NDRange(XDIM_ALIGNED), cl::NDRange(workGroupSize));
    cl::LocalSpaceArg localsize = cl::Local(workGroupSize * sizeof(double));
    (*vector_NormSquarePartial_barier)(eargs_red1, X, *tmp_x_red1, localsize, XDIM).wait();
    cl::EnqueueArgs eargs_red2(*Q[0], cl::NDRange(XDIM_REDUCED1_ALIGNED),
                               cl::NDRange(workGroupSize));
    (*vector_SumPartial_barier)(eargs_red2, *tmp_x_red1, *tmp_x_red2, localsize, XDIM_REDUCED1)
        .wait();
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(1));
    (*vector_SumPartial)(eargs, *tmp_x_red2, *tmp_x_red1, XDIM_REDUCED2).wait();
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
double AlgorithmsBarierBuffers::normBBuffer_barier_double(cl::Buffer& B)
{ // Use workGroupSize that is private constant default to 256
    double sum;
    cl::EnqueueArgs eargs_red1(*Q[0], cl::NDRange(BDIM_ALIGNED), cl::NDRange(workGroupSize));
    cl::LocalSpaceArg localsize = cl::Local(workGroupSize * sizeof(double));
    (*vector_NormSquarePartial_barier)(eargs_red1, B, *tmp_b_red1, localsize, BDIM).wait();
    cl::EnqueueArgs eargs_red2(*Q[0], cl::NDRange(BDIM_REDUCED1_ALIGNED),
                               cl::NDRange(workGroupSize));
    (*vector_SumPartial_barier)(eargs_red2, *tmp_b_red1, *tmp_b_red2, localsize, BDIM_REDUCED1)
        .wait();
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(1));
    (*vector_SumPartial)(eargs, *tmp_b_red2, *tmp_b_red1, BDIM_REDUCED2).wait();
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
double AlgorithmsBarierBuffers::scalarProductXBuffer_barier_double(cl::Buffer& A, cl::Buffer& B)
{

    double sum;
    cl::EnqueueArgs eargs_red1(*Q[0], cl::NDRange(XDIM_ALIGNED), cl::NDRange(workGroupSize));
    cl::LocalSpaceArg localsize = cl::Local(workGroupSize * sizeof(double));
    (*vector_ScalarProductPartial_barier)(eargs_red1, A, B, *tmp_x_red1, localsize, XDIM).wait();
    cl::EnqueueArgs eargs_red2(*Q[0], cl::NDRange(XDIM_REDUCED1_ALIGNED),
                               cl::NDRange(workGroupSize));
    (*vector_SumPartial_barier)(eargs_red2, *tmp_x_red1, *tmp_x_red2, localsize, XDIM_REDUCED1)
        .wait();
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(1));
    (*vector_SumPartial)(eargs, *tmp_x_red2, *tmp_x_red1, XDIM_REDUCED2).wait();
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
double AlgorithmsBarierBuffers::scalarProductBBuffer_barier_double(cl::Buffer& A, cl::Buffer& B)
{

    double sum;
    cl::EnqueueArgs eargs_red1(*Q[0], cl::NDRange(BDIM_ALIGNED), cl::NDRange(workGroupSize));
    cl::LocalSpaceArg localsize = cl::Local(workGroupSize * sizeof(double));
    (*vector_ScalarProductPartial_barier)(eargs_red1, A, B, *tmp_b_red1, localsize, BDIM).wait();
    cl::EnqueueArgs eargs_red2(*Q[0], cl::NDRange(BDIM_REDUCED1_ALIGNED),
                               cl::NDRange(workGroupSize));
    (*vector_SumPartial_barier)(eargs_red2, *tmp_b_red1, *tmp_b_red2, localsize, BDIM_REDUCED1)
        .wait();
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(1));
    (*vector_SumPartial)(eargs, *tmp_b_red2, *tmp_b_red1, BDIM_REDUCED2).wait();
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
float AlgorithmsBarierBuffers::normXBuffer_frame(cl::Buffer& X)
{
    float sum;
    uint32_t framesize = vdimx * vdimy;
    cl::EnqueueArgs eargs1(*Q[0], cl::NDRange(vdimz));
    (*FLOATvector_NormSquarePartial)(eargs1, X, *tmp_x_red1, framesize).wait();
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(1));
    unsigned int arg = vdimz;
    (*FLOATvector_SumPartial)(eargs, *tmp_x_red1, *tmp_x_red2, arg).wait();
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
float AlgorithmsBarierBuffers::normBBuffer_frame(cl::Buffer& B)
{ // Use workGroupSize that is private constant default to 256
    float sum;
    uint32_t framesize = pdimx * pdimy;
    cl::EnqueueArgs eargs1(*Q[0], cl::NDRange(pdimz));
    (*FLOATvector_NormSquarePartial)(eargs1, B, *tmp_b_red1, framesize).wait();
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(1));
    unsigned int arg = pdimz;
    (*FLOATvector_SumPartial)(eargs, *tmp_b_red1, *tmp_b_red2, arg).wait();
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
float AlgorithmsBarierBuffers::normXBuffer_barier(cl::Buffer& X)
{

    float sum;
    cl::EnqueueArgs eargs_red1(*Q[0], cl::NDRange(XDIM_ALIGNED), cl::NDRange(workGroupSize));
    cl::LocalSpaceArg localsize = cl::Local(workGroupSize * sizeof(float));
    (*FLOATvector_NormSquarePartial_barier)(eargs_red1, X, *tmp_x_red1, localsize, XDIM).wait();
    cl::EnqueueArgs eargs_red2(*Q[0], cl::NDRange(XDIM_REDUCED1_ALIGNED),
                               cl::NDRange(workGroupSize));
    (*FLOATvector_SumPartial_barier)(eargs_red2, *tmp_x_red1, *tmp_x_red2, localsize, XDIM_REDUCED1)
        .wait();
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(1));
    (*FLOATvector_SumPartial)(eargs, *tmp_x_red2, *tmp_x_red1, XDIM_REDUCED2).wait();
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
float AlgorithmsBarierBuffers::normBBuffer_barier(cl::Buffer& B)
{ // Use workGroupSize that is private constant default to 256
    float sum;
    cl::EnqueueArgs eargs_red1(*Q[0], cl::NDRange(BDIM_ALIGNED), cl::NDRange(workGroupSize));
    cl::LocalSpaceArg localsize = cl::Local(workGroupSize * sizeof(float));
    (*FLOATvector_NormSquarePartial_barier)(eargs_red1, B, *tmp_b_red1, localsize, BDIM).wait();
    cl::EnqueueArgs eargs_red2(*Q[0], cl::NDRange(BDIM_REDUCED1_ALIGNED),
                               cl::NDRange(workGroupSize));
    (*FLOATvector_SumPartial_barier)(eargs_red2, *tmp_b_red1, *tmp_b_red2, localsize, BDIM_REDUCED1)
        .wait();
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(1));
    (*FLOATvector_SumPartial)(eargs, *tmp_b_red2, *tmp_b_red1, BDIM_REDUCED2).wait();
    Q[0]->enqueueReadBuffer(*tmp_b_red1, CL_TRUE, 0, sizeof(float), &sum);
    return sum;
}

/**
 * Copy given float vector into the buffer. The buffer must have appropriate size.
 *
 * @param X Buffer
 * @param v vector
 * @param size size
 *
 * @return
 */
int AlgorithmsBarierBuffers::vectorIntoBuffer(cl::Buffer X, float* v, std::size_t size)
{
    cl_int err = CL_SUCCESS;
    std::string e;
    std::size_t bufferSize;
    X.getInfo(CL_MEM_SIZE, &bufferSize);
    std::size_t totalSize = size * sizeof(float);
    e = io::xprintf("The buffer is %d bytes to represent vector of size %d that is %d bytes.",
                    bufferSize, size, totalSize);
    LOGE << e;
    if(bufferSize >= totalSize)
    {
        err = Q[0]->enqueueWriteBuffer(X, CL_TRUE, 0, totalSize, (void*)v);
        if(err != CL_SUCCESS)
        {
            e = io::xprintf("Unsucessful initialization of Volume with error code %d!", err);
            LOGE << e;
            throw std::runtime_error(e);
        }
    } else
    {
        e = io::xprintf(
            "The buffer is too small %d bytes to represent vector of size %d that is %d bytes.",
            bufferSize, size, totalSize);
        LOGE << e;
        throw std::runtime_error(e);
    }
    return 0;
}

int AlgorithmsBarierBuffers::initializeAlgorithmsBuffers()
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

} // namespace CTL