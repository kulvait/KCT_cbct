#include "AlgorithmsBarrierBuffers.hpp"

namespace KCT {
/**
 * Funcion computes the norm of the Buffer that has vdimx * vdimy * vdimz elements.
 *
 * @param X
 *
 * @return
 */
double AlgorithmsBarrierBuffers::normXBuffer_frame_double(cl::Buffer& X,
                                                          std::shared_ptr<ReductionParameters> rp,
                                                          uint32_t QID)
{
    if(rp == nullptr)
    {
        rp = this->rp;
    }
    double sum;
    uint32_t partialFrameSize = rp->vFrameSize;
    uint32_t partialFrameCount = rp->vdimz;
    algvector_NormSquarePartial(X, *tmp_red1[QID], partialFrameSize, partialFrameCount);
    partialFrameSize = rp->vdimz;
    partialFrameCount = 1;
    algvector_SumPartial(*tmp_red1[QID], *tmp_red2[QID], partialFrameSize, partialFrameCount);
    Q[QID]->enqueueReadBuffer(*tmp_red2[QID], CL_TRUE, 0, sizeof(double), &sum);
    return sum;
}

/**
 * Funcion computes the norm of the Buffer that has vdimx * vdimy * vdimz elements.
 *
 * @param X
 *
 * @return
 */
float AlgorithmsBarrierBuffers::normXBuffer_frame(cl::Buffer& X,
                                                  std::shared_ptr<ReductionParameters> rp,
                                                  uint32_t QID)
{
    if(rp == nullptr)
    {
        rp = this->rp;
    }
    float sum;
    uint32_t partialFrameSize = rp->vFrameSize;
    uint32_t partialFrameCount = rp->vdimz;
    algFLOATvector_NormSquarePartial(X, *tmp_red1[QID], partialFrameSize, partialFrameCount);
    partialFrameSize = rp->vdimz;
    partialFrameCount = 1;
    algFLOATvector_SumPartial(*tmp_red1[QID], *tmp_red2[QID], partialFrameSize, partialFrameCount);
    Q[QID]->enqueueReadBuffer(*tmp_red2[QID], CL_TRUE, 0, sizeof(float), &sum);
    return sum;
}

/**
 * Funcion computes the norm of the Buffer that has vdimx * vdimy * vdimz elements.
 *
 * @param X
 *
 * @return
 */
double AlgorithmsBarrierBuffers::normXBuffer_barrier_double(cl::Buffer& X,
                                                            std::shared_ptr<ReductionParameters> rp,
                                                            uint32_t QID)
{
    if(rp == nullptr)
    {
        rp = this->rp;
    }
    double sum;
    uint32_t workGroupSize = rp->workGroupSize;
    cl::EnqueueArgs eargs_red1(*Q[QID], cl::NDRange(rp->XDIM_ALIGNED), cl::NDRange(workGroupSize));
    cl::LocalSpaceArg localsize = cl::Local(workGroupSize * sizeof(double));
    (*vector_NormSquarePartial_barrier)(eargs_red1, X, *tmp_red1[QID], localsize, rp->XDIM);
    cl::EnqueueArgs eargs_red2(*Q[QID], cl::NDRange(rp->XDIM_REDUCED1_ALIGNED),
                               cl::NDRange(workGroupSize));
    (*vector_SumPartial_barrier)(eargs_red2, *tmp_red1[QID], *tmp_red2[QID], localsize,
                                 rp->XDIM_REDUCED1);
    uint32_t partialFrameSize = rp->XDIM_REDUCED2;
    uint32_t partialFrameCount = 1;
    algvector_SumPartial(*tmp_red2[QID], *tmp_red1[QID], partialFrameSize, partialFrameCount);
    Q[QID]->enqueueReadBuffer(*tmp_red1[QID], CL_TRUE, 0, sizeof(double), &sum);
    return sum;
}

/**
 * Funcion computes the norm of the Buffer that has vdimx * vdimy * vdimz elements.
 *
 * @param X
 *
 * @return
 */
float AlgorithmsBarrierBuffers::normXBuffer_barrier(cl::Buffer& X,
                                                    std::shared_ptr<ReductionParameters> rp,
                                                    uint32_t QID)
{
    if(rp == nullptr)
    {
        rp = this->rp;
    }

    float sum;
    uint32_t workGroupSize = rp->workGroupSize;
    cl::EnqueueArgs eargs_red1(*Q[QID], cl::NDRange(rp->XDIM_ALIGNED), cl::NDRange(workGroupSize));
    cl::LocalSpaceArg localsize = cl::Local(workGroupSize * sizeof(float));
    (*FLOATvector_NormSquarePartial_barrier)(eargs_red1, X, *tmp_red1[QID], localsize, rp->XDIM);
    cl::EnqueueArgs eargs_red2(*Q[QID], cl::NDRange(rp->XDIM_REDUCED1_ALIGNED),
                               cl::NDRange(workGroupSize));
    (*FLOATvector_SumPartial_barrier)(eargs_red2, *tmp_red1[QID], *tmp_red2[QID], localsize,
                                      rp->XDIM_REDUCED1);
    uint32_t partialFrameSize = rp->XDIM_REDUCED2;
    uint32_t partialFrameCount = 1;
    algFLOATvector_SumPartial(*tmp_red2[QID], *tmp_red1[QID], partialFrameSize, partialFrameCount);
    Q[QID]->enqueueReadBuffer(*tmp_red1[QID], CL_TRUE, 0, sizeof(float), &sum);
    return sum;
}

/**
 * Funcion computes the norm of the Buffer that has pdimx * pdimy * pdimz elements.
 *
 * @param X
 *
 * @return
 */
double AlgorithmsBarrierBuffers::normBBuffer_frame_double(cl::Buffer& B,
                                                          std::shared_ptr<ReductionParameters> rp,
                                                          uint32_t QID)
{ // Use workGroupSize that is private constant default to 256
    if(rp == nullptr)
    {
        rp = this->rp;
    }
    double sum;
    uint32_t partialFrameSize = rp->pFrameSize;
    uint32_t partialFrameCount = rp->pdimz;
    algvector_NormSquarePartial(B, *tmp_red1[QID], partialFrameSize, partialFrameCount);
    partialFrameSize = rp->pdimz;
    partialFrameCount = 1;
    algvector_SumPartial(*tmp_red1[QID], *tmp_red2[QID], partialFrameSize, partialFrameCount);
    Q[QID]->enqueueReadBuffer(*tmp_red2[QID], CL_TRUE, 0, sizeof(double), &sum);
    return sum;
}

/**
 * Funcion computes the norm of the Buffer that has pdimx * pdimy * pdimz elements.
 *
 * @param X
 *
 * @return
 */
float AlgorithmsBarrierBuffers::normBBuffer_frame(cl::Buffer& B,
                                                  std::shared_ptr<ReductionParameters> rp,
                                                  uint32_t QID)
{ // Use workGroupSize that is private constant default to 256
    if(rp == nullptr)
    {
        rp = this->rp;
    }
    float sum;
    uint32_t partialFrameSize = rp->pFrameSize;
    uint32_t partialFrameCount = rp->pdimz;
    algFLOATvector_NormSquarePartial(B, *tmp_red1[QID], partialFrameSize, partialFrameCount);
    partialFrameSize = rp->pdimz;
    partialFrameCount = 1;
    algFLOATvector_SumPartial(*tmp_red1[QID], *tmp_red2[QID], partialFrameSize, partialFrameCount);
    Q[QID]->enqueueReadBuffer(*tmp_red2[QID], CL_TRUE, 0, sizeof(float), &sum);
    return sum;
}

/**
 * Funcion computes the norm of the Buffer that has pdimx * pdimy * pdimz elements.
 *
 * @param X
 *
 * @return
 */
double AlgorithmsBarrierBuffers::normBBuffer_barrier_double(cl::Buffer& B,
                                                            std::shared_ptr<ReductionParameters> rp,
                                                            uint32_t QID)
{ // Use workGroupSize that is private constant default to 256
    if(rp == nullptr)
    {
        rp = this->rp;
    }
    double sum;
    bool blocking = false;
    uint32_t workGroupSize = rp->workGroupSize;
    algvector_NormSquarePartial_barrier(B, *tmp_red1[QID], rp->BDIM, rp->BDIM_ALIGNED,
                                        workGroupSize, blocking, QID);
    algvector_SumPartial_barrier(*tmp_red1[QID], *tmp_red2[QID], rp->BDIM_REDUCED1,
                                 rp->BDIM_REDUCED1_ALIGNED, workGroupSize, blocking, QID);
    uint32_t partialFrameSize = rp->BDIM_REDUCED2;
    uint32_t partialFrameCount = 1;
    algvector_SumPartial(*tmp_red2[QID], *tmp_red1[QID], partialFrameSize, partialFrameCount);
    Q[QID]->enqueueReadBuffer(*tmp_red1[QID], CL_TRUE, 0, sizeof(double), &sum);
    return sum;
}

/**
 * Funcion computes the norm of the Buffer that has pdimx * pdimy * pdimz elements.
 *
 * @param X
 *
 * @return
 */
float AlgorithmsBarrierBuffers::normBBuffer_barrier(cl::Buffer& B,
                                                    std::shared_ptr<ReductionParameters> rp,
                                                    uint32_t QID)
{ // Use workGroupSize that is private constant default to 256
    if(rp == nullptr)
    {
        rp = this->rp;
    }
    float sum;
    uint32_t workGroupSize = rp->workGroupSize;
    cl::EnqueueArgs eargs_red1(*Q[QID], cl::NDRange(rp->BDIM_ALIGNED), cl::NDRange(workGroupSize));
    cl::LocalSpaceArg localsize = cl::Local(workGroupSize * sizeof(float));
    (*FLOATvector_NormSquarePartial_barrier)(eargs_red1, B, *tmp_red1[QID], localsize, rp->BDIM);
    cl::EnqueueArgs eargs_red2(*Q[QID], cl::NDRange(rp->BDIM_REDUCED1_ALIGNED),
                               cl::NDRange(workGroupSize));
    (*FLOATvector_SumPartial_barrier)(eargs_red2, *tmp_red1[QID], *tmp_red2[QID], localsize,
                                      rp->BDIM_REDUCED1);
    uint32_t partialFrameSize = rp->BDIM_REDUCED2;
    uint32_t partialFrameCount = 1;
    algFLOATvector_SumPartial(*tmp_red2[QID], *tmp_red1[QID], partialFrameSize, partialFrameCount);
    Q[QID]->enqueueReadBuffer(*tmp_red1[QID], CL_TRUE, 0, sizeof(float), &sum);
    return sum;
}

/**
 * Funcion computes the norm of the Buffer that has vdimx * vdimy * vdimz elements.
 *
 * @param X
 *
 * @return
 */
float AlgorithmsBarrierBuffers::sumXBuffer_barrier_float(cl::Buffer& X,
                                                         std::shared_ptr<ReductionParameters> rp,
                                                         uint32_t QID)
{
    if(rp == nullptr)
    {
        rp = this->rp;
    }
    float sum;
    cl::EnqueueArgs eargs_red1(*Q[QID], cl::NDRange(rp->XDIM_ALIGNED),
                               cl::NDRange(rp->workGroupSize));
    cl::LocalSpaceArg localsize = cl::Local(rp->workGroupSize * sizeof(float));
    (*FLOATvector_SumPartial_barrier)(eargs_red1, X, *tmp_red1[QID], localsize, rp->XDIM);
    cl::EnqueueArgs eargs_red2(*Q[QID], cl::NDRange(rp->XDIM_REDUCED1_ALIGNED),
                               cl::NDRange(rp->workGroupSize));
    (*FLOATvector_SumPartial_barrier)(eargs_red2, *tmp_red1[QID], *tmp_red2[QID], localsize,
                                      rp->XDIM_REDUCED1);
    uint32_t partialFrameSize = rp->XDIM_REDUCED2;
    uint32_t partialFrameCount = 1;
    algFLOATvector_SumPartial(*tmp_red2[QID], *tmp_red1[QID], partialFrameSize, partialFrameCount);
    Q[QID]->enqueueReadBuffer(*tmp_red1[QID], CL_TRUE, 0, sizeof(float), &sum);
    return sum;
}

float AlgorithmsBarrierBuffers::maxXBuffer_barrier_float(cl::Buffer& X,
                                                         std::shared_ptr<ReductionParameters> rp,
                                                         uint32_t QID)
{
    if(rp == nullptr)
    {
        rp = this->rp;
    }
    float sum;
    uint32_t workGroupSize = rp->workGroupSize;
    cl::EnqueueArgs eargs_red1(*Q[QID], cl::NDRange(rp->XDIM_ALIGNED), cl::NDRange(workGroupSize));
    cl::LocalSpaceArg localsize = cl::Local(workGroupSize * sizeof(float));
    (*FLOATvector_MaxPartial_barrier)(eargs_red1, X, *tmp_red1[QID], localsize, rp->XDIM);
    cl::EnqueueArgs eargs_red2(*Q[QID], cl::NDRange(rp->XDIM_REDUCED1_ALIGNED),
                               cl::NDRange(workGroupSize));
    (*FLOATvector_MaxPartial_barrier)(eargs_red2, *tmp_red1[QID], *tmp_red2[QID], localsize,
                                      rp->XDIM_REDUCED1);
    uint32_t partialFrameSize = rp->XDIM_REDUCED2;
    uint32_t partialFrameCount = 1;
    algFLOATvector_MaxPartial(*tmp_red2[QID], *tmp_red1[QID], partialFrameSize, partialFrameCount);
    Q[QID]->enqueueReadBuffer(*tmp_red1[QID], CL_TRUE, 0, sizeof(float), &sum);
    return sum;
}

/**
 * Funcion computes the norm of the Buffer that has pdimx * pdimy * pdimz elements.
 *
 * @param X
 *
 * @return
 */
float AlgorithmsBarrierBuffers::sumBBuffer_barrier_float(cl::Buffer& B,
                                                         std::shared_ptr<ReductionParameters> rp,
                                                         uint32_t QID)
{ // Use workGroupSize that is private constant default to 256
    if(rp == nullptr)
    {
        rp = this->rp;
    }
    float sum;
    uint32_t workGroupSize = rp->workGroupSize;
    cl::EnqueueArgs eargs_red1(*Q[QID], cl::NDRange(rp->BDIM_ALIGNED), cl::NDRange(workGroupSize));
    cl::LocalSpaceArg localsize = cl::Local(workGroupSize * sizeof(float));
    (*FLOATvector_SumPartial_barrier)(eargs_red1, B, *tmp_red1[QID], localsize, rp->BDIM);
    cl::EnqueueArgs eargs_red2(*Q[QID], cl::NDRange(rp->BDIM_REDUCED1_ALIGNED),
                               cl::NDRange(workGroupSize));
    (*FLOATvector_SumPartial_barrier)(eargs_red2, *tmp_red1[QID], *tmp_red2[QID], localsize,
                                      rp->BDIM_REDUCED1);
    uint32_t partialFrameSize = rp->BDIM_REDUCED2;
    uint32_t partialFrameCount = 1;
    algFLOATvector_SumPartial(*tmp_red2[QID], *tmp_red1[QID], partialFrameSize, partialFrameCount);
    Q[QID]->enqueueReadBuffer(*tmp_red1[QID], CL_TRUE, 0, sizeof(float), &sum);
    return sum;
}

float AlgorithmsBarrierBuffers::maxBBuffer_barrier_float(cl::Buffer& B,
                                                         std::shared_ptr<ReductionParameters> rp,
                                                         uint32_t QID)
{ // Use workGroupSize that is private constant default to 256
    if(rp == nullptr)
    {
        rp = this->rp;
    }
    float sum;
    uint32_t workGroupSize = rp->workGroupSize;
    cl::EnqueueArgs eargs_red1(*Q[QID], cl::NDRange(rp->BDIM_ALIGNED), cl::NDRange(workGroupSize));
    cl::LocalSpaceArg localsize = cl::Local(workGroupSize * sizeof(float));
    (*FLOATvector_MaxPartial_barrier)(eargs_red1, B, *tmp_red1[QID], localsize, rp->BDIM);
    cl::EnqueueArgs eargs_red2(*Q[QID], cl::NDRange(rp->BDIM_REDUCED1_ALIGNED),
                               cl::NDRange(workGroupSize));
    (*FLOATvector_MaxPartial_barrier)(eargs_red2, *tmp_red1[QID], *tmp_red2[QID], localsize,
                                      rp->BDIM_REDUCED1);
    uint32_t partialFrameSize = rp->BDIM_REDUCED2;
    uint32_t partialFrameCount = 1;
    algFLOATvector_MaxPartial(*tmp_red2[QID], *tmp_red1[QID], partialFrameSize, partialFrameCount);
    Q[QID]->enqueueReadBuffer(*tmp_red1[QID], CL_TRUE, 0, sizeof(float), &sum);
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
double AlgorithmsBarrierBuffers::scalarProductXBuffer_barrier_double(
    cl::Buffer& A, cl::Buffer& B, std::shared_ptr<ReductionParameters> rp, uint32_t QID)
{
    if(rp == nullptr)
    {
        rp = this->rp;
    }
    double sum;
    uint32_t workGroupSize = rp->workGroupSize;
    cl::EnqueueArgs eargs_red1(*Q[QID], cl::NDRange(rp->XDIM_ALIGNED), cl::NDRange(workGroupSize));
    cl::LocalSpaceArg localsize = cl::Local(workGroupSize * sizeof(double));
    (*vector_ScalarProductPartial_barrier)(eargs_red1, A, B, *tmp_red1[QID], localsize, rp->XDIM);
    cl::EnqueueArgs eargs_red2(*Q[QID], cl::NDRange(rp->XDIM_REDUCED1_ALIGNED),
                               cl::NDRange(workGroupSize));
    (*vector_SumPartial_barrier)(eargs_red2, *tmp_red1[QID], *tmp_red2[QID], localsize,
                                 rp->XDIM_REDUCED1);
    uint32_t partialFrameSize = rp->XDIM_REDUCED2;
    uint32_t partialFrameCount = 1;
    algvector_SumPartial(*tmp_red2[QID], *tmp_red1[QID], partialFrameSize, partialFrameCount);
    Q[QID]->enqueueReadBuffer(*tmp_red1[QID], CL_TRUE, 0, sizeof(double), &sum);
    return sum;
}

/**
 * Funcion computes the norm of the Buffer that has vdimx * vdimy * vdimz elements.
 *
 * @param X
 *
 * @return
 */
double AlgorithmsBarrierBuffers::isotropicTVNormXBuffer_barrier_double(
    cl::Buffer& GX, cl::Buffer& GY, std::shared_ptr<ReductionParameters> rp, uint32_t QID)
{
    if(rp == nullptr)
    {
        rp = this->rp;
    }
    double sum;
    uint32_t workGroupSize = rp->workGroupSize;
    cl::EnqueueArgs eargs_red1(*Q[QID], cl::NDRange(rp->XDIM_ALIGNED), cl::NDRange(workGroupSize));
    cl::LocalSpaceArg localsize = cl::Local(workGroupSize * sizeof(double));
    (*vector_L1L2norm_barrier)(eargs_red1, GX, GY, *tmp_red1[QID], localsize, rp->XDIM);
    cl::EnqueueArgs eargs_red2(*Q[QID], cl::NDRange(rp->XDIM_REDUCED1_ALIGNED),
                               cl::NDRange(workGroupSize));
    (*vector_SumPartial_barrier)(eargs_red2, *tmp_red1[QID], *tmp_red2[QID], localsize,
                                 rp->XDIM_REDUCED1);
    uint32_t partialFrameSize = rp->XDIM_REDUCED2;
    uint32_t partialFrameCount = 1;
    algvector_SumPartial(*tmp_red2[QID], *tmp_red1[QID], partialFrameSize, partialFrameCount);
    Q[QID]->enqueueReadBuffer(*tmp_red1[QID], CL_TRUE, 0, sizeof(double), &sum);
    return sum;
}

/**
 * Funcion computes the norm of the Buffer that has vdimx * vdimy * vdimz elements.
 *
 * @param X
 *
 * @return
 */
float AlgorithmsBarrierBuffers::isotropicTVNormXBuffer_barrier_float(
    cl::Buffer& GX, cl::Buffer& GY, std::shared_ptr<ReductionParameters> rp, uint32_t QID)
{
    if(rp == nullptr)
    {
        rp = this->rp;
    }

    float sum;
    uint32_t workGroupSize = rp->workGroupSize;
    cl::EnqueueArgs eargs_red1(*Q[QID], cl::NDRange(rp->XDIM_ALIGNED), cl::NDRange(workGroupSize));
    cl::LocalSpaceArg localsize = cl::Local(workGroupSize * sizeof(float));
    (*FLOATvector_L1L2norm_barrier)(eargs_red1, GX, GY, *tmp_red1[QID], localsize, rp->XDIM);
    cl::EnqueueArgs eargs_red2(*Q[QID], cl::NDRange(rp->XDIM_REDUCED1_ALIGNED),
                               cl::NDRange(workGroupSize));
    (*FLOATvector_SumPartial_barrier)(eargs_red2, *tmp_red1[QID], *tmp_red2[QID], localsize,
                                      rp->XDIM_REDUCED1);
    uint32_t partialFrameSize = rp->XDIM_REDUCED2;
    uint32_t partialFrameCount = 1;
    algFLOATvector_SumPartial(*tmp_red2[QID], *tmp_red1[QID], partialFrameSize, partialFrameCount);
    Q[QID]->enqueueReadBuffer(*tmp_red1[QID], CL_TRUE, 0, sizeof(float), &sum);
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
double AlgorithmsBarrierBuffers::scalarProductBBuffer_barrier_double(
    cl::Buffer& A, cl::Buffer& B, std::shared_ptr<ReductionParameters> rp, uint32_t QID)
{
    if(rp == nullptr)
    {
        rp = this->rp;
    }

    double sum;
    uint32_t workGroupSize = rp->workGroupSize;
    cl::EnqueueArgs eargs_red1(*Q[QID], cl::NDRange(rp->BDIM_ALIGNED), cl::NDRange(workGroupSize));
    cl::LocalSpaceArg localsize = cl::Local(workGroupSize * sizeof(double));
    (*vector_ScalarProductPartial_barrier)(eargs_red1, A, B, *tmp_red1[QID], localsize, rp->BDIM);
    cl::EnqueueArgs eargs_red2(*Q[QID], cl::NDRange(rp->BDIM_REDUCED1_ALIGNED),
                               cl::NDRange(workGroupSize));
    (*vector_SumPartial_barrier)(eargs_red2, *tmp_red1[QID], *tmp_red2[QID], localsize,
                                 rp->BDIM_REDUCED1);
    uint32_t partialFrameSize = rp->BDIM_REDUCED2;
    uint32_t partialFrameCount = 1;
    algvector_SumPartial(*tmp_red2[QID], *tmp_red1[QID], partialFrameSize, partialFrameCount);
    Q[QID]->enqueueReadBuffer(*tmp_red1[QID], CL_TRUE, 0, sizeof(double), &sum);
    return sum;
}

int AlgorithmsBarrierBuffers::arrayIntoBuffer(float* c_array,
                                              cl::Buffer cl_buffer,
                                              uint64_t size,
                                              uint32_t QID)
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
            err = Q[QID]->enqueueWriteBuffer(cl_buffer, CL_TRUE, 0, arrayByteSize, (void*)c_array);
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

int AlgorithmsBarrierBuffers::bufferIntoArray(cl::Buffer cl_buffer,
                                              float* c_array,
                                              uint64_t size,
                                              uint32_t QID)
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
            err = Q[QID]->enqueueReadBuffer(cl_buffer, CL_TRUE, 0, arrayByteSize, (void*)c_array);
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

void AlgorithmsBarrierBuffers::initReductionParameters(uint32_t pdimx,
                                                       uint32_t pdimy,
                                                       uint32_t pdimz,
                                                       uint32_t vdimx,
                                                       uint32_t vdimy,
                                                       uint32_t vdimz,
                                                       uint32_t workGroupSize)

{
    rp = std::make_shared<ReductionParameters>(pdimx, pdimy, pdimz, vdimx, vdimy, vdimz,
                                               workGroupSize);
    const uint32_t UINT32_MAXXX = ((uint32_t)-1);
    if(rp->XDIM_REDUCED2 > UINT32_MAXXX)
    {
        err = io::xprintf(
            "Barrier algorithms are based on the assumption that XDIM_REDUCED2=%lu fits into "
            "UINT32_MAXXX=%u. In the last step they call algFLOATvector_SumPartial.",
            rp->XDIM_REDUCED2, UINT32_MAXXX);
        KCTERR(err);
    }
    if(rp->BDIM_REDUCED2 > UINT32_MAXXX)
    {
        err = io::xprintf("BDIM_REDUCED2_ALIGNED * 8=%lu is bigger than UINT32_MAXXX=%u",
                          rp->BDIM_REDUCED2_ALIGNED, UINT32_MAXXX);
        KCTERR(err);
    }
    if(rp->XDIM_ALIGNED > UINT32_MAXXX)
    {
        err = io::xprintf(
            "Size of the volume buffer xdim_aligned=%lu is bigger than UINT32_MAXXX=%u",
            rp->XDIM_ALIGNED, UINT32_MAXXX);
        LOGW << err;
    } else if(rp->XDIM_ALIGNED * 4 > UINT32_MAXXX)
    {
        err = io::xprintf(
            "Byte size of the volume buffer xdim_aligned=%lu is bigger than UINT32_MAXXX=%u",
            4 * rp->XDIM_ALIGNED, UINT32_MAXXX);
        LOGW << err;
    }
    if(rp->BDIM_ALIGNED > UINT32_MAXXX)
    {
        err = io::xprintf(
            "Size of the projection buffer bdim_aligned=%lu is bigger than UINT32_MAXXX=%u",
            rp->BDIM_ALIGNED, UINT32_MAXXX);
        LOGW << err;
    } else if(rp->BDIM_ALIGNED * 4 > UINT32_MAXXX)
    {
        err = io::xprintf("Byte size of the projection buffer bdim_aligned=%lu is bigger than "
                          "UINT32_MAXXX=%u",
                          4 * rp->BDIM_ALIGNED, UINT32_MAXXX);
        LOGW << err;
    }
    reductionParametersSet = true;
}

int AlgorithmsBarrierBuffers::initReductionBuffers()
{
    cl_int err;
    if(!reductionParametersSet)
    {
        KCTERR("First call initReductionParameters.");
    }
    if(algorithmsBuffersInitialized)
    {
        KCTERR("Buffers already initialized, call updateReductionParameters instead.");
    }
    std::shared_ptr<cl::Buffer> red1_buf, red2_buf;
    uint32_t qsize = Q.size();
    std::vector<cl::Memory> mem;
    for(unsigned int QID = 0; QID != qsize; QID++)
    {
        red1_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                rp->BYTESIZE_REDUCED1_MIN, nullptr, &err);
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
            return -1;
        }
        red2_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                rp->BYTESIZE_REDUCED2_MIN, nullptr, &err);
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
            return -1;
        }
        tmp_red1.emplace_back(red1_buf);
        tmp_red2.emplace_back(red2_buf);
        mem.clear();
        mem.emplace_back(*red1_buf);
        mem.emplace_back(*red2_buf);
        Q[QID]->enqueueMigrateMemObjects(mem, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED);
    }
    tmp_red1_bytesize = rp->BYTESIZE_REDUCED1_MIN;
    tmp_red2_bytesize = rp->BYTESIZE_REDUCED2_MIN;
    algorithmsBuffersInitialized = true;
    return 0;
}

int AlgorithmsBarrierBuffers::updateReductionParameters(uint32_t pdimx,
                                                        uint32_t pdimy,
                                                        uint32_t pdimz,
                                                        uint32_t vdimx,
                                                        uint32_t vdimy,
                                                        uint32_t vdimz,
                                                        uint32_t workGroupSize)

{
    cl_int err;
    if(pdimx == rp->pdimx && pdimy == rp->pdimy && pdimz == rp->pdimz && vdimx == rp->vdimx
       && rp->vdimy == vdimy && rp->vdimz == vdimz && rp->workGroupSize == workGroupSize)
    {
        return 0;
    }
    initReductionParameters(pdimx, pdimy, pdimz, vdimx, vdimy, vdimz, workGroupSize);
    if(!algorithmsBuffersInitialized)
    {
        initReductionBuffers();
    } else
    {
        std::shared_ptr<cl::Buffer> red1_buf, red2_buf;
        uint32_t qsize = Q.size();
        std::vector<cl::Memory> mem;
        if(rp->BYTESIZE_REDUCED1_MIN > tmp_red1_bytesize)
        {
            tmp_red1.clear();
            for(unsigned int QID = 0; QID != qsize; QID++)
            {
                red1_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                        rp->BYTESIZE_REDUCED1_MIN, nullptr, &err);
                if(err != CL_SUCCESS)
                {
                    LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!",
                                        err);
                    return -1;
                }
                tmp_red1.emplace_back(red1_buf);
                mem.clear();
                mem.emplace_back(*red1_buf);
                Q[QID]->enqueueMigrateMemObjects(mem, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED);
            }
            tmp_red1_bytesize = rp->BYTESIZE_REDUCED1_MIN;
        }
        if(rp->BYTESIZE_REDUCED2_MIN > tmp_red2_bytesize)
        {
            tmp_red2.clear();
            for(unsigned int QID = 0; QID != qsize; QID++)
            {
                red2_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                        rp->BYTESIZE_REDUCED2_MIN, nullptr, &err);
                if(err != CL_SUCCESS)
                {
                    LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!",
                                        err);
                    return -1;
                }
                tmp_red2.emplace_back(red2_buf);
                mem.clear();
                mem.emplace_back(*red2_buf);
                Q[QID]->enqueueMigrateMemObjects(mem, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED);
            }
            tmp_red2_bytesize = rp->BYTESIZE_REDUCED2_MIN;
        }
    }
    return 0;
}

} // namespace KCT
