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

void AlgorithmsBarrierBuffers::initReductionParameters(uint32_t pdimx,
                                                       uint32_t pdimy,
                                                       uint32_t pdimz,
                                                       uint32_t vdimx,
                                                       uint32_t vdimy,
                                                       uint32_t vdimz,
                                                       uint32_t workGroupSize)

{
    this->pdimx = pdimx;
    this->pdimy = pdimy;
    this->pdimz = pdimz;
    this->vdimx = vdimx;
    this->vdimy = vdimy;
    this->vdimz = vdimz;
    this->workGroupSize = workGroupSize;
    const uint32_t UINT32_MAXXX = ((uint32_t)-1);
    const uint64_t xdim = uint64_t(vdimx) * uint64_t(vdimy) * uint64_t(vdimz);
    const uint64_t bdim = uint64_t(pdimx) * uint64_t(pdimy) * uint64_t(pdimz);
    const uint64_t xframesize = uint64_t(vdimx) * uint64_t(vdimy);
    const uint64_t bframesize = uint64_t(pdimx) * uint64_t(pdimy);
    const uint64_t xdim_aligned = xdim + (workGroupSize - xdim % workGroupSize) % workGroupSize;
    const uint64_t bdim_aligned = bdim + (workGroupSize - bdim % workGroupSize) % workGroupSize;
    XDIM = xdim;
    XDIM_ALIGNED = xdim_aligned;
    XDIM_REDUCED1 = xdim_aligned / workGroupSize; // It is divisible by design
    XDIM_REDUCED1_ALIGNED = XDIM_REDUCED1
        + (workGroupSize - XDIM_REDUCED1 % workGroupSize)
            % workGroupSize; // It shall be XDIM_REDUCED1 == XDIM_REDUCED1_ALIGNED
    XDIM_REDUCED2 = XDIM_REDUCED1_ALIGNED / workGroupSize;
    XDIM_REDUCED2_ALIGNED
        = XDIM_REDUCED2 + (workGroupSize - XDIM_REDUCED2 % workGroupSize) % workGroupSize;
    BDIM = bdim;
    BDIM_ALIGNED = bdim_aligned;
    BDIM_REDUCED1 = bdim_aligned / workGroupSize;
    BDIM_REDUCED1_ALIGNED
        = BDIM_REDUCED1 + (workGroupSize - BDIM_REDUCED1 % workGroupSize) % workGroupSize;
    BDIM_REDUCED2 = BDIM_REDUCED1_ALIGNED / workGroupSize;
    BDIM_REDUCED2_ALIGNED
        = BDIM_REDUCED2 + (workGroupSize - BDIM_REDUCED2 % workGroupSize) % workGroupSize;
    std::string err;
    if(xframesize > UINT32_MAXXX)
    {
        err = io::xprintf(
            "Algorithms are based on the assumption that the x y volume slice can be "
            "indexed by uint32_t but xframesize=%lu that is bigger than UINT32_MAXXX=%lu",
            xframesize, UINT32_MAXXX);
        KCTERR(err);
    }
    if(bframesize > UINT32_MAXXX)
    {
        err = io ::xprintf(
            "Algorithms are based on the assumption that the projection size can be "
            "indexed by uint32_t but bframesize=%lu that is bigger than UINT32_MAXXX=%lu",
            bframesize, UINT32_MAXXX);
        KCTERR(err);
    }
    if(XDIM_REDUCED2 > UINT32_MAXXX)
    {
        err = io::xprintf(
            "Barrier algorithms are based on the assumption that XDIM_REDUCED2=%lu fits into "
            "UINT32_MAXXX=%u. In the last step they call algFLOATvector_SumPartial.",
            XDIM_REDUCED2, UINT32_MAXXX);
        KCTERR(err);
    }
    if(BDIM_REDUCED2 > UINT32_MAXXX)
    {
        err = io::xprintf("BDIM_REDUCED2_ALIGNED * 8=%lu is bigger than UINT32_MAXXX=%u",
                          BDIM_REDUCED2_ALIGNED, UINT32_MAXXX);
        KCTERR(err);
    }
    if(xdim_aligned > UINT32_MAXXX)
    {
        err = io::xprintf(
            "Size of the volume buffer xdim_aligned=%lu is bigger than UINT32_MAXXX=%u",
            xdim_aligned, UINT32_MAXXX);
        LOGW << err;
    } else if(xdim_aligned * 4 > UINT32_MAXXX)
    {
        err = io::xprintf(
            "Byte size of the volume buffer xdim_aligned=%lu is bigger than UINT32_MAXXX=%u",
            4 * xdim_aligned, UINT32_MAXXX);
        LOGW << err;
    }
    if(bdim_aligned > UINT32_MAXXX)
    {
        err = io::xprintf(
            "Size of the projection buffer bdim_aligned=%lu is bigger than UINT32_MAXXX=%u",
            bdim_aligned, UINT32_MAXXX);
        LOGW << err;
    } else if(bdim_aligned * 4 > UINT32_MAXXX)
    {
        err = io::xprintf("Byte size of the projection buffer bdim_aligned=%lu is bigger than "
                          "UINT32_MAXXX=%u",
                          4 * bdim_aligned, UINT32_MAXXX);
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
        tmp_red1_bytesize = sizeof(double) * XDIM_REDUCED1;
        tmp_red2_bytesize = sizeof(double) * XDIM_REDUCED2;
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
        tmp_red1_bytesize = sizeof(double) * BDIM_REDUCED1;
        tmp_red2_bytesize = sizeof(double) * BDIM_REDUCED2;
    }
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
    if(pdimx == this->pdimx && pdimy == this->pdimy && pdimz == this->pdimz && vdimx == this->vdimx
       && this->vdimy == vdimy && this->vdimz == vdimz && this->workGroupSize == workGroupSize)
    {
        return 0;
    }
    initReductionParameters(pdimx, pdimy, pdimz, vdimx, vdimy, vdimz, workGroupSize);
    if(!algorithmsBuffersInitialized)
    {
        initReductionBuffers();
    } else
    {
        if(XDIM > BDIM)
        {
            if(sizeof(double) * XDIM_REDUCED1 > tmp_red1_bytesize)
            {
                tmp_x_red1 = std::make_shared<cl::Buffer>(
                    *context, CL_MEM_READ_WRITE, sizeof(double) * XDIM_REDUCED1, nullptr, &err);
                if(err != CL_SUCCESS)
                {
                    LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!",
                                        err);
                    return -1;
                }
                tmp_b_red1 = tmp_x_red1;
                tmp_red1_bytesize = sizeof(double) * XDIM_REDUCED1;
            }
            if(sizeof(double) * XDIM_REDUCED2 > tmp_red2_bytesize)
            {
                tmp_x_red2 = std::make_shared<cl::Buffer>(
                    *context, CL_MEM_READ_WRITE, sizeof(double) * XDIM_REDUCED2, nullptr, &err);
                if(err != CL_SUCCESS)
                {
                    LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!",
                                        err);
                    return -1;
                }
                tmp_b_red2 = tmp_x_red2;
                tmp_red2_bytesize = sizeof(double) * XDIM_REDUCED2;
            }
        } else
        {
            if(sizeof(double) * BDIM_REDUCED1 > tmp_red1_bytesize)
            {

                tmp_b_red1 = std::make_shared<cl::Buffer>(
                    *context, CL_MEM_READ_WRITE, sizeof(double) * BDIM_REDUCED1, nullptr, &err);
                if(err != CL_SUCCESS)
                {
                    LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!",
                                        err);
                    return -1;
                }
                tmp_x_red1 = tmp_b_red1;
                tmp_red1_bytesize = sizeof(double) * BDIM_REDUCED1;
            }
            if(sizeof(double) * BDIM_REDUCED2 > tmp_red2_bytesize)
            {
                tmp_b_red2 = std::make_shared<cl::Buffer>(
                    *context, CL_MEM_READ_WRITE, sizeof(double) * BDIM_REDUCED2, nullptr, &err);
                if(err != CL_SUCCESS)
                {
                    LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!",
                                        err);
                    return -1;
                }
                tmp_x_red2 = tmp_b_red2;
                tmp_red2_bytesize = sizeof(double) * BDIM_REDUCED2;
            }
        }
    }
    return 0;
}

} // namespace KCT
