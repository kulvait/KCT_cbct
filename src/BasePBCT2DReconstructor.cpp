#include "BasePBCT2DReconstructor.hpp"

namespace KCT {

/**
 * @brief
 *
 * @param projections The b vector to invert.
 * @param volume Allocated memory to store x. Might contain the initial guess.
 *
 * @return
 */
int BasePBCT2DReconstructor::initializeVectors(float* projections,
                                               float* volume,
                                               bool useVolumeAsInitialX0)
{
    this->useVolumeAsInitialX0 = useVolumeAsInitialX0;
    this->b = projections;
    this->x = volume;
    cl_int err;

    // Initialize buffers
    if(useVolumeAsInitialX0)
    {
        x_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                             sizeof(float) * XDIM, (void*)volume, &err);
    } else
    {
        x_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * XDIM,
                                             nullptr, &err);
    }
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
        return -1;
    }
    size_t BDIM_bs = sizeof(float) * uint64_t(BDIM);
    b_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, BDIM_bs,
                                         (void*)projections, &err);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
        return -1;
    }
    tmp_b_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, BDIM_bs, nullptr, &err);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
        return -1;
    }
    return 0;
}

void BasePBCT2DReconstructor::simpleProjection()
{
    Q[0]->enqueueFillBuffer<cl_float>(*tmp_b_buf, FLOATZERO, 0, BDIM * sizeof(float));
    project(*x_buf, *tmp_b_buf);
    Q[0]->enqueueReadBuffer(*tmp_b_buf, CL_TRUE, 0, sizeof(float) * BDIM, b);
}

const float pi = 3.1415927;

void BasePBCT2DReconstructor::simpleBackprojection(BackprojectorScalingMethod scalingType)
{
    Q[0]->enqueueFillBuffer<cl_float>(*x_buf, FLOATZERO, 0, XDIM * sizeof(float));
    float backprojectorScaling = 1.0;
    if(scalingType == BackprojectorScalingMethod::FBPScaling)
    {
        backprojectorScaling = pi / pdimz;
        backproject(*b_buf, *x_buf, 0, 1, backprojectorScaling);
    } else if(scalingType == BackprojectorScalingMethod::NaturalScaling)
    {
        allocateBBuffers(2);
        allocateXBuffers(1);
        std::shared_ptr<cl::Buffer> ones_xbuf = getXBuffer(0);
        std::shared_ptr<cl::Buffer> lengths_bbuf = getBBuffer(0);
        std::shared_ptr<cl::Buffer> bscaled_bbuf = getBBuffer(1);
        Q[0]->enqueueFillBuffer<cl_float>(*ones_xbuf, FLOATONE, 0, XDIM * sizeof(float));
        project(*ones_xbuf, *lengths_bbuf);
        algFLOATvector_invert_except_zero(*lengths_bbuf, BDIM); // Invert for multiplication
        algFLOATvector_C_equals_A_times_B(*b_buf, *lengths_bbuf, *bscaled_bbuf, BDIM);
        backprojectorScaling = 1.0f / pdimz;
        backproject(*bscaled_bbuf, *x_buf, 0, 1, backprojectorScaling);
    } else if(scalingType == BackprojectorScalingMethod::KaczmarzScaling)
    {
        allocateBBuffers(2);
        std::shared_ptr<cl::Buffer> kaczmarz_bbuf =  getBBuffer(0);
        std::shared_ptr<cl::Buffer> bscaled_bbuf = getBBuffer(1);
        kaczmarz_product(*kaczmarz_bbuf);
        algFLOATvector_invert_except_zero(*kaczmarz_bbuf, BDIM); // Invert it for multiplication
        algFLOATvector_C_equals_A_times_B(*b_buf, *kaczmarz_bbuf, *bscaled_bbuf, BDIM);
        backprojectorScaling = 1.0f / pdimz;
        backproject_kaczmarz(*bscaled_bbuf, *x_buf, 0, 1, backprojectorScaling);
    } else// scalingType == BackprojectorScalingMethod::NoScaling
    {
        backproject(*b_buf, *x_buf);
    }
    Q[0]->enqueueReadBuffer(*x_buf, CL_TRUE, 0, sizeof(float) * XDIM, x);
}

void BasePBCT2DReconstructor::writeVolume(cl::Buffer& X, std::string path)
{
    bufferIntoArray(X, x, XDIM);
    bool arrayxmajor = true;
    bool outxmajor = true;
    io::DenFileInfo::create3DDenFileFromArray(x, arrayxmajor, path, io::DenSupportedType::FLOAT32,
                                              vdimx, vdimy, vdimz, outxmajor);
}

void BasePBCT2DReconstructor::writeProjections(cl::Buffer& B, std::string path)
{
    bufferIntoArray(B, b, BDIM);
    bool arrayxmajor = false;
    bool outxmajor = true;
    io::DenFileInfo::create3DDenFileFromArray(b, arrayxmajor, path, io::DenSupportedType::FLOAT32,
                                              pdimx, pdimy, pdimz, outxmajor);
}

void BasePBCT2DReconstructor::setReportingParameters(bool verbose,
                                                     uint32_t reportKthIteration,
                                                     std::string intermediatePrefix)
{
    this->verbose = verbose;
    this->reportKthIteration = reportKthIteration;
    this->intermediatePrefix = intermediatePrefix;
}

double BasePBCT2DReconstructor::adjointProductTest()
{
    std::shared_ptr<cl::Buffer> xa_buf; // X buffers
    allocateXBuffers(1);
    xa_buf = getXBuffer(0);
    allocateBBuffers(1);
    std::shared_ptr<cl::Buffer> ba_buf; // B buffers
    ba_buf = getBBuffer(0);
    project(*x_buf, *ba_buf);
    backproject(*b_buf, *xa_buf);
    double bdotAx = scalarProductBBuffer_barrier_double(*b_buf, *ba_buf);
    double ATbdotx = scalarProductXBuffer_barrier_double(*x_buf, *xa_buf);
    return (bdotAx / ATbdotx);
}
} // namespace KCT
