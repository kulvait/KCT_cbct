#include "BaseReconstructor.hpp"

namespace CTL {

void BaseReconstructor::initializeCVPProjector(bool useExactScaling)
{
    if(!openCLinitialized)
    {
        useCVPProjector = true;
        exactProjectionScaling = useExactScaling;
        useSidonProjector = false;
        pixelGranularity = { 1, 1 };
        useTTProjector = false;
    } else
    {
        std::string err = "Could not initialize projector when OpenCL was already initialized.";
        LOGE << err;
        throw std::runtime_error(err.c_str());
    }
}

void BaseReconstructor::initializeSidonProjector(uint32_t probesPerEdgeX, uint32_t probesPerEdgeY)
{
    if(!openCLinitialized)
    {
        useSidonProjector = true;
        pixelGranularity = { probesPerEdgeX, probesPerEdgeY };
        useCVPProjector = false;
        exactProjectionScaling = false;
        useTTProjector = false;
    } else
    {
        std::string err = "Could not initialize projector when OpenCL was already initialized.";
        LOGE << err;
        throw std::runtime_error(err.c_str());
    }
}

void BaseReconstructor::initializeTTProjector()
{
    if(!openCLinitialized)
    {
        useTTProjector = true;
        useCVPProjector = false;
        exactProjectionScaling = false;
        useSidonProjector = false;
        pixelGranularity = { 1, 1 };
    } else
    {
        std::string err = "Could not initialize projector when OpenCL was already initialized.";
        LOGE << err;
        throw std::runtime_error(err.c_str());
    }
}

int BaseReconstructor::initializeOpenCL(std::string xpath, uint32_t platformId, bool debug)
{
    // Select the first available platform.
    std::shared_ptr<cl::Platform> platform = util::OpenCLManager::getPlatform(platformId, true);
    if(platform == nullptr)
    {
        return -1;
    }
    // Select the first available device for given platform
    device = util::OpenCLManager::getDevice(*platform, 0, true);
    if(device == nullptr)
    {
        return -2;
    }
    cl::Context tmp({ *device });
    context = std::make_shared<cl::Context>(tmp);

    // Debug info
    // https://software.intel.com/en-us/openclsdk-devguide-enabling-debugging-in-opencl-runtime
    std::string clFile;
    std::string sourceText;
    // clFile = io::xprintf("%s/opencl/centerVoxelProjector.cl", xpath.c_str());
    clFile = io::xprintf("%s/opencl/allsources.cl", xpath.c_str());
    if(useSidonProjector)
    {
        io::concatenateTextFiles(
            clFile, true,
            { io::xprintf("%s/opencl/utils.cl", xpath.c_str()),
              io::xprintf("%s/opencl/projector_sidon.cl", xpath.c_str()),
              io::xprintf("%s/opencl/backprojector_sidon.cl", xpath.c_str()) });
    } else if(useTTProjector)
    {
        io::concatenateTextFiles(clFile, true,
                                 { io::xprintf("%s/opencl/utils.cl", xpath.c_str()),
                                   io::xprintf("%s/opencl/projector.cl", xpath.c_str()),
                                   io::xprintf("%s/opencl/backprojector.cl", xpath.c_str()),
                                   io::xprintf("%s/opencl/projector_tt.cl", xpath.c_str()),
                                   io::xprintf("%s/opencl/backprojector_tt.cl", xpath.c_str()) });
    } else
    {
        io::concatenateTextFiles(clFile, true,
                                 { io::xprintf("%s/opencl/utils.cl", xpath.c_str()),
                                   io::xprintf("%s/opencl/projector.cl", xpath.c_str()),
                                   io::xprintf("%s/opencl/backprojector.cl", xpath.c_str()) });
    }
    std::string projectorSource = io::fileToString(clFile);
    cl::Program program(*context, projectorSource);
    LOGI << io::xprintf("Building file %s.", clFile.c_str());
    if(debug)
    {
        std::string options = io::xprintf("-g -s \"%s\"", clFile.c_str());
        if(program.build({ *device }, options.c_str()) != CL_SUCCESS)
        {
            LOGE << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(*device);
            return -3;
        }
    } else
    {
        if(program.build({ *device }) != CL_SUCCESS)
        {
            LOGE << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(*device);
            return -3;
        }
    }
    // OpenCL 1.2 got rid of KernelFunctor
    // https://forums.khronos.org/showthread.php/8317-cl-hpp-KernelFunctor-gone-replaced-with-KernelFunctorGlobal
    // https://stackoverflow.com/questions/23992369/what-should-i-use-instead-of-clkernelfunctor/54344990#54344990
    //    FLOATcutting_voxel_project
    //        = std::make_shared<cl::make_kernel<cl::Buffer&, cl::Buffer&, cl_double16&,
    //        cl_double4&,
    //                                           cl_double4&, cl_int4&, cl_double4&, cl_int2&,
    //                                           float&>>(
    //            cl::Kernel(program, "FLOATcutting_voxel_project"));
    FLOAT_scaleVector = std::make_shared<cl::make_kernel<cl::Buffer&, float&>>(
        cl::Kernel(program, "FLOAT_scale_vector"));
    FLOAT_addIntoFirstVectorSecondVectorScaled
        = std::make_shared<cl::make_kernel<cl::Buffer&, cl::Buffer&, float&>>(
            cl::Kernel(program, "FLOAT_add_into_first_vector_second_vector_scaled"));
    FLOAT_addIntoFirstVectorScaledSecondVector
        = std::make_shared<cl::make_kernel<cl::Buffer&, cl::Buffer&, float&>>(
            cl::Kernel(program, "FLOAT_add_into_first_vector_scaled_second_vector"));
    FLOAT_NormSquare = std::make_shared<cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&>>(
        cl::Kernel(program, "FLOATvector_NormSquarePartial"));
    FLOAT_SumPartial = std::make_shared<cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&>>(
        cl::Kernel(program, "FLOATvector_SumPartial"));
    FLOAT_NormSquare_barier = std::make_shared<
        cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::LocalSpaceArg&, unsigned int&>>(
        cl::Kernel(program, "FLOATvector_NormSquarePartial_barier"));
    FLOAT_Sum_barier = std::make_shared<
        cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::LocalSpaceArg&, unsigned int&>>(
        cl::Kernel(program, "FLOATvector_SumPartial_barier"));
    NormSquare = std::make_shared<cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&>>(
        cl::Kernel(program, "vector_NormSquarePartial"));
    SumPartial = std::make_shared<cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&>>(
        cl::Kernel(program, "vector_SumPartial"));
    NormSquare_barier = std::make_shared<
        cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::LocalSpaceArg&, unsigned int&>>(
        cl::Kernel(program, "vector_NormSquarePartial_barier"));
    Sum_barier = std::make_shared<
        cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::LocalSpaceArg&, unsigned int&>>(
        cl::Kernel(program, "vector_SumPartial_barier"));
    FLOAT_CopyVector = std::make_shared<cl::make_kernel<cl::Buffer&, cl::Buffer&>>(
        cl::Kernel(program, "FLOAT_copy_vector"));
    FLOAT_CopyVector_offset
        = std::make_shared<cl::make_kernel<cl::Buffer&, unsigned int&, cl::Buffer&, unsigned int&>>(
            cl::Kernel(program, "FLOAT_copy_vector_offset"));
    ScalarProductPartial_barier = std::make_shared<
        cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::LocalSpaceArg&, unsigned int&>>(
        cl::Kernel(program, "vector_ScalarProductPartial_barier"));
    if(useSidonProjector)
    {
        FLOATprojector_sidon = std::make_shared<
            cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&, cl_double16&, cl_double3&,
                            cl_double3&, cl_int3&, cl_double3&, cl_int2&, float&, cl_uint2&>>(
            cl::Kernel(program, "FLOATsidon_project"));
        FLOATbackprojector_sidon = std::make_shared<
            cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&, cl_double16&, cl_double3&,
                            cl_double3&, cl_int3&, cl_double3&, cl_int2&, float&, cl_uint2&>>(
            cl::Kernel(program, "FLOATsidon_backproject"));
    } else if(useTTProjector)
    {
        FLOATta3_project = std::make_shared<
            cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&, cl_double16&, cl_double3&,
                            cl_double3&, cl_int3&, cl_double3&, cl_int2&, float&>>(
            cl::Kernel(program, "FLOATtta3_project"));
        FLOATta3_backproject = std::make_shared<
            cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&, cl_double16&, cl_double3&,
                            cl_double3&, cl_int3&, cl_double3&, cl_int2&, float&>>(
            cl::Kernel(program, "FLOATtta3_backproject"));
    } else
    {
        FLOATcutting_voxel_project = std::make_shared<
            cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&, cl_double16&, cl_double3&,
                            cl_double3&, cl_int3&, cl_double3&, cl_int2&, float&>>(
            cl::Kernel(program, "FLOATcutting_voxel_project"));
        FLOATcutting_voxel_backproject = std::make_shared<
            cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&, cl_double16&, cl_double3&,
                            cl_double3&, cl_int3&, cl_double3&, cl_int2&, float&>>(
            cl::Kernel(program, "FLOATcutting_voxel_backproject"));
        scalingProjectionsCos
            = std::make_shared<cl::make_kernel<cl::Buffer&, unsigned int&, cl_double16&,
                                               cl_double3&, cl_double3&, cl_uint2&, float&>>(
                cl::Kernel(program, "FLOATrescale_projections_cos"));
        scalingProjectionsExact
            = std::make_shared<cl::make_kernel<cl::Buffer&, unsigned int&, cl_uint2&, cl_double2&,
                                               cl_double2&, double&>>(
                cl::Kernel(program, "FLOATrescale_projections_exact"));
    }
    Q = std::make_shared<cl::CommandQueue>(*context, *device);
    return 0;
}

/**
 * @brief
 *
 * @param projections The b vector to invert.
 * @param volume Allocated memory to store x. Might contain the initial guess.
 *
 * @return
 */
int BaseReconstructor::initializeVectors(float* projections, float* volume)
{
    this->b = projections;
    this->x = volume;
    cl_int err;

    // Initialize buffers x_buf, v_buf and v_buf by zeros
    x_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                         sizeof(float) * XDIM, (void*)volume, &err);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
        return -1;
    }

    b_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         sizeof(float) * BDIM, (void*)projections, &err);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
        return -1;
    }

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

    tmp_b_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * BDIM,
                                             nullptr, &err);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
        return -1;
    }

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
    return 0;
}

int BaseReconstructor::allocateXBuffers(uint32_t xBufferCount)
{
    cl_int err;
    std::shared_ptr<cl::Buffer> x_buf;
    while(this->x_buffers.size() < xBufferCount)
    {
        x_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * XDIM,
                                             nullptr, &err);
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of X buffer with error code %d!", err);
            return -1;
        }
        x_buffers.push_back(x_buf);
    }
    return 0;
}

int BaseReconstructor::allocateBBuffers(uint32_t bBufferCount)
{
    cl_int err;
    std::shared_ptr<cl::Buffer> b_buf;
    while(this->b_buffers.size() < bBufferCount)
    {
        b_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * BDIM,
                                             nullptr, &err);
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of B buffer with error code %d!", err);
            return -1;
        }
        b_buffers.push_back(b_buf);
    }
    return 0;
}

int BaseReconstructor::allocateTmpXBuffers(uint32_t xBufferCount)
{
    cl_int err;
    std::shared_ptr<cl::Buffer> x_buf;
    while(this->tmp_x_buffers.size() < xBufferCount)
    {
        x_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * XDIM,
                                             nullptr, &err);
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of X buffer with error code %d!", err);
            return -1;
        }
        tmp_x_buffers.push_back(x_buf);
    }
    return 0;
}

int BaseReconstructor::allocateTmpBBuffers(uint32_t bBufferCount)
{
    cl_int err;
    std::shared_ptr<cl::Buffer> b_buf;
    while(this->tmp_b_buffers.size() < bBufferCount)
    {
        b_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * BDIM,
                                             nullptr, &err);
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of B buffer with error code %d!", err);
            return -1;
        }
        tmp_b_buffers.push_back(b_buf);
    }
    return 0;
}

std::shared_ptr<cl::Buffer> BaseReconstructor::getXBuffer(uint32_t i)
{
    if(i < x_buffers.size())
    {
        return x_buffers[i];
    } else
    {
        std::string err = io::xprintf(
            "Index %d is out of range of the x_buffers vector of size %d!", i, x_buffers.size());
        LOGE << err;
        throw std::runtime_error(err);
    }
}

std::shared_ptr<cl::Buffer> BaseReconstructor::getBBuffer(uint32_t i)
{
    if(i < b_buffers.size())
    {
        return b_buffers[i];
    } else
    {
        std::string err = io::xprintf(
            "Index %d is out of range of the b_buffers vector of size %d!", i, b_buffers.size());
        LOGE << err;
        throw std::runtime_error(err);
    }
}

std::shared_ptr<cl::Buffer> BaseReconstructor::getTmpXBuffer(uint32_t i)
{
    if(i < tmp_x_buffers.size())
    {
        return tmp_x_buffers[i];
    } else
    {
        std::string err
            = io::xprintf("Index %d is out of range of the tmp_x_buffers vector of size %d!", i,
                          tmp_x_buffers.size());
        LOGE << err;
        throw std::runtime_error(err);
    }
}

std::shared_ptr<cl::Buffer> BaseReconstructor::getTmpBBuffer(uint32_t i)
{
    if(i < tmp_b_buffers.size())
    {
        return tmp_b_buffers[i];
    } else
    {
        std::string err
            = io::xprintf("Index %d is out of range of the tmp_b_buffers vector of size %d!", i,
                          tmp_b_buffers.size());
        LOGE << err;
        throw std::runtime_error(err);
    }
}

int BaseReconstructor::copyFloatVectorOffset(cl::Buffer& from,
                                             unsigned int from_offset,
                                             cl::Buffer& to,
                                             unsigned int to_offset,
                                             unsigned int size)
{
    cl::EnqueueArgs eargs(*Q, cl::NDRange(size));
    (*FLOAT_CopyVector_offset)(eargs, from, from_offset, to, to_offset).wait();
    return 0;
}

/**
 * Funcion computes the norm of the Buffer that has vdimx * vdimy * vdimz elements.
 *
 * @param X
 *
 * @return
 */
double BaseReconstructor::normXBuffer_frame_double(cl::Buffer& X)
{
    double sum;
    uint32_t framesize = vdimx * vdimy;
    cl::EnqueueArgs eargs1(*Q, cl::NDRange(vdimz));
    (*NormSquare)(eargs1, X, *tmp_x_red1, framesize).wait();
    cl::EnqueueArgs eargs(*Q, cl::NDRange(1));
    unsigned int arg = vdimz;
    (*SumPartial)(eargs, *tmp_x_red1, *tmp_x_red2, arg).wait();
    Q->enqueueReadBuffer(*tmp_x_red2, CL_TRUE, 0, sizeof(double), &sum);
    return sum;
}

/**
 * Funcion computes the norm of the Buffer that has vdimx * vdimy * vdimz elements.
 *
 * @param X
 *
 * @return
 */
double BaseReconstructor::normXBuffer_barier_double(cl::Buffer& X)
{

    double sum;
    cl::EnqueueArgs eargs_red1(*Q, cl::NDRange(XDIM_ALIGNED), cl::NDRange(workGroupSize));
    cl::LocalSpaceArg localsize = cl::Local(workGroupSize * sizeof(double));
    (*NormSquare_barier)(eargs_red1, X, *tmp_x_red1, localsize, XDIM).wait();
    cl::EnqueueArgs eargs_red2(*Q, cl::NDRange(XDIM_REDUCED1_ALIGNED), cl::NDRange(workGroupSize));
    (*Sum_barier)(eargs_red2, *tmp_x_red1, *tmp_x_red2, localsize, XDIM_REDUCED1).wait();
    cl::EnqueueArgs eargs(*Q, cl::NDRange(1));
    (*SumPartial)(eargs, *tmp_x_red2, *tmp_x_red1, XDIM_REDUCED2).wait();
    Q->enqueueReadBuffer(*tmp_x_red1, CL_TRUE, 0, sizeof(double), &sum);
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
double BaseReconstructor::scalarProductXBuffer_barier_double(cl::Buffer& A, cl::Buffer& B)
{

    double sum;
    cl::EnqueueArgs eargs_red1(*Q, cl::NDRange(XDIM_ALIGNED), cl::NDRange(workGroupSize));
    cl::LocalSpaceArg localsize = cl::Local(workGroupSize * sizeof(double));
    (*ScalarProductPartial_barier)(eargs_red1, A, B, *tmp_x_red1, localsize, XDIM).wait();
    cl::EnqueueArgs eargs_red2(*Q, cl::NDRange(XDIM_REDUCED1_ALIGNED), cl::NDRange(workGroupSize));
    (*Sum_barier)(eargs_red2, *tmp_x_red1, *tmp_x_red2, localsize, XDIM_REDUCED1).wait();
    cl::EnqueueArgs eargs(*Q, cl::NDRange(1));
    (*SumPartial)(eargs, *tmp_x_red2, *tmp_x_red1, XDIM_REDUCED2).wait();
    Q->enqueueReadBuffer(*tmp_x_red1, CL_TRUE, 0, sizeof(double), &sum);
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
double BaseReconstructor::scalarProductBBuffer_barier_double(cl::Buffer& A, cl::Buffer& B)
{

    double sum;
    cl::EnqueueArgs eargs_red1(*Q, cl::NDRange(BDIM_ALIGNED), cl::NDRange(workGroupSize));
    cl::LocalSpaceArg localsize = cl::Local(workGroupSize * sizeof(double));
    (*ScalarProductPartial_barier)(eargs_red1, A, B, *tmp_b_red1, localsize, BDIM).wait();
    cl::EnqueueArgs eargs_red2(*Q, cl::NDRange(BDIM_REDUCED1_ALIGNED), cl::NDRange(workGroupSize));
    (*Sum_barier)(eargs_red2, *tmp_b_red1, *tmp_b_red2, localsize, BDIM_REDUCED1).wait();
    cl::EnqueueArgs eargs(*Q, cl::NDRange(1));
    (*SumPartial)(eargs, *tmp_b_red2, *tmp_b_red1, BDIM_REDUCED2).wait();
    Q->enqueueReadBuffer(*tmp_b_red1, CL_TRUE, 0, sizeof(double), &sum);
    return sum;
}

/**
 * Funcion computes the norm of the Buffer that has pdimx * pdimy * pdimz elements.
 *
 * @param X
 *
 * @return
 */
double BaseReconstructor::normBBuffer_frame_double(cl::Buffer& B)
{ // Use workGroupSize that is private constant default to 256
    double sum;
    uint32_t framesize = pdimx * pdimy;
    cl::EnqueueArgs eargs1(*Q, cl::NDRange(pdimz));
    (*NormSquare)(eargs1, B, *tmp_b_red1, framesize).wait();
    cl::EnqueueArgs eargs(*Q, cl::NDRange(1));
    unsigned int arg = pdimz;
    (*SumPartial)(eargs, *tmp_b_red1, *tmp_b_red2, arg).wait();
    Q->enqueueReadBuffer(*tmp_b_red2, CL_TRUE, 0, sizeof(double), &sum);
    return sum;
}

/**
 * Funcion computes the norm of the Buffer that has pdimx * pdimy * pdimz elements.
 *
 * @param X
 *
 * @return
 */
double BaseReconstructor::normBBuffer_barier_double(cl::Buffer& B)
{ // Use workGroupSize that is private constant default to 256
    double sum;
    cl::EnqueueArgs eargs_red1(*Q, cl::NDRange(BDIM_ALIGNED), cl::NDRange(workGroupSize));
    cl::LocalSpaceArg localsize = cl::Local(workGroupSize * sizeof(double));
    (*NormSquare_barier)(eargs_red1, B, *tmp_b_red1, localsize, BDIM).wait();
    cl::EnqueueArgs eargs_red2(*Q, cl::NDRange(BDIM_REDUCED1_ALIGNED), cl::NDRange(workGroupSize));
    (*Sum_barier)(eargs_red2, *tmp_b_red1, *tmp_b_red2, localsize, BDIM_REDUCED1).wait();
    cl::EnqueueArgs eargs(*Q, cl::NDRange(1));
    (*SumPartial)(eargs, *tmp_b_red2, *tmp_b_red1, BDIM_REDUCED2).wait();
    Q->enqueueReadBuffer(*tmp_b_red1, CL_TRUE, 0, sizeof(double), &sum);
    return sum;
}

/**
 * Funcion computes the norm of the Buffer that has vdimx * vdimy * vdimz elements.
 *
 * @param X
 *
 * @return
 */
float BaseReconstructor::normXBuffer_frame(cl::Buffer& X)
{
    float sum;
    uint32_t framesize = vdimx * vdimy;
    cl::EnqueueArgs eargs1(*Q, cl::NDRange(vdimz));
    (*FLOAT_NormSquare)(eargs1, X, *tmp_x_red1, framesize).wait();
    cl::EnqueueArgs eargs(*Q, cl::NDRange(1));
    unsigned int arg = vdimz;
    (*FLOAT_SumPartial)(eargs, *tmp_x_red1, *tmp_x_red2, arg).wait();
    Q->enqueueReadBuffer(*tmp_x_red2, CL_TRUE, 0, sizeof(float), &sum);
    return sum;
}

/**
 * Funcion computes the norm of the Buffer that has vdimx * vdimy * vdimz elements.
 *
 * @param X
 *
 * @return
 */
float BaseReconstructor::normXBuffer_barier(cl::Buffer& X)
{

    float sum;
    cl::EnqueueArgs eargs_red1(*Q, cl::NDRange(XDIM_ALIGNED), cl::NDRange(workGroupSize));
    cl::LocalSpaceArg localsize = cl::Local(workGroupSize * sizeof(float));
    (*FLOAT_NormSquare_barier)(eargs_red1, X, *tmp_x_red1, localsize, XDIM).wait();
    cl::EnqueueArgs eargs_red2(*Q, cl::NDRange(XDIM_REDUCED1_ALIGNED), cl::NDRange(workGroupSize));
    (*FLOAT_Sum_barier)(eargs_red2, *tmp_x_red1, *tmp_x_red2, localsize, XDIM_REDUCED1).wait();
    cl::EnqueueArgs eargs(*Q, cl::NDRange(1));
    (*FLOAT_SumPartial)(eargs, *tmp_x_red2, *tmp_x_red1, XDIM_REDUCED2).wait();
    Q->enqueueReadBuffer(*tmp_x_red1, CL_TRUE, 0, sizeof(float), &sum);
    return sum;
}

/**
 * Funcion computes the norm of the Buffer that has pdimx * pdimy * pdimz elements.
 *
 * @param X
 *
 * @return
 */
float BaseReconstructor::normBBuffer_frame(cl::Buffer& B)
{ // Use workGroupSize that is private constant default to 256
    float sum;
    uint32_t framesize = pdimx * pdimy;
    cl::EnqueueArgs eargs1(*Q, cl::NDRange(pdimz));
    (*FLOAT_NormSquare)(eargs1, B, *tmp_b_red1, framesize).wait();
    cl::EnqueueArgs eargs(*Q, cl::NDRange(1));
    unsigned int arg = pdimz;
    (*FLOAT_SumPartial)(eargs, *tmp_b_red1, *tmp_b_red2, arg).wait();
    Q->enqueueReadBuffer(*tmp_b_red2, CL_TRUE, 0, sizeof(float), &sum);
    return sum;
}

/**
 * Funcion computes the norm of the Buffer that has pdimx * pdimy * pdimz elements.
 *
 * @param X
 *
 * @return
 */
float BaseReconstructor::normBBuffer_barier(cl::Buffer& B)
{ // Use workGroupSize that is private constant default to 256
    float sum;
    cl::EnqueueArgs eargs_red1(*Q, cl::NDRange(BDIM_ALIGNED), cl::NDRange(workGroupSize));
    cl::LocalSpaceArg localsize = cl::Local(workGroupSize * sizeof(float));
    (*FLOAT_NormSquare_barier)(eargs_red1, B, *tmp_b_red1, localsize, BDIM).wait();
    cl::EnqueueArgs eargs_red2(*Q, cl::NDRange(BDIM_REDUCED1_ALIGNED), cl::NDRange(workGroupSize));
    (*FLOAT_Sum_barier)(eargs_red2, *tmp_b_red1, *tmp_b_red2, localsize, BDIM_REDUCED1).wait();
    cl::EnqueueArgs eargs(*Q, cl::NDRange(1));
    (*FLOAT_SumPartial)(eargs, *tmp_b_red2, *tmp_b_red1, BDIM_REDUCED2).wait();
    Q->enqueueReadBuffer(*tmp_b_red1, CL_TRUE, 0, sizeof(float), &sum);
    return sum;
}

int BaseReconstructor::backproject(cl::Buffer& B,
                                   cl::Buffer& X,
                                   std::vector<matrix::ProjectionMatrix>& V,
                                   std::vector<cl_double16>& invertedProjectionMatrices,
                                   std::vector<float>& scalingFactors)
{
    Q->enqueueFillBuffer<cl_float>(X, FLOATZERO, 0, XDIM * sizeof(float));
    unsigned int frameSize = pdimx * pdimy;
    copyFloatVector(B, *tmp_b_buf, BDIM);
    cl::EnqueueArgs eargs(*Q, cl::NDRange(vdimz, vdimy, vdimx));
    cl::EnqueueArgs eargs2(*Q, cl::NDRange(pdimx, pdimy));
    double normalProjectionX, normalProjectionY, projection45X, projection45Y, fX, fY;
    double sourceToDetector;
    for(std::size_t i = 0; i != pdimz; i++)
    {
        matrix::ProjectionMatrix mat = V[i];
        float scalingFactor = scalingFactors[i];
        std::array<double, 3> sourcePosition = mat.sourcePosition();
        std::array<double, 3> normalToDetector = mat.normalToDetector();
        std::array<double, 3> tangentToDetector = mat.tangentToDetectorYDirection();
        mat.project(sourcePosition[0] - normalToDetector[0] + tangentToDetector[0],
                    sourcePosition[1] - normalToDetector[1] + tangentToDetector[1],
                    sourcePosition[2] - normalToDetector[2] + tangentToDetector[2], &projection45X,
                    &projection45Y);
        mat.project(
            sourcePosition[0] - normalToDetector[0], sourcePosition[1] - normalToDetector[1],
            sourcePosition[2] - normalToDetector[2], &normalProjectionX, &normalProjectionY);
        fX = (projection45X - normalProjectionX) * pixelSpacingX;
        fY = (projection45Y - normalProjectionY) * pixelSpacingY;
        sourceToDetector = std::sqrt(fX * fX + fY * fY);
        cl_double2 normalProjection({ normalProjectionX, normalProjectionY });
        double* P = mat.getPtr();
        cl_double16 PM({ P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10], P[11],
                         0.0, 0.0, 0.0, 0.0 });
        cl_double16 ICM = invertedProjectionMatrices[i];
        cl_double3 SOURCEPOSITION({ sourcePosition[0], sourcePosition[1], sourcePosition[2] });
        cl_double3 NORMALTODETECTOR(
            { normalToDetector[0], normalToDetector[1], normalToDetector[2] });
        unsigned int offset = i * frameSize;
        if(useSidonProjector)
        {
            (*FLOATbackprojector_sidon)(eargs2, X, *tmp_b_buf, offset, ICM, SOURCEPOSITION,
                                        NORMALTODETECTOR, vdims, voxelSizes, pdims, FLOATONE,
                                        pixelGranularity);
        } else if(useTTProjector)
        {
            (*FLOATta3_backproject)(eargs, X, *tmp_b_buf, offset, PM, SOURCEPOSITION,
                                    NORMALTODETECTOR, vdims, voxelSizes, pdims, FLOATONE);
        } else
        {
            if(exactProjectionScaling)
            {
                (*scalingProjectionsExact)(eargs2, *tmp_b_buf, offset, pdims_uint, normalProjection,
                                           pixelSizes, sourceToDetector);
            } else
            {
                (*scalingProjectionsCos)(eargs2, *tmp_b_buf, offset, ICM, SOURCEPOSITION,
                                         NORMALTODETECTOR, pdims_uint, scalingFactor);
            }
            (*FLOATcutting_voxel_backproject)(eargs, X, *tmp_b_buf, offset, PM, SOURCEPOSITION,
                                              NORMALTODETECTOR, vdims, voxelSizes, pdims, FLOATONE);
        }
    }
    return 0;
}

int BaseReconstructor::project(cl::Buffer& X,
                               cl::Buffer& B,
                               std::vector<matrix::ProjectionMatrix>& V,
                               std::vector<cl_double16>& invertedProjectionMatrices,
                               std::vector<float>& scalingFactors)
{
    Q->enqueueFillBuffer<cl_float>(B, FLOATZERO, 0, BDIM * sizeof(float));
    unsigned int frameSize = pdimx * pdimy;
    cl::EnqueueArgs eargs(*Q, cl::NDRange(vdimz, vdimy, vdimx));
    cl::EnqueueArgs eargs2(*Q, cl::NDRange(pdimx, pdimy));
    double normalProjectionX, normalProjectionY, projection45X, projection45Y, fX, fY;
    double sourceToDetector;
    for(std::size_t i = 0; i != pdimz; i++)
    {
        matrix::ProjectionMatrix mat = V[i];
        float scalingFactor = scalingFactors[i];
        std::array<double, 3> sourcePosition = mat.sourcePosition();
        std::array<double, 3> normalToDetector = mat.normalToDetector();
        std::array<double, 3> tangentToDetector = mat.tangentToDetectorYDirection();
        mat.project(sourcePosition[0] - normalToDetector[0] + tangentToDetector[0],
                    sourcePosition[1] - normalToDetector[1] + tangentToDetector[1],
                    sourcePosition[2] - normalToDetector[2] + tangentToDetector[2], &projection45X,
                    &projection45Y);
        mat.project(
            sourcePosition[0] - normalToDetector[0], sourcePosition[1] - normalToDetector[1],
            sourcePosition[2] - normalToDetector[2], &normalProjectionX, &normalProjectionY);
        fX = (projection45X - normalProjectionX) * pixelSpacingX;
        fY = (projection45Y - normalProjectionY) * pixelSpacingY;
        sourceToDetector = std::sqrt(fX * fX + fY * fY);
        cl_double2 normalProjection({ normalProjectionX, normalProjectionY });
        double* P = mat.getPtr();
        cl_double16 PM({ P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10], P[11],
                         0.0, 0.0, 0.0, 0.0 });
        cl_double16 ICM = invertedProjectionMatrices[i];
        cl_double3 SOURCEPOSITION({ sourcePosition[0], sourcePosition[1], sourcePosition[2] });
        cl_double3 NORMALTODETECTOR(
            { normalToDetector[0], normalToDetector[1], normalToDetector[2] });
        unsigned int offset = i * frameSize;
        if(useSidonProjector)
        {
            (*FLOATprojector_sidon)(eargs2, X, B, offset, ICM, SOURCEPOSITION, NORMALTODETECTOR,
                                    vdims, voxelSizes, pdims, FLOATONE, pixelGranularity);
        } else if(useTTProjector)
        {
            (*FLOATta3_project)(eargs, X, B, offset, PM, SOURCEPOSITION, NORMALTODETECTOR, vdims,
                                voxelSizes, pdims, FLOATONE);
        } else
        {
            (*FLOATcutting_voxel_project)(eargs, X, B, offset, PM, SOURCEPOSITION, NORMALTODETECTOR,
                                          vdims, voxelSizes, pdims, FLOATONE);
            if(exactProjectionScaling)
            {
                (*scalingProjectionsExact)(eargs2, B, offset, pdims_uint, normalProjection,
                                           pixelSizes, sourceToDetector);
            } else
            {
                (*scalingProjectionsCos)(eargs2, B, offset, ICM, SOURCEPOSITION, NORMALTODETECTOR,
                                         pdims_uint, scalingFactor);
            }
        }
    }
    return 0;
}

int BaseReconstructor::copyFloatVector(cl::Buffer& from, cl::Buffer& to, unsigned int size)
{
    cl::EnqueueArgs eargs(*Q, cl::NDRange(size));
    (*FLOAT_CopyVector)(eargs, from, to).wait();
    return 0;
}

int BaseReconstructor::scaleFloatVector(cl::Buffer& v, float f, unsigned int size)
{
    cl::EnqueueArgs eargs(*Q, cl::NDRange(size));
    (*FLOAT_scaleVector)(eargs, v, f).wait();
    return 0;
}

int BaseReconstructor::addIntoFirstVectorSecondVectorScaled(cl::Buffer& a,
                                                            cl::Buffer& b,
                                                            float f,
                                                            unsigned int size)
{
    cl::EnqueueArgs eargs(*Q, cl::NDRange(size));
    (*FLOAT_addIntoFirstVectorSecondVectorScaled)(eargs, a, b, f).wait();
    return 0;
}

int BaseReconstructor::addIntoFirstVectorScaledSecondVector(cl::Buffer& a,
                                                            cl::Buffer& b,
                                                            float f,
                                                            unsigned int size)
{
    cl::EnqueueArgs eargs(*Q, cl::NDRange(size));
    (*FLOAT_addIntoFirstVectorScaledSecondVector)(eargs, a, b, f).wait();
    return 0;
}

std::vector<matrix::ProjectionMatrix>
BaseReconstructor::encodeProjectionMatrices(std::shared_ptr<io::DenProjectionMatrixReader> pm)
{
    std::vector<matrix::ProjectionMatrix> v;
    for(std::size_t i = 0; i != pdimz; i++)
    {
        matrix::ProjectionMatrix p = pm->readMatrix(i);
        v.push_back(p);
    }
    return v;
}

std::vector<cl_double16>
BaseReconstructor::inverseProjectionMatrices(std::vector<matrix::ProjectionMatrix> CM)
{
    std::vector<cl_double16> inverseProjectionMatrices;
    for(std::size_t i = 0; i != pdimz; i++)
    {
        matrix::ProjectionMatrix matrix = CM[i];

        double* P = matrix.getPtr();
        std::array<double, 3> sourcePosition = matrix.sourcePosition();
        CTL::matrix::SquareMatrix CME(4,
                                      { P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9],
                                        P[10], P[11], sourcePosition[0], sourcePosition[1],
                                        sourcePosition[2], 1.0 });
        matrix::LUDoolittleForm lu = matrix::LUDoolittleForm::LUDecomposeDoolittle(CME, 0.001);
        matrix::SquareMatrix invertedCameraMatrix = lu.inverseMatrix();
        double* icm = invertedCameraMatrix.getPtr();
        //    cl::Buffer buffer_P(*context, CL_MEM_COPY_HOST_PTR, sizeof(double) * 12, (void*)P);
        cl_double16 ICM({ icm[0], icm[1], icm[2], icm[3], icm[4], icm[5], icm[6], icm[7], icm[8],
                          icm[9], icm[10], icm[11], icm[12], icm[13], icm[14], icm[15] });
        inverseProjectionMatrices.push_back(ICM);
    }
    return inverseProjectionMatrices;
}

std::vector<float>
BaseReconstructor::computeScalingFactors(std::vector<matrix::ProjectionMatrix> PM)
{
    std::vector<float> scalingFactors;
    double xoveryspacing = pixelSpacingX / pixelSpacingY;
    double yoverxspacing = pixelSpacingY / pixelSpacingX;
    for(std::size_t i = 0; i != pdimz; i++)
    {
        double x1, x2, y1, y2;
        matrix::ProjectionMatrix pm = PM[i];
        std::array<double, 3> sourcePosition = pm.sourcePosition();
        std::array<double, 3> normalToDetector = pm.normalToDetector();
        std::array<double, 3> tangentToDetector = pm.tangentToDetectorYDirection();
        pm.project(sourcePosition[0] - normalToDetector[0], sourcePosition[1] - normalToDetector[1],
                   sourcePosition[2] - normalToDetector[2], &x1, &y1);
        pm.project(sourcePosition[0] - normalToDetector[0] + tangentToDetector[0],
                   sourcePosition[1] - normalToDetector[1] + tangentToDetector[1],
                   sourcePosition[2] - normalToDetector[2] + tangentToDetector[2], &x2, &y2);
        double scalingFactor
            = (x1 - x2) * (x1 - x2) * xoveryspacing + (y1 - y2) * (y1 - y2) * yoverxspacing;
        scalingFactors.push_back(scalingFactor);
    }
    return scalingFactors;
}

void BaseReconstructor::writeVolume(cl::Buffer& X, std::string path)
{
    uint16_t buf[3];
    buf[0] = vdimy;
    buf[1] = vdimx;
    buf[2] = vdimz;
    io::createEmptyFile(path, 0, true); // Try if this is faster
    io::appendBytes(path, (uint8_t*)buf, 6);
    Q->enqueueReadBuffer(X, CL_TRUE, 0, sizeof(float) * XDIM, x);
    io::appendBytes(path, (uint8_t*)x, XDIM * sizeof(float));
}

void BaseReconstructor::writeProjections(cl::Buffer& B, std::string path)
{
    uint16_t buf[3];
    buf[0] = pdimy;
    buf[1] = pdimx;
    buf[2] = pdimz;
    io::createEmptyFile(path, 0, true); // Try if this is faster
    io::appendBytes(path, (uint8_t*)buf, 6);
    Q->enqueueReadBuffer(B, CL_TRUE, 0, sizeof(float) * BDIM, b);
    io::appendBytes(path, (uint8_t*)b, BDIM * sizeof(float));
}

void BaseReconstructor::setTimepoint() { timepoint = std::chrono::steady_clock::now(); }

void BaseReconstructor::reportTime(std::string msg)
{
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - timepoint);
    LOGW << io::xprintf("%s: %ds", msg.c_str(), duration.count());
    setTimepoint();
}

double
BaseReconstructor::adjointProductTest(std::shared_ptr<io::DenProjectionMatrixReader> matrices)
{
    std::shared_ptr<cl::Buffer> xa_buf; // X buffers
    allocateXBuffers(1);
    xa_buf = getXBuffer(0);
    allocateBBuffers(1);
    std::shared_ptr<cl::Buffer> ba_buf; // B buffers
    ba_buf = getBBuffer(0);
    std::vector<matrix::ProjectionMatrix> PM = encodeProjectionMatrices(matrices);
    std::vector<cl_double16> ICM = inverseProjectionMatrices(PM);
    std::vector<float> scalingFactors = computeScalingFactors(PM);
    project(*x_buf, *ba_buf, PM, ICM, scalingFactors);
    backproject(*b_buf, *xa_buf, PM, ICM, scalingFactors);
    double bdotAx = scalarProductBBuffer_barier_double(*b_buf, *ba_buf);
    double ATbdotx = scalarProductXBuffer_barier_double(*x_buf, *xa_buf);
    return (bdotAx / ATbdotx);
}

} // namespace CTL
