#include "GLSQRPerfusionReconstructor.hpp"

namespace CTL {

struct Watches
{
    bool pressed = false;
    std::chrono::time_point<std::chrono::high_resolution_clock> lastTime;
    std::chrono::time_point<std::chrono::high_resolution_clock> now;
    void press(std::string txt = "")
    {
        using namespace std::chrono;
        if(!pressed)
        {
            pressed = true;
            LOGI << io::xprintf("%s", txt.c_str());
            lastTime = high_resolution_clock::now();
        } else
        {
            now = high_resolution_clock::now();
            duration<double> xxx = duration_cast<duration<double>>(now - lastTime);
            LOGI << io::xprintf("%s %0.3fs", txt.c_str(), xxx.count());
            lastTime = now;
        }
    };

    void pressE(std::string txt = "")
    {
        using namespace std::chrono;
        if(!pressed)
        {
            pressed = true;
            LOGE << io::xprintf("%s", txt.c_str());
            lastTime = high_resolution_clock::now();
        } else
        {
            now = high_resolution_clock::now();
            duration<double> xxx = duration_cast<duration<double>>(now - lastTime);
            LOGE << io::xprintf("%s %0.3fs", txt.c_str(), xxx.count());
            lastTime = now;
        }
    };
};

int GLSQRPerfusionReconstructor::initializeOpenCL(uint32_t platformId)
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
    // clFile = io::xprintf("%s/opencl/centerVoxelProjector.cl", this->xpath.c_str());
    clFile = io::xprintf("%s/opencl/allsources.cl", this->xpath.c_str());
    if(sidon)
    {
        io::concatenateTextFiles(
            clFile, true,
            { io::xprintf("%s/opencl/utils.cl", this->xpath.c_str()),
              io::xprintf("%s/opencl/projector_sidon.cl", this->xpath.c_str()),
              io::xprintf("%s/opencl/backprojector_sidon.cl", this->xpath.c_str()) });
    } else
    {
        io::concatenateTextFiles(
            clFile, true,
            { io::xprintf("%s/opencl/utils.cl", this->xpath.c_str()),
              io::xprintf("%s/opencl/projector.cl", this->xpath.c_str()),
              io::xprintf("%s/opencl/backprojector.cl", this->xpath.c_str()) });
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
    FLOAT_addIntoFirstVectorSecondVectorScaledOffset
        = std::make_shared<cl::make_kernel<cl::Buffer&, cl::Buffer&, float&, unsigned int&>>(
            cl::Kernel(program, "FLOAT_add_into_first_vector_second_vector_scaled_offset"));
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
    FLOAT_ZeroVector
        = std::make_shared<cl::make_kernel<cl::Buffer&>>(cl::Kernel(program, "FLOAT_zero_vector"));
    FLOATcutting_voxel_project = std::make_shared<
        cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&, cl_double16&, cl_double3&,
                        cl_double3&, cl_int3&, cl_double3&, cl_int2&, float&>>(
        cl::Kernel(program, "FLOATcutting_voxel_project"));
    FLOATcutting_voxel_backproject = std::make_shared<
        cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&, cl_double16&, cl_double3&,
                        cl_double3&, cl_int3&, cl_double3&, cl_int2&, float&>>(
        cl::Kernel(program, "FLOATcutting_voxel_backproject"));
    ScalarProductPartial_barier = std::make_shared<
        cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::LocalSpaceArg&, unsigned int&>>(
        cl::Kernel(program, "vector_ScalarProductPartial_barier"));
    scalingProjections
        = std::make_shared<cl::make_kernel<cl::Buffer&, unsigned int&, cl_double16&, cl_double3&,
                                           cl_double3&, cl_int2&, float&>>(
            cl::Kernel(program, "FLOATrescale_projections"));
    Q = std::make_shared<cl::CommandQueue>(*context, *device);
    return 0;
}

int GLSQRPerfusionReconstructor::updateX(std::vector<float*> volumes)
{
    std::string ERR;
    if(this->x.size() != volumes.size())
    {
        ERR = "Dimension mismatch";
        LOGE << ERR;
        throw std::runtime_error(ERR);
    }
    this->x = volumes;
    for(std::size_t i = 0; i != x.size(); i++)
    {
        Q->enqueueWriteBuffer(*x_buf[i], CL_BLOCKING, 0, sizeof(float) * XDIM, (void*)x[i], NULL,
                              NULL);
    }
    return 0;
}

int GLSQRPerfusionReconstructor::updateB(std::vector<float*> projections)
{
    std::string ERR;
    if(this->b.size() != projections.size())
    {
        ERR = "Dimension mismatch";
        LOGE << ERR;
        throw std::runtime_error(ERR);
    }
    this->b = projections;
    for(std::size_t i = 0; i != b.size(); i++)
    {
        Q->enqueueWriteBuffer(*b_buf[i], CL_BLOCKING, 0, sizeof(float) * BDIM, (void*)b[i], NULL,
                              NULL);
    }
    return 0;
}

int GLSQRPerfusionReconstructor::initializeData(std::vector<float*> projections,
                                                std::vector<float*> basisVectorValues,
                                                std::vector<float*> volumes)
{
    std::string ERR;
    if(basisVectorValues.size() != volumes.size())
    {
        ERR = "Dimension mismatch";
        LOGE << ERR;
        throw std::runtime_error(ERR);
    }
    this->b = projections;
    this->basisFunctionsValues = basisVectorValues;
    this->x = volumes;
    cl_int err;

    // Initialize buffers x_buf, v_buf and v_buf by zeros
    for(std::size_t i = 0; i != x.size(); i++)
    {
        x_buf.push_back(std::make_shared<cl::Buffer>(*context,
                                                     CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                     sizeof(float) * XDIM, (void*)x[i], &err));
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
            return -1;
        }
        xa_buf.push_back(std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                      sizeof(float) * XDIM, nullptr, &err));
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
            return -1;
        }
        xb_buf.push_back(std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                      sizeof(float) * XDIM, nullptr, &err));
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
            return -1;
        }
        xc_buf.push_back(std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                      sizeof(float) * XDIM, nullptr, &err));
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
            return -1;
        }
        xd_buf.push_back(std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                      sizeof(float) * XDIM, nullptr, &err));
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
            return -1;
        }
        xe_buf.push_back(std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                      sizeof(float) * XDIM, nullptr, &err));
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
            return -1;
        }
        xf_buf.push_back(std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                      sizeof(float) * XDIM, nullptr, &err));
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
            return -1;
        }
        xg_buf.push_back(std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                      sizeof(float) * XDIM, nullptr, &err));
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
            return -1;
        }
        xh_buf.push_back(std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                      sizeof(float) * XDIM, nullptr, &err));
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
            return -1;
        }
        xi_buf.push_back(std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                      sizeof(float) * XDIM, nullptr, &err));
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
            return -1;
        }
        xj_buf.push_back(std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                      sizeof(float) * XDIM, nullptr, &err));
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
            return -1;
        }
        xk_buf.push_back(std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                      sizeof(float) * XDIM, nullptr, &err));
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
            return -1;
        }
        xl_buf.push_back(std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                      sizeof(float) * XDIM, nullptr, &err));
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
            return -1;
        }
        xm_buf.push_back(std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                      sizeof(float) * XDIM, nullptr, &err));
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
            return -1;
        }
        xn_buf.push_back(std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                      sizeof(float) * XDIM, nullptr, &err));
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
            return -1;
        }
    }
    for(std::size_t i = 0; i != b.size(); i++)
    {
        // Special buffer to optimize backprojection time not to erase it every time
        xB_tmp_buf.push_back(std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                          sizeof(float) * XDIM, nullptr, &err));
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
            return -1;
        }

        b_buf.push_back(
            std::make_shared<cl::Buffer>(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         sizeof(float) * BDIM, (void*)projections[i], &err));
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
            return -1;
        }
        ba_buf.push_back(std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                      sizeof(float) * BDIM, nullptr, &err));
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
            return -1;
        }
        bb_buf.push_back(std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                      sizeof(float) * BDIM, nullptr, &err));
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
            return -1;
        }
        bc_buf.push_back(std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                      sizeof(float) * BDIM, nullptr, &err));
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
            return -1;
        }
        bd_buf.push_back(std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                      sizeof(float) * BDIM, nullptr, &err));
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
            return -1;
        }
        be_buf.push_back(std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                      sizeof(float) * BDIM, nullptr, &err));
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
            return -1;
        }
        tmp_b_buf.push_back(std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                         sizeof(float) * BDIM, nullptr, &err));
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
            return -1;
        }
    }
    tmp_x = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * XDIM, nullptr,
                                         &err);
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
    tmp_b = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * BDIM, nullptr,
                                         &err);
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

/**
 * Funcion computes the norm of the Buffer that has vdimx * vdimy * vdimz elements.
 *
 * @param X
 *
 * @return
 */
double GLSQRPerfusionReconstructor::normXBuffer_frame_double(cl::Buffer& X)
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
double GLSQRPerfusionReconstructor::normXBuffer_barier_double(cl::Buffer& X)
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

double
GLSQRPerfusionReconstructor::normXBuffer_barier_double(std::vector<std::shared_ptr<cl::Buffer>>& X)
{
    double sum = 0;
    for(std::shared_ptr<cl::Buffer> x : X)
    {
        sum += normXBuffer_barier_double(*x);
    }
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
double GLSQRPerfusionReconstructor::scalarProductXBuffer_barier_double(cl::Buffer& A, cl::Buffer& B)
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

double GLSQRPerfusionReconstructor::scalarProductXBuffer_barier_double(
    std::vector<std::shared_ptr<cl::Buffer>>& A, std::vector<std::shared_ptr<cl::Buffer>>& B)
{
    if(A.size() != B.size())
    {
        std::string err = "Vectors A and B has a different sizes!";
        LOGE << err;
        throw std::runtime_error(err);
    }
    double sum = 0;
    for(std::size_t i = 0; i != A.size(); i++)
    {
        sum += scalarProductXBuffer_barier_double(*A[i], *B[i]);
    }
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
double GLSQRPerfusionReconstructor::scalarProductBBuffer_barier_double(cl::Buffer& A, cl::Buffer& B)
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

double GLSQRPerfusionReconstructor::scalarProductBBuffer_barier_double(
    std::vector<std::shared_ptr<cl::Buffer>>& A, std::vector<std::shared_ptr<cl::Buffer>>& B)
{
    if(A.size() != B.size())
    {
        std::string err = "Vectors A and B has a different sizes!";
        LOGE << err;
        throw std::runtime_error(err);
    }
    double sum = 0;
    for(std::size_t i = 0; i != A.size(); i++)
    {
        sum += scalarProductBBuffer_barier_double(*A[i], *B[i]);
    }
    return sum;
}

/**
 * Computes square of the Euclidean norm of the buffer with pdimx * pdimy * pdimz float elements
 *
 * @param B input buffer
 *
 * @return square Euclidean norm
 */
double
GLSQRPerfusionReconstructor::normBBuffer_barier_double(std::vector<std::shared_ptr<cl::Buffer>>& B)
{ // Use workGroupSize that is private constant default to 256
    double sum = 0;
    for(std::size_t i = 0; i != B.size(); i++)
    {
        sum += normBBuffer_barier_double(*B[i]);
    }
    return sum;
}

/**
 * Funcion computes the norm of the Buffer that has pdimx * pdimy * pdimz elements.
 *
 * @param X
 *
 * @return
 */
double GLSQRPerfusionReconstructor::normBBuffer_frame_double(cl::Buffer& B)
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
double GLSQRPerfusionReconstructor::normBBuffer_barier_double(cl::Buffer& B)
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
float GLSQRPerfusionReconstructor::normXBuffer_frame(cl::Buffer& X)
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
float GLSQRPerfusionReconstructor::normXBuffer_barier(cl::Buffer& X)
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
float GLSQRPerfusionReconstructor::normBBuffer_frame(cl::Buffer& B)
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
float GLSQRPerfusionReconstructor::normBBuffer_barier(cl::Buffer& B)
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

int GLSQRPerfusionReconstructor::backproject(std::vector<std::shared_ptr<cl::Buffer>>& B,
                                             std::vector<std::shared_ptr<cl::Buffer>>& X,
                                             std::vector<matrix::ProjectionMatrix>& V,
                                             std::vector<cl_double16>& invertedProjectionMatrices,
                                             std::vector<float>& scalingFactors)
{
    Watches w;
    Q->finish();
    w.press("START Backrojection");
    zeroXBuffers(X);
    unsigned int frameSize = pdimx * pdimy;
    for(std::size_t sweepID = 0; sweepID != B.size(); sweepID++)
    {
        copyFloatVector(*B[sweepID], *tmp_b_buf[sweepID], BDIM);
    }
    unsigned int frameID;
    cl_int3 vdims({ int(vdimx), int(vdimy), int(vdimz) });
    cl_double3 voxelSizes({ voxelSpacingX, voxelSpacingY, voxelSpacingZ });
    cl_int2 pdims({ int(pdimx), int(pdimy) });
    // float FLOATONE = 1.0;
    cl::EnqueueArgs eargs(*Q, cl::NDRange(vdimz, vdimy, vdimx));
    cl::EnqueueArgs eargs2(*Q, cl::NDRange(pdimx, pdimy));
    float FLOATONE = 1.0;
    if(!sidon)
    {
        for(std::size_t angleID = 0; angleID != pdimz; angleID++)
        {
            unsigned int offset = angleID * frameSize;
            matrix::ProjectionMatrix mat = V[angleID];
            float scalingFactor = scalingFactors[angleID];
            std::array<double, 3> sourcePosition = mat.sourcePosition();
            std::array<double, 3> normalToDetector = mat.normalToDetector();
            double* P = mat.getPtr();
            cl_double16 PM({ P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10],
                             P[11], 0.0, 0.0, 0.0, 0.0 });
            cl_double16 ICM = invertedProjectionMatrices[angleID];
            cl_double3 SOURCEPOSITION({ sourcePosition[0], sourcePosition[1], sourcePosition[2] });
            cl_double3 NORMALTODETECTOR(
                { normalToDetector[0], normalToDetector[1], normalToDetector[2] });
            // Somehow better from the optimalization point of view
            // zeroFloatVector(*xB_tmp_buf[sweepID], XDIM);
            //        zeroFloatVector(*tmp_x, XDIM);
            //   Q->finish();
            //    w.press(io::xprintf("A angle %03d sweepID %02d", angleID, sweepID));
            // frameID = sweepID * pdimz + angleID;
            // Q->enqueueFillBuffer<cl_float>(*tmp_x, FLOATZERO, 0, XDIM * sizeof(float));
            // zeroFloatVector(*tmp_x, XDIM);
            for(std::size_t sweepID = 0; sweepID != B.size(); sweepID++)
            {
                (*scalingProjections)(eargs2, *tmp_b_buf[sweepID], offset, ICM, SOURCEPOSITION,
                                      NORMALTODETECTOR, pdims, scalingFactor);
            }
        }
    }
    Q->finish();
    w.press("Prepared for Backrojection");
    for(std::size_t basisIND = 0; basisIND != X.size(); basisIND++)
    {
        for(std::size_t sweepID = 0; sweepID != B.size(); sweepID++)
        {
            for(std::size_t angleID = 0; angleID != pdimz; angleID++)
            {
                unsigned int offset = angleID * frameSize;
                matrix::ProjectionMatrix mat = V[angleID];
                std::array<double, 3> sourcePosition = mat.sourcePosition();
                std::array<double, 3> normalToDetector = mat.normalToDetector();
                double* P = mat.getPtr();
                cl_double16 PM({ P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10],
                                 P[11], 0.0, 0.0, 0.0, 0.0 });
                cl_double3 SOURCEPOSITION(
                    { sourcePosition[0], sourcePosition[1], sourcePosition[2] });
                cl_double3 NORMALTODETECTOR(
                    { normalToDetector[0], normalToDetector[1], normalToDetector[2] });
                cl_double16 ICM = invertedProjectionMatrices[angleID];
                // Somehow better from the optimalization point of view
                // zeroFloatVector(*xB_tmp_buf[sweepID], XDIM);
                //        zeroFloatVector(*tmp_x, XDIM);
                //   Q->finish();
                //    w.press(io::xprintf("A angle %03d sweepID %02d", angleID, sweepID));
                // frameID = sweepID * pdimz + angleID;
                // Q->enqueueFillBuffer<cl_float>(*tmp_x, FLOATZERO, 0, XDIM * sizeof(float));
                // zeroFloatVector(*tmp_x, XDIM);
                //            .wait();

                //            (*FLOATcutting_voxel_backproject)(eargs, *tmp_x, *tmp_b_buf[sweepID],
                //            offset, PM,
                //                                              SOURCEPOSITION, NORMALTODETECTOR,
                //                                              vdims, voxelSizes, pdims, FLOATONE);

                // (*FLOATcutting_voxel_backproject)(eargs, *xB_tmp_buf[sweepID],
                // *tmp_b_buf[sweepID],
                //                                   offset, PM, SOURCEPOSITION, NORMALTODETECTOR,
                //                                   vdims, voxelSizes, pdims, FLOATONE);
                // Q->finish();
                //     w.press(io::xprintf("B angle %03d sweepID %02d", angleID, sweepID));
                // sleep(100);
                // w.press(io::xprintf("W angle %03d", angleID));

                // sleep(10);
                frameID = sweepID * pdimz + angleID;
                float scaleBy = basisFunctionsValues[basisIND][frameID];
                //       addIntoFirstVectorSecondVectorScaled(*X[basisIND], *tmp_x, scaleBy, XDIM);
                if(sidon)
                {
                    cl_uint2 pixelGranularity({ 1, 1 });
                    (*FLOATbackprojector_sidon)(eargs2, *tmp_x, *tmp_b_buf[sweepID], offset, ICM,
                                                SOURCEPOSITION, NORMALTODETECTOR, vdims, voxelSizes,
                                                pdims, FLOATONE, pixelGranularity);
                    addIntoFirstVectorSecondVectorScaled(*X[basisIND], *tmp_x, scaleBy, XDIM);
                } else
                {
                    (*FLOATcutting_voxel_backproject)(eargs, *X[basisIND], *tmp_b_buf[sweepID],
                                                      offset, PM, SOURCEPOSITION, NORMALTODETECTOR,
                                                      vdims, voxelSizes, pdims, scaleBy);
                }

                // addIntoFirstVectorSecondVectorScaled(*X[basisIND], *xB_tmp_buf[sweepID], scaleBy,
                //                                      XDIM);
            }
            // Q->finish();
            // w.press(io::xprintf("X angle %03d sweepID %02d", angleID, sweepID));
        }
    }
    Q->finish();
    w.press("END Backrojection");
    return 0;
}

int GLSQRPerfusionReconstructor::project(std::vector<std::shared_ptr<cl::Buffer>>& X,
                                         std::vector<std::shared_ptr<cl::Buffer>>& B,
                                         std::vector<matrix::ProjectionMatrix>& V,
                                         std::vector<cl_double16>& invertedProjectionMatrices,
                                         std::vector<float>& scalingFactors)
{
    Watches w;
    Q->finish();
    w.press("START Projection");
    zeroBBuffers(B);
    cl_int3 vdims({ int(vdimx), int(vdimy), int(vdimz) });
    cl_double3 voxelSizes({ voxelSpacingX, voxelSpacingY, voxelSpacingZ });
    cl_int2 pdims({ int(pdimx), int(pdimy) });
    unsigned int frameSize = pdimx * pdimy;
    float FLOATONE = 1.0;
    cl::EnqueueArgs eargs(*Q, cl::NDRange(vdimz, vdimy, vdimx));
    cl::EnqueueArgs eargs2(*Q, cl::NDRange(pdimx, pdimy));
    for(uint32_t basisIND = 0; basisIND != X.size(); ++basisIND)
    {
        Q->enqueueFillBuffer<cl_float>(*tmp_b, FLOATZERO, 0, BDIM * sizeof(float));
        for(std::size_t angleID = 0; angleID != pdimz; angleID++)
        {
            matrix::ProjectionMatrix mat = V[angleID];
            float scalingFactor = scalingFactors[angleID];
            std::array<double, 3> sourcePosition = mat.sourcePosition();
            std::array<double, 3> normalToDetector = mat.normalToDetector();
            double* P = mat.getPtr();
            cl_double16 PM({ P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10],
                             P[11], 0.0, 0.0, 0.0, 0.0 });
            cl_double16 ICM = invertedProjectionMatrices[angleID];
            cl_double3 SOURCEPOSITION({ sourcePosition[0], sourcePosition[1], sourcePosition[2] });
            cl_double3 NORMALTODETECTOR(
                { normalToDetector[0], normalToDetector[1], normalToDetector[2] });
            unsigned int offset = angleID * frameSize;
            if(sidon)
            {
                cl_uint2 pixelGranularity({ 1, 1 });
                (*FLOATprojector_sidon)(eargs2, *X[basisIND], *tmp_b, offset, ICM, SOURCEPOSITION,
                                        NORMALTODETECTOR, vdims, voxelSizes, pdims, FLOATONE,
                                        pixelGranularity); //        .wait();
            } else
            {
                (*FLOATcutting_voxel_project)(eargs, *X[basisIND], *tmp_b, offset, PM,
                                              SOURCEPOSITION, NORMALTODETECTOR, vdims, voxelSizes,
                                              pdims,
                                              FLOATONE); //        .wait();
                (*scalingProjections)(eargs2, *tmp_b, offset, ICM, SOURCEPOSITION, NORMALTODETECTOR,
                                      pdims, scalingFactor); //     .wait();
            }
        }
        for(uint32_t sweepID = 0; sweepID != B.size(); sweepID++)
        {
            for(std::size_t angleID = 0; angleID != pdimz; angleID++)
            {
                unsigned int offset = frameSize * angleID;
                float scaleBy = basisFunctionsValues[basisIND][sweepID * pdimz + angleID];
                addIntoFirstVectorSecondVectorScaledOffset(*B[sweepID], *tmp_b, scaleBy, frameSize,
                                                           offset);
            }
        }
    }
    Q->finish();
    w.press("END Projection");
    return 0;
}

int GLSQRPerfusionReconstructor::zeroFloatVector(cl::Buffer& b, unsigned int size)
{
    cl::EnqueueArgs eargs(*Q, cl::NDRange(size));
    (*FLOAT_ZeroVector)(eargs, b);
    return 0;
}

int GLSQRPerfusionReconstructor::copyFloatVector(cl::Buffer& from,
                                                 cl::Buffer& to,
                                                 unsigned int size)
{
    cl::EnqueueArgs eargs(*Q, cl::NDRange(size));
    (*FLOAT_CopyVector)(eargs, from, to).wait();
    return 0;
}

int GLSQRPerfusionReconstructor::copyFloatVector(std::vector<std::shared_ptr<cl::Buffer>>& From,
                                                 std::vector<std::shared_ptr<cl::Buffer>>& To,
                                                 unsigned int size)
{
    if(From.size() != To.size())
    {
        std::string err = "Vectors A and B has a different sizes!";
        LOGE << err;
        throw std::runtime_error(err);
    }
    for(std::size_t i = 0; i != From.size(); i++)
    {
        copyFloatVector(*From[i], *To[i], size);
    }
    return 0;
}

int GLSQRPerfusionReconstructor::scaleFloatVector(cl::Buffer& v, float f, unsigned int size)
{
    cl::EnqueueArgs eargs(*Q, cl::NDRange(size));
    (*FLOAT_scaleVector)(eargs, v, f).wait();
    return 0;
}

// Vector version
int GLSQRPerfusionReconstructor::scaleFloatVector(std::vector<std::shared_ptr<cl::Buffer>>& A,
                                                  float f,
                                                  unsigned int size)
{
    for(uint32_t i = 0; i != A.size(); i++)
    {
        scaleFloatVector(*A[i], f, size);
    }
    return 0;
}

int GLSQRPerfusionReconstructor::addIntoFirstVectorSecondVectorScaled(cl::Buffer& a,
                                                                      cl::Buffer& b,
                                                                      float f,
                                                                      unsigned int size)
{
    cl::EnqueueArgs eargs(*Q, cl::NDRange(size));
    (*FLOAT_addIntoFirstVectorSecondVectorScaled)(eargs, a, b, f);
    return 0;
}

int GLSQRPerfusionReconstructor::addIntoFirstVectorSecondVectorScaledOffset(
    cl::Buffer& a, cl::Buffer& b, float f, unsigned int size, unsigned int offset)
{
    cl::EnqueueArgs eargs(*Q, cl::NDRange(size));
    (*FLOAT_addIntoFirstVectorSecondVectorScaledOffset)(eargs, a, b, f, offset);
    return 0;
}

int GLSQRPerfusionReconstructor::addIntoFirstVectorSecondVectorScaled(
    std::vector<std::shared_ptr<cl::Buffer>>& A,
    std::vector<std::shared_ptr<cl::Buffer>>& B,
    float f,
    unsigned int size)
{
    if(A.size() != B.size())
    {
        std::string err = "Vectors A and B has a different sizes!";
        LOGE << err;
        throw std::runtime_error(err);
    }
    for(uint32_t i = 0; i != A.size(); i++)
    {
        addIntoFirstVectorSecondVectorScaled(*A[i], *B[i], f, size);
    }
    return 0;
}

int GLSQRPerfusionReconstructor::addIntoFirstVectorScaledSecondVector(cl::Buffer& a,
                                                                      cl::Buffer& b,
                                                                      float f,
                                                                      unsigned int size)
{
    cl::EnqueueArgs eargs(*Q, cl::NDRange(size));
    (*FLOAT_addIntoFirstVectorScaledSecondVector)(eargs, a, b, f).wait();
    return 0;
}

int GLSQRPerfusionReconstructor::addIntoFirstVectorScaledSecondVector(
    std::vector<std::shared_ptr<cl::Buffer>>& A,
    std::vector<std::shared_ptr<cl::Buffer>>& B,
    float f,
    unsigned int size)
{
    if(A.size() != B.size())
    {
        std::string err = "Vectors A and B has a different sizes!";
        LOGE << err;
        throw std::runtime_error(err);
    }
    for(uint32_t i = 0; i != A.size(); i++)
    {
        addIntoFirstVectorScaledSecondVector(*A[i], *B[i], f, size);
    }
    return 0;
}

std::vector<matrix::ProjectionMatrix> GLSQRPerfusionReconstructor::encodeProjectionMatrices(
    std::shared_ptr<io::DenProjectionMatrixReader> pm)
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
GLSQRPerfusionReconstructor::inverseProjectionMatrices(std::vector<matrix::ProjectionMatrix> CM)
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
GLSQRPerfusionReconstructor::computeScalingFactors(std::vector<matrix::ProjectionMatrix> PM)
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

void GLSQRPerfusionReconstructor::zeroXBuffers(std::vector<std::shared_ptr<cl::Buffer>>& X)
{
    for(std::size_t basisIND = 0; basisIND != X.size(); ++basisIND)
    {
        Q->enqueueFillBuffer<cl_float>(*X[basisIND], FLOATZERO, 0, XDIM * sizeof(float));
    }
}

void GLSQRPerfusionReconstructor::zeroBBuffers(std::vector<std::shared_ptr<cl::Buffer>>& B)
{
    for(std::size_t sweepID = 0; sweepID != B.size(); sweepID++)
    {
        Q->enqueueFillBuffer<cl_float>(*B[sweepID], FLOATZERO, 0, BDIM * sizeof(float));
    }
}

void GLSQRPerfusionReconstructor::writeVolume(std::vector<std::shared_ptr<cl::Buffer>>& X,
                                              std::string path)
{
    uint16_t buf[3];
    buf[0] = vdimy;
    buf[1] = vdimx;
    buf[2] = vdimz;
    for(std::size_t i = 0; i != X.size(); i++)
    {
        std::string newpath = io::xprintf("%s_elm%02d", path.c_str(), i);
        io::createEmptyFile(newpath, 0, true); // Try if this is faster
        io::appendBytes(newpath, (uint8_t*)buf, 6);
        Q->enqueueReadBuffer(*X[i], CL_TRUE, 0, sizeof(float) * XDIM, x[i]);
        io::appendBytes(newpath, (uint8_t*)x[i], XDIM * sizeof(float));
    }
}

void GLSQRPerfusionReconstructor::writeProjections(std::vector<std::shared_ptr<cl::Buffer>>& B,
                                                   std::string path)
{
    uint16_t buf[3];
    buf[0] = pdimy;
    buf[1] = pdimx;
    buf[2] = pdimz;
    for(std::size_t i = 0; i != B.size(); i++)
    {
        std::string newpath = io::xprintf("%s_proj%d", path.c_str(), i);
        io::createEmptyFile(newpath, 0, true); // Try if this is faster
        io::appendBytes(newpath, (uint8_t*)buf, 6);
        Q->enqueueReadBuffer(*B[i], CL_TRUE, 0, sizeof(float) * BDIM, b[i]);
        io::appendBytes(newpath, (uint8_t*)b[i], BDIM * sizeof(float));
    }
}

void GLSQRPerfusionReconstructor::setTimepoint() { timepoint = std::chrono::steady_clock::now(); }

void GLSQRPerfusionReconstructor::reportTime(std::string msg)
{
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - timepoint);
    LOGW << io::xprintf("%s: %ds", msg.c_str(), duration.count());
    setTimepoint();
}

int GLSQRPerfusionReconstructor::reconstruct(
    std::shared_ptr<io::DenProjectionMatrixReader> matrices,
    uint32_t maxIterations,
    float errCondition)
{
    LOGI << io::xprintf("GLSQR WITH PERFUSION");
    reportTime("GLSQR INIT");
    bool x0_iszero = true;
    std::vector<matrix::ProjectionMatrix> PM = encodeProjectionMatrices(matrices);
    std::vector<cl_double16> ICM = inverseProjectionMatrices(PM);
    std::vector<float> scalingFactors = computeScalingFactors(PM);
    uint32_t iteration = 0;

    // Initialization
    double NB0 = std::sqrt(normBBuffer_barier_double(b_buf));
    LOGI << io::xprintf("||b||=%f", NB0);
    std::vector<std::shared_ptr<cl::Buffer>>*u_prev, *u_cur, *u_next;
    std::vector<std::shared_ptr<cl::Buffer>>*v_prev, *v_cur, *v_next;
    std::vector<std::shared_ptr<cl::Buffer>>*w_prev_prev, *w_prev, *w_cur;
    std::vector<std::shared_ptr<cl::Buffer>>*x_prev, *x_cur;
    std::vector<std::shared_ptr<cl::Buffer>>*XZ, *BZ;

    LOGE << io::xprintf("Reconstruction01");
    // Inicializace vektoru v_1, ktery predstavuje normalizovany vektor, ktery je v klasickem
    // algoritmu normalizovany A^T b
    v_next = &xa_buf;
    bool initializedByScaledBackprojectedRightSide;
    if(x0_iszero)
    {
        initializedByScaledBackprojectedRightSide = true;
    } else
    {
        initializedByScaledBackprojectedRightSide = false;
    }
    double vnextnorm;
    if(initializedByScaledBackprojectedRightSide)
    {
        // v_next = 0 jako součást backproject procedury
        backproject(b_buf, *v_next, PM, ICM, scalingFactors);
        vnextnorm = std::sqrt(normXBuffer_barier_double(*v_next));
        LOGI << io::xprintf("vnextnorm=%f", vnextnorm);
        scaleFloatVector(*v_next, float(1.0 / vnextnorm), XDIM);
    } else
    {
        vnextnorm = std::sqrt(normXBuffer_barier_double(x_buf));
        zeroXBuffers(*v_next);
        addIntoFirstVectorSecondVectorScaled(*v_next, x_buf, float(1.0 / vnextnorm), XDIM);
    }
    LOGE << io::xprintf("Reconstruction02");
    double d = 0.0;

    u_cur = &ba_buf;
    zeroBBuffers(*u_cur);

    v_cur = &xb_buf;
    zeroXBuffers(*v_cur);

    double varphi_hat = NB0;

    u_next = &bb_buf;
    zeroBBuffers(*u_next);
    addIntoFirstVectorSecondVectorScaled(*u_next, b_buf, float(1.0 / varphi_hat), BDIM);
    LOGE << io::xprintf("Reconstruction03");

    x_cur = &x_buf;
    zeroXBuffers(*x_cur);

    w_cur = &xc_buf;
    zeroXBuffers(*w_cur);

    w_prev = &xd_buf;
    zeroXBuffers(*w_prev);

    double rho_cur = 1.0;
    double rho_prev = 1.0;
    double c_cur = -1.0;
    double c_prev = -1.0;
    double s_cur = 0.0;
    double s_prev = 0.0;

    // Now allocate memmory for the buffers that I will need
    std::vector<std::shared_ptr<cl::Buffer>>* tmp_buf;
    u_prev = &bc_buf;
    v_prev = &xf_buf;
    x_prev = &xg_buf;
    w_prev_prev = &xh_buf;
    XZ = &xi_buf;
    BZ = &bd_buf;
    double c_prev_prev;
    double s_prev_prev;
    double rho_prev_prev;
    double sigma_prev;
    double sigma_cur;
    double sigma_next;
    double sigma_tol = 0.001; // Based on the fact that numerical error is on the level ~0.0002
    double tau_cur, tau_prev, tau_next;
    double gamma;
    double varphi;
    double theta;
    double rho_hat;

    while(std::abs(varphi_hat) / NB0 > errCondition && iteration < maxIterations)
    {
        // Iteration
        iteration = iteration + 1;

        tmp_buf = u_prev;
        u_prev = u_cur;
        u_cur = u_next;
        u_next = tmp_buf;

        tmp_buf = v_prev;
        v_prev = v_cur;
        v_cur = v_next;
        v_next = tmp_buf;

        c_prev_prev = c_prev;
        c_prev = c_cur;

        s_prev_prev = s_prev;
        s_prev = s_cur;

        tmp_buf = x_prev;
        x_prev = x_cur;
        x_cur = tmp_buf;

        tmp_buf = w_prev_prev;
        w_prev_prev = w_prev;
        w_prev = w_cur;
        w_cur = tmp_buf;

        rho_prev_prev = rho_prev;
        rho_prev = rho_cur;

        LOGE << io::xprintf("Reconstruction05");
        backproject(*u_cur, *XZ, PM, ICM, scalingFactors);
        sigma_prev = scalarProductXBuffer_barier_double(*XZ, *v_prev);
        addIntoFirstVectorSecondVectorScaled(*XZ, *v_prev, float(-sigma_prev), XDIM);
        LOGI << io::xprintf("sigma_prev=%f", sigma_prev);
        LOGE << io::xprintf("Reconstruction06");

        if(d == 0.0)
        {
            LOGI << "d=0.0";
            sigma_cur = scalarProductXBuffer_barier_double(*XZ, *v_cur);
            addIntoFirstVectorSecondVectorScaled(*XZ, *v_cur, float(-sigma_cur), XDIM);

            sigma_next = std::sqrt(normXBuffer_barier_double(*XZ));
            LOGI << io::xprintf("sigma_next=%f", sigma_next);

            if(initializedByScaledBackprojectedRightSide)
            {
                LOGI << io::xprintf("Size of numerical error sigma_next=%f", sigma_next);
                sigma_next = 0;
            }

            if(sigma_next > sigma_tol)
            {
                zeroXBuffers(*v_next);
                addIntoFirstVectorSecondVectorScaled(*v_next, *XZ, float(1.0 / sigma_next), XDIM);
            } else
            {
                d = 1.0;
            }
        } else
        {
            LOGE << io::xprintf("Reconstruction07");
            LOGI << "d=1.0";
            sigma_cur = std::sqrt(normXBuffer_barier_double(*XZ));
            LOGI << io::xprintf("sigma_cur=%f", sigma_cur);

            if(sigma_cur > sigma_tol)
            {
                zeroXBuffers(*v_cur);
                addIntoFirstVectorSecondVectorScaled(*v_cur, *XZ, float(1.0 / sigma_cur), XDIM);
            } else
            {
                LOGI << "Ending due to the convergence";
                break;
            }
        }
        LOGE << io::xprintf("Reconstruction08");

        project(*v_cur, *BZ, PM, ICM, scalingFactors);
        tau_prev = scalarProductBBuffer_barier_double(*BZ, *u_prev);
        addIntoFirstVectorSecondVectorScaled(*BZ, *u_prev, float(-tau_prev), BDIM);

        tau_cur = scalarProductBBuffer_barier_double(*BZ, *u_cur);
        addIntoFirstVectorSecondVectorScaled(*BZ, *u_cur, float(-tau_cur), BDIM);
        tau_next = std::sqrt(normBBuffer_barier_double(*BZ));
        LOGE << io::xprintf("tau_prev=%f, tau_cur=%f, tau_next=%f", tau_prev, tau_cur, tau_next);

        if(tau_next != 0)
        {
            zeroBBuffers(*u_next);
            addIntoFirstVectorSecondVectorScaled(*u_next, *BZ, float(1 / tau_next), BDIM);
        }

        gamma = s_prev_prev * tau_prev;
        theta = -c_prev * c_prev_prev * tau_prev + s_prev * tau_cur;
        rho_hat = -s_prev * c_prev_prev * tau_prev - c_prev * tau_cur;
        LOGE << io::xprintf("gamma=%f, theta=%f, rho_hat=%f", gamma, theta, rho_hat);

        rho_cur = std::sqrt(rho_hat * rho_hat + tau_next * tau_next);
        c_cur = rho_hat / rho_cur;
        s_cur = tau_next / rho_cur;
        LOGE << io::xprintf("rho_cur=%f, s_cur=%f, c_cur=%f", rho_cur, s_cur, c_cur);
        // 24
        varphi = c_cur * varphi_hat;
        varphi_hat = s_cur * varphi_hat;
        // 25
        copyFloatVector(*v_cur, *w_cur, XDIM);
        addIntoFirstVectorSecondVectorScaled(*w_cur, *w_prev, float(-theta / rho_prev), XDIM);
        addIntoFirstVectorSecondVectorScaled(*w_cur, *w_prev_prev, float(-gamma / rho_prev_prev),
                                             XDIM);
        // 26
        copyFloatVector(*x_prev, *x_cur, XDIM);
        addIntoFirstVectorSecondVectorScaled(*x_cur, *w_cur, float(varphi / rho_cur), XDIM);
        if(tau_next == 0)
        {
            LOGI << "Ending due to the convergence";
            break;
        }

        LOGW << io::xprintf("After iteration %d, the norm of |Ax-b| is %f that is %0.2f%% of NB0.",
                            iteration, std::abs(varphi_hat), 100.0 * std::abs(varphi_hat) / NB0);
        if(reportEachK > 0 && iteration % reportEachK == 0)
        {
            LOGD << io::xprintf("Writing file %sx_iteration%02d.den", progressBeginPath.c_str(),
                                iteration);
            writeVolume(*x_cur,
                        io::xprintf("%sx_iteration%02d.den", progressBeginPath.c_str(), iteration));
        }
    }
    for(uint32_t basisIND = 0; basisIND != x_cur->size(); basisIND++)
    {
        Q->enqueueReadBuffer(*(*x_cur)[basisIND], CL_TRUE, 0, sizeof(float) * XDIM,
                             (void*)x[basisIND]);
    }
    return 0;
}

int GLSQRPerfusionReconstructor::projectXtoB(
    std::shared_ptr<io::DenProjectionMatrixReader> matrices)
{
    std::vector<matrix::ProjectionMatrix> PM = encodeProjectionMatrices(matrices);
    std::vector<cl_double16> ICM = inverseProjectionMatrices(PM);
    std::vector<float> scalingFactors = computeScalingFactors(PM);
    double NB0 = std::sqrt(normBBuffer_barier_double(b_buf));
    LOGI << io::xprintf("Current ||b||=%f", NB0);
    double NX0 = std::sqrt(normXBuffer_barier_double(x_buf));
    LOGI << io::xprintf("Current ||x||=%f", NX0);
    zeroBBuffers(ba_buf);
    project(x_buf, ba_buf, PM, ICM, scalingFactors);
    double NB1 = std::sqrt(normBBuffer_barier_double(ba_buf));
    LOGI << io::xprintf("Projected ||Ax||=%f", NB1);
    for(uint32_t sweepID = 0; sweepID != b_buf.size(); sweepID++)
    {
        Q->enqueueReadBuffer(*ba_buf[sweepID], CL_TRUE, 0, sizeof(float) * BDIM, (void*)b[sweepID]);
    }
    addIntoFirstVectorSecondVectorScaled(ba_buf, b_buf, -1.0f, BDIM);
    double NB2 = std::sqrt(normBBuffer_barier_double(ba_buf));
    LOGI << io::xprintf(" ||Ax-b||=%f", NB2);
    return 0;
}

int GLSQRPerfusionReconstructor::backprojectBtoX(
    std::shared_ptr<io::DenProjectionMatrixReader> matrices)
{
    std::vector<matrix::ProjectionMatrix> PM = encodeProjectionMatrices(matrices);
    std::vector<cl_double16> ICM = inverseProjectionMatrices(PM);
    std::vector<float> scalingFactors = computeScalingFactors(PM);
    double NB0 = std::sqrt(normBBuffer_barier_double(b_buf));
    LOGI << io::xprintf("Current ||b||=%f", NB0);
    double NX0 = std::sqrt(normXBuffer_barier_double(x_buf));
    LOGI << io::xprintf("Current ||x||=%f", NX0);
    zeroXBuffers(xa_buf);
    backproject(b_buf, xa_buf, PM, ICM, scalingFactors);
    double NX1 = std::sqrt(normXBuffer_barier_double(xa_buf));
    LOGI << io::xprintf("Backprojected ||A^Tb||=%f", NX1);
    for(uint32_t basisIND = 0; basisIND != x_buf.size(); basisIND++)
    {
        Q->enqueueReadBuffer(*xa_buf[basisIND], CL_TRUE, 0, sizeof(float) * XDIM,
                             (void*)x[basisIND]);
    }
    addIntoFirstVectorSecondVectorScaled(xa_buf, x_buf, -1.0f, XDIM);
    double NX2 = std::sqrt(normXBuffer_barier_double(xa_buf));
    LOGI << io::xprintf("Backprojected ||A^Tb-x||=%f", NX2);
    return 0;
}
/*
int GLSQRPerfusionReconstructor::reconstructTikhonov(
    std::shared_ptr<io::DenProjectionMatrixReader> matrices,
    double lambda,
    uint32_t maxIterations,
    float errCondition)
{
    LOGI << io::xprintf("TIKHONOV GLSQR");
    reportTime("INIT");
    // Ke vsem b bufferum je treba pridat jeden x buffer
    if(reportProgress)
    {
        // writeProjections(*b_buf, io::xprintf("%sb.den", progressBeginPath.c_str()));
        // writeVolume(*x_buf, io::xprintf("%sx_0.den", progressBeginPath.c_str()));
    }
    std::vector<matrix::ProjectionMatrix> PM = encodeProjectionMatrices(matrices);
    std::vector<cl_double16> ICM = inverseProjectionMatrices(PM);
    std::vector<float> scalingFactors = computeScalingFactors(PM);
    uint32_t iteration = 0;

    // Initialization
    double NB0 = std::sqrt(normBBuffer_barier_double(*b_buf));
    LOGI << io::xprintf("||b||=%f", NB0);
    std::shared_ptr<cl::Buffer> u_prev, u_cur, u_next;
    std::shared_ptr<cl::Buffer> v_prev, v_cur, v_next;
    std::shared_ptr<cl::Buffer> w_prev_prev, w_prev, w_cur;
    std::shared_ptr<cl::Buffer> x_prev, x_cur;
    std::shared_ptr<cl::Buffer> XZ, BZ;
    std::shared_ptr<cl::Buffer> u_prev_x, u_cur_x, u_next_x, BZ_x; // Aditional x buffers

    BZ_x = xl_buf;
    Q->enqueueFillBuffer<cl_float>(*BZ_x, FLOATZERO, 0, XDIM * sizeof(float));

    // Anything might be supplied here, but we will do standard initialization first
    v_next = xa_buf;
    Q->enqueueFillBuffer<cl_float>(*v_next, FLOATZERO, 0, XDIM * sizeof(float));
    backproject(*b_buf, *v_next, PM, ICM,
                scalingFactors); // Backprojection of zero is obviously zero for potential b_buf_x
    double vnextnorm = std::sqrt(normXBuffer_barier_double(*v_next));
    LOGI << io::xprintf("vnextnorm=%f", vnextnorm);
    scaleFloatVector(*v_next, float(1.0 / vnextnorm), XDIM);
    bool initializedByScaledBackprojectedRightSide = true;


//Initilization of GLSQR
    double d = 0.0;

    u_cur = ba_buf;
    Q->enqueueFillBuffer<cl_float>(*u_cur, FLOATZERO, 0, BDIM * sizeof(float));
    u_cur_x = xj_buf;
    Q->enqueueFillBuffer<cl_float>(*u_cur_x, FLOATZERO, 0, XDIM * sizeof(float));

    v_cur = xb_buf;
    Q->enqueueFillBuffer<cl_float>(*v_cur, FLOATZERO, 0, XDIM * sizeof(float));

    double varphi_hat = NB0;

    u_next = bb_buf;
    Q->enqueueFillBuffer<cl_float>(*u_next, FLOATZERO, 0, BDIM * sizeof(float));
    u_next_x = xk_buf;
    Q->enqueueFillBuffer<cl_float>(*u_next_x, FLOATZERO, 0, XDIM * sizeof(float));
    addIntoFirstVectorSecondVectorScaled(*u_next, *b_buf, float(1.0 / varphi_hat), BDIM);

    x_cur = x_buf;
    Q->enqueueFillBuffer<cl_float>(*x_cur, FLOATZERO, 0, XDIM * sizeof(float));

    w_cur = xc_buf;
    Q->enqueueFillBuffer<cl_float>(*w_cur, FLOATZERO, 0, XDIM * sizeof(float));

    w_prev = xd_buf;
    Q->enqueueFillBuffer<cl_float>(*w_prev, FLOATZERO, 0, XDIM * sizeof(float));

    double rho_cur = 1.0;
    double rho_prev = 1.0;
    double c_cur = -1.0;
    double c_prev = -1.0;
    double s_cur = 0.0;
    double s_prev = 0.0;

    // Now allocate memmory for the buffers that I will need
    std::shared_ptr<cl::Buffer> tmp_buf;
    u_prev = bc_buf;
    u_prev_x = xm_buf; // x part
    v_prev = xf_buf;
    x_prev = xg_buf;
    w_prev_prev = xh_buf;
    XZ = xi_buf;
    BZ = bd_buf;
    BZ_x = xl_buf;
    double c_prev_prev;
    double s_prev_prev;
    double rho_prev_prev;
    double sigma_prev;
    double sigma_cur;
    double sigma_next;
    double sigma_tol = 0.001; // Based on the fact that numerical error is on the level ~0.0002
    double tau_cur, tau_prev, tau_next;
    double gamma;
    double varphi;
    double theta;
    double rho_hat;

    while(std::abs(varphi_hat) / NB0 > errCondition && iteration < maxIterations)
    {
        // Iteration
        iteration = iteration + 1;

        tmp_buf = u_prev;
        u_prev = u_cur;
        u_cur = u_next;
        u_next = tmp_buf;
        tmp_buf = u_prev_x; // X part
        u_prev_x = u_cur_x;
        u_cur_x = u_next_x;
        u_next_x = tmp_buf;

        tmp_buf = v_prev;
        v_prev = v_cur;
        v_cur = v_next;
        v_next = tmp_buf;

        c_prev_prev = c_prev;
        c_prev = c_cur;

        s_prev_prev = s_prev;
        s_prev = s_cur;

        tmp_buf = x_prev;
        x_prev = x_cur;
        x_cur = tmp_buf;

        tmp_buf = w_prev_prev;
        w_prev_prev = w_prev;
        w_prev = w_cur;
        w_cur = tmp_buf;

        rho_prev_prev = rho_prev;
        rho_prev = rho_cur;

        backproject(*u_cur, *XZ, PM, ICM, scalingFactors);
        addIntoFirstVectorSecondVectorScaled(*XZ, *u_cur_x, lambda, XDIM);
        sigma_prev = scalarProductXBuffer_barier_double(*XZ, *v_prev);
        addIntoFirstVectorSecondVectorScaled(*XZ, *v_prev, float(-sigma_prev), XDIM);
        LOGI << io::xprintf("sigma_prev=%f", sigma_prev);

        if(d == 0.0)
        {
            LOGI << "d=0.0";
            sigma_cur = scalarProductXBuffer_barier_double(*XZ, *v_cur);
            addIntoFirstVectorSecondVectorScaled(*XZ, *v_cur, float(-sigma_cur), XDIM);

            sigma_next = std::sqrt(normXBuffer_barier_double(*XZ));
            LOGI << io::xprintf("sigma_next=%f", sigma_next);

            if(initializedByScaledBackprojectedRightSide)
            {
                LOGI << io::xprintf("Size of numerical error sigma_next=%f", sigma_next);
                sigma_next = 0;
            }

            if(sigma_next > sigma_tol)
            {
                Q->enqueueFillBuffer<cl_float>(*v_next, FLOATZERO, 0, XDIM * sizeof(float));
                addIntoFirstVectorSecondVectorScaled(*v_next, *XZ, float(1.0 / sigma_next), XDIM);
            } else
            {
                d = 1.0;
            }
        } else
        {
            LOGI << "d=1.0";
            sigma_cur = std::sqrt(normXBuffer_barier_double(*XZ));
            LOGI << io::xprintf("sigma_cur=%f", sigma_cur);

            if(sigma_cur > sigma_tol)
            {
                Q->enqueueFillBuffer<cl_float>(*v_cur, FLOATZERO, 0, XDIM * sizeof(float));
                addIntoFirstVectorSecondVectorScaled(*v_cur, *XZ, float(1.0 / sigma_cur), XDIM);
            } else
            {
                LOGI << "Ending due to the convergence";
                break;
            }
        }

        project(*v_cur, *BZ, PM, ICM, scalingFactors);
        copyFloatVector(*v_cur, *BZ_x, XDIM);
        scaleFloatVector(*BZ_x, lambda, XDIM);
        tau_prev = scalarProductBBuffer_barier_double(*BZ, *u_prev);
        tau_prev += scalarProductXBuffer_barier_double(*BZ_x, *u_prev_x);
        addIntoFirstVectorSecondVectorScaled(*BZ, *u_prev, float(-tau_prev), BDIM);
        addIntoFirstVectorSecondVectorScaled(*BZ_x, *u_prev_x, float(-tau_prev), XDIM);

        tau_cur = scalarProductBBuffer_barier_double(*BZ, *u_cur);
        tau_cur += scalarProductXBuffer_barier_double(*BZ_x, *u_cur_x);
        addIntoFirstVectorSecondVectorScaled(*BZ, *u_cur, float(-tau_cur), BDIM);
        addIntoFirstVectorSecondVectorScaled(*BZ_x, *u_cur_x, float(-tau_cur), XDIM);
        tau_next = std::sqrt(normBBuffer_barier_double(*BZ) + normXBuffer_barier_double(*BZ_x));
        LOGE << io::xprintf("tau_prev=%f, tau_cur=%f, tau_next=%f", tau_prev, tau_cur, tau_next);

        if(tau_next != 0)
        {
            Q->enqueueFillBuffer<cl_float>(*u_next, FLOATZERO, 0, BDIM * sizeof(float));
            Q->enqueueFillBuffer<cl_float>(*u_next_x, FLOATZERO, 0, XDIM * sizeof(float));
            addIntoFirstVectorSecondVectorScaled(*u_next, *BZ, float(1 / tau_next), BDIM);
            addIntoFirstVectorSecondVectorScaled(*u_next_x, *BZ_x, float(1 / tau_next), XDIM);
        }

        gamma = s_prev_prev * tau_prev;
        theta = -c_prev * c_prev_prev * tau_prev + s_prev * tau_cur;
        rho_hat = -s_prev * c_prev_prev * tau_prev - c_prev * tau_cur;
        LOGE << io::xprintf("gamma=%f, theta=%f, rho_hat=%f", gamma, theta, rho_hat);

        rho_cur = std::sqrt(rho_hat * rho_hat + tau_next * tau_next);
        c_cur = rho_hat / rho_cur;
        s_cur = tau_next / rho_cur;
        LOGE << io::xprintf("rho_cur=%f, s_cur=%f, c_cur=%f", rho_cur, s_cur, c_cur);
        // 24
        varphi = c_cur * varphi_hat;
        varphi_hat = s_cur * varphi_hat;
        // 25
        copyFloatVector(*v_cur, *w_cur, XDIM);
        addIntoFirstVectorSecondVectorScaled(*w_cur, *w_prev, float(-theta / rho_prev), XDIM);
        addIntoFirstVectorSecondVectorScaled(*w_cur, *w_prev_prev, float(-gamma / rho_prev_prev),
                                             XDIM);
        // 26
        copyFloatVector(*x_prev, *x_cur, XDIM);
        addIntoFirstVectorSecondVectorScaled(*x_cur, *w_cur, float(varphi / rho_cur), XDIM);
        if(tau_next == 0)
        {
            LOGI << "Ending due to the convergence";
            break;
        }

        LOGW << io::xprintf("After iteration %d, the norm of |Ax-b| is %f that is %0.2f%% of NB0.",
                            iteration, std::abs(varphi_hat), 100.0 * std::abs(varphi_hat) / NB0);
        if(reportProgress)
        {
            LOGD << io::xprintf("Writing file %sx_%d.den", progressBeginPath.c_str(), iteration);
            writeVolume(*x_cur, io::xprintf("%sx_%d.den", progressBeginPath.c_str(), iteration));
        }
    }
    Q->enqueueReadBuffer(*x_cur, CL_TRUE, 0, sizeof(float) * XDIM, x);
    return 0;
}
*/

double GLSQRPerfusionReconstructor::adjointProductTest(
    std::shared_ptr<io::DenProjectionMatrixReader> matrices)
{
    std::vector<matrix::ProjectionMatrix> PM = encodeProjectionMatrices(matrices);
    std::vector<cl_double16> ICM = inverseProjectionMatrices(PM);
    std::vector<float> scalingFactors = computeScalingFactors(PM);
    backproject(b_buf, xa_buf, PM, ICM, scalingFactors);
    project(x_buf, ba_buf, PM, ICM, scalingFactors);
    double bdotAx = scalarProductBBuffer_barier_double(b_buf, ba_buf);
    double ATbdotx = scalarProductXBuffer_barier_double(x_buf, xa_buf);
    return (bdotAx / ATbdotx);
}

} // namespace CTL
