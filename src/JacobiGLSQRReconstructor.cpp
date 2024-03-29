#include "JacobiGLSQRReconstructor.hpp"

namespace KCT {

int GLSQRReconstructor::initializeOpenCL(uint32_t platformId)
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
    io::concatenateTextFiles(
        clFile, true,
        { io::xprintf("%s/opencl/utils.cl", this->xpath.c_str()),
          io::xprintf("%s/opencl/jacobiPreconditionedProjector.cl", this->xpath.c_str()),
          io::xprintf("%s/opencl/precomputeJacobiPreconditioner.cl", this->xpath.c_str()),
          io::xprintf("%s/opencl/jacobiPreconditionedBackprojector.cl", this->xpath.c_str()) });
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
    FLOAT_multiply_vectors_into_first_vector
        = std::make_shared<cl::make_kernel<cl::Buffer&, cl::Buffer&>>(
            cl::Kernel(program, "FLOAT_multiply_vectors_into_first_vector"));
    FLOATjacobiPreconditionedCutting_voxel_project = std::make_shared<
        cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, unsigned int&, cl_double16&,
                        cl_double3&, cl_double3&, cl_int3&, cl_double3&, cl_int2&, float&>>(
        cl::Kernel(program, "FLOATjacobiPreconditionedCutting_voxel_project"));
    FLOATjacobiPreconditionedCutting_voxel_backproject = std::make_shared<
        cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, unsigned int&, cl_double16&,
                        cl_double3&, cl_double3&, cl_int3&, cl_double3&, cl_int2&, float&>>(
        cl::Kernel(program, "FLOATjacobiPreconditionedCutting_voxel_backproject"));
    ScalarProductPartial_barier = std::make_shared<
        cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::LocalSpaceArg&, unsigned int&>>(
        cl::Kernel(program, "vector_ScalarProductPartial_barier"));
    FLOATcutting_voxel_jacobiPreconditionerVector
        = std::make_shared<cl::make_kernel<cl::Buffer&, cl_double16&, cl_double3&, cl_double3&,
                                           cl_int3&, cl_double3&, cl_int2&, float&>>(
            cl::Kernel(program, "FLOATcutting_voxel_jacobiPreconditionerVector"));
    FLOAT_compute_sqrt
        = std::make_shared<cl::make_kernel<cl::Buffer&>>(cl::Kernel(program, "FLOAT_compute_sqrt"));
    FLOAT_compute_inverse = std::make_shared<cl::make_kernel<cl::Buffer&>>(
        cl::Kernel(program, "FLOAT_compute_inverse"));
    Q = std::make_shared<cl::CommandQueue>(*context, *device);
    return 0;
}

int GLSQRReconstructor::initializeVectors(float* projections, float* volume)
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
    xa_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * XDIM,
                                          nullptr, &err);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
        return -1;
    }
    xb_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * XDIM,
                                          nullptr, &err);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
        return -1;
    }
    xc_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * XDIM,
                                          nullptr, &err);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
        return -1;
    }
    xd_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * XDIM,
                                          nullptr, &err);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
        return -1;
    }
    xe_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * XDIM,
                                          nullptr, &err);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
        return -1;
    }
    xf_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * XDIM,
                                          nullptr, &err);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
        return -1;
    }
    xg_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * XDIM,
                                          nullptr, &err);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
        return -1;
    }
    xh_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * XDIM,
                                          nullptr, &err);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
        return -1;
    }
    xi_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * XDIM,
                                          nullptr, &err);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
        return -1;
    }
    xj_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * XDIM,
                                          nullptr, &err);
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

    ba_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * BDIM,
                                          nullptr, &err);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
        return -1;
    }
    bb_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * BDIM,
                                          nullptr, &err);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
        return -1;
    }
    bc_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * BDIM,
                                          nullptr, &err);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
        return -1;
    }
    bd_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * BDIM,
                                          nullptr, &err);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
        return -1;
    }
    be_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * BDIM,
                                          nullptr, &err);
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

/**
 * Funcion computes the norm of the Buffer that has vdimx * vdimy * vdimz elements.
 *
 * @param X
 *
 * @return
 */
double GLSQRReconstructor::normXBuffer_frame_double(cl::Buffer& X)
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
double GLSQRReconstructor::normXBuffer_barier_double(cl::Buffer& X)
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
double GLSQRReconstructor::scalarProductXBuffer_barier_double(cl::Buffer& A, cl::Buffer& B)
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
double GLSQRReconstructor::scalarProductBBuffer_barier_double(cl::Buffer& A, cl::Buffer& B)
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
double GLSQRReconstructor::normBBuffer_frame_double(cl::Buffer& B)
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
double GLSQRReconstructor::normBBuffer_barier_double(cl::Buffer& B)
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
float GLSQRReconstructor::normXBuffer_frame(cl::Buffer& X)
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
float GLSQRReconstructor::normXBuffer_barier(cl::Buffer& X)
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
float GLSQRReconstructor::normBBuffer_frame(cl::Buffer& B)
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
float GLSQRReconstructor::normBBuffer_barier(cl::Buffer& B)
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

int GLSQRReconstructor::precomputeJacobiPreconditioner(cl::Buffer& X,
                                                       std::vector<matrix::ProjectionMatrix>& V,
                                                       std::vector<float>& scalingFactors)
{
    Q->enqueueFillBuffer<cl_float>(X, FLOATZERO, 0, XDIM * sizeof(float));
    for(std::size_t i = 0; i != pdimz; i++)
    {
        matrix::ProjectionMatrix mat = V[i];
        float scalingFactor = scalingFactors[i];
        std::array<double, 3> sourcePosition = mat.sourcePosition();
        std::array<double, 3> normalToDetector = mat.normalToDetector();
        double* P = mat.getPtr();
        cl_double16 PM({ P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10], P[11],
                         0.0, 0.0, 0.0, 0.0 });
        cl_double3 SOURCEPOSITION({ sourcePosition[0], sourcePosition[1], sourcePosition[2] });
        cl_double3 NORMALTODETECTOR(
            { normalToDetector[0], normalToDetector[1], normalToDetector[2] });
        cl_int3 vdims({ int(vdimx), int(vdimy), int(vdimz) });
        cl_double3 voxelSizes({ voxelSpacingX, voxelSpacingY, voxelSpacingZ });
        cl_int2 pdims({ int(pdimx), int(pdimy) });
        cl::EnqueueArgs eargs(*Q, cl::NDRange(vdimz, vdimy, vdimx));
        (*FLOATcutting_voxel_jacobiPreconditionerVector)(
            eargs, X, PM, SOURCEPOSITION, NORMALTODETECTOR, vdims, voxelSizes, pdims, scalingFactor)
            .wait();
    }
    cl::EnqueueArgs eargs(*Q, cl::NDRange(XDIM));
    (*FLOAT_compute_sqrt)(eargs, X).wait();
    cl::EnqueueArgs eargs2(*Q, cl::NDRange(XDIM));
    (*FLOAT_compute_inverse)(eargs2, X).wait();
    return 0;
}

/**
 * @brief
 *
 * @param B
 * @param X
 * @param P Contains preconditioner J^-(1/2)
 * @param V
 * @param scalingFactors
 *
 * @return
 */
int GLSQRReconstructor::backproject(cl::Buffer& B,
                                    cl::Buffer& X,
                                    cl::Buffer& JAP,
                                    std::vector<matrix::ProjectionMatrix>& V,
                                    std::vector<float>& scalingFactors)
{
    Q->enqueueFillBuffer<cl_float>(X, FLOATZERO, 0, XDIM * sizeof(float));
    unsigned int frameSize = pdimx * pdimy;
    for(std::size_t i = 0; i != pdimz; i++)
    {
        matrix::ProjectionMatrix mat = V[i];
        float scalingFactor = scalingFactors[i];
        std::array<double, 3> sourcePosition = mat.sourcePosition();
        std::array<double, 3> normalToDetector = mat.normalToDetector();
        double* P = mat.getPtr();
        cl_double16 PM({ P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10], P[11],
                         0.0, 0.0, 0.0, 0.0 });
        cl_double3 SOURCEPOSITION({ sourcePosition[0], sourcePosition[1], sourcePosition[2] });
        cl_double3 NORMALTODETECTOR(
            { normalToDetector[0], normalToDetector[1], normalToDetector[2] });
        cl_int3 vdims({ int(vdimx), int(vdimy), int(vdimz) });
        cl_double3 voxelSizes({ voxelSpacingX, voxelSpacingY, voxelSpacingZ });
        cl_int2 pdims({ int(pdimx), int(pdimy) });
        cl::EnqueueArgs eargs(*Q, cl::NDRange(vdimz, vdimy, vdimx));
        unsigned int offset = i * frameSize;
        (*FLOATjacobiPreconditionedCutting_voxel_backproject)(
            eargs, X, JAP, B, offset, PM, SOURCEPOSITION, NORMALTODETECTOR, vdims, voxelSizes,
            pdims, scalingFactor)
            .wait();
    }
    return 0;
}

int GLSQRReconstructor::project(cl::Buffer& X,
                                cl::Buffer& JAP,
                                cl::Buffer& B,
                                std::vector<matrix::ProjectionMatrix>& V,
                                std::vector<float>& scalingFactors)
{
    Q->enqueueFillBuffer<cl_float>(B, FLOATZERO, 0, BDIM * sizeof(float));
    unsigned int frameSize = pdimx * pdimy;
    for(std::size_t i = 0; i != pdimz; i++)
    {
        matrix::ProjectionMatrix mat = V[i];
        float scalingFactor = scalingFactors[i];
        std::array<double, 3> sourcePosition = mat.sourcePosition();
        std::array<double, 3> normalToDetector = mat.normalToDetector();
        double* P = mat.getPtr();
        cl_double16 PM({ P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10], P[11],
                         0.0, 0.0, 0.0, 0.0 });
        cl_double3 SOURCEPOSITION({ sourcePosition[0], sourcePosition[1], sourcePosition[2] });
        cl_double3 NORMALTODETECTOR(
            { normalToDetector[0], normalToDetector[1], normalToDetector[2] });
        cl_int3 vdims({ int(vdimx), int(vdimy), int(vdimz) });
        cl_double3 voxelSizes({ voxelSpacingX, voxelSpacingY, voxelSpacingZ });
        cl_int2 pdims({ int(pdimx), int(pdimy) });
        unsigned int offset = i * frameSize;
        cl::EnqueueArgs eargs(*Q, cl::NDRange(vdimz, vdimy, vdimx));
        (*FLOATjacobiPreconditionedCutting_voxel_project)(eargs, X, JAP, B, offset, PM,
                                                          SOURCEPOSITION, NORMALTODETECTOR, vdims,
                                                          voxelSizes, pdims, scalingFactor)
            .wait();
    }
    return 0;
}

int GLSQRReconstructor::copyFloatVector(cl::Buffer& from, cl::Buffer& to, unsigned int size)
{
    cl::EnqueueArgs eargs(*Q, cl::NDRange(size));
    (*FLOAT_CopyVector)(eargs, from, to).wait();
    return 0;
}

int GLSQRReconstructor::scaleFloatVector(cl::Buffer& v, float f, unsigned int size)
{
    cl::EnqueueArgs eargs(*Q, cl::NDRange(size));
    (*FLOAT_scaleVector)(eargs, v, f).wait();
    return 0;
}

int GLSQRReconstructor::addIntoFirstVectorSecondVectorScaled(cl::Buffer& a,
                                                             cl::Buffer& b,
                                                             float f,
                                                             unsigned int size)
{
    cl::EnqueueArgs eargs(*Q, cl::NDRange(size));
    (*FLOAT_addIntoFirstVectorSecondVectorScaled)(eargs, a, b, f).wait();
    return 0;
}

int GLSQRReconstructor::addIntoFirstVectorScaledSecondVector(cl::Buffer& a,
                                                             cl::Buffer& b,
                                                             float f,
                                                             unsigned int size)
{
    cl::EnqueueArgs eargs(*Q, cl::NDRange(size));
    (*FLOAT_addIntoFirstVectorScaledSecondVector)(eargs, a, b, f).wait();
    return 0;
}

std::vector<matrix::ProjectionMatrix>
GLSQRReconstructor::encodeProjectionMatrices(std::shared_ptr<io::DenProjectionMatrixReader> pm)
{
    std::vector<matrix::ProjectionMatrix> v;
    for(std::size_t i = 0; i != pdimz; i++)
    {
        matrix::ProjectionMatrix p = pm->readMatrix(i);
        v.push_back(p);
    }
    return v;
}

std::vector<float>
GLSQRReconstructor::computeScalingFactors(std::vector<matrix::ProjectionMatrix> PM)
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
        /*
    double xspacing2 = pixelSpacingX * pixelSpacingX;
    double yspacing2 = pixelSpacingY * pixelSpacingY;
        double distToDetector
            = std::sqrt((x1 - x2) * (x1 - x2) * xspacing2
                        + (y1 - y2) * (y1 - y2) * yspacing2); // Here I am using the fact that I
                                                              // have projected two vectors with the
                                                              // angle 45deg and so the distance on
                                                              // the detector is the same as the
                                                              // distance from source to detector
        double scalingFactor = distToDetector * distToDetector / pixelSpacingX / pixelSpacingY;*/
        double scalingFactor
            = (x1 - x2) * (x1 - x2) * xoveryspacing + (y1 - y2) * (y1 - y2) * yoverxspacing;
        scalingFactors.push_back(scalingFactor);
    }
    return scalingFactors;
}

void GLSQRReconstructor::writeVolume(cl::Buffer& X, std::string path)
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

void GLSQRReconstructor::writeProjections(cl::Buffer& B, std::string path)
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

void GLSQRReconstructor::setTimepoint() { timepoint = std::chrono::steady_clock::now(); }

void GLSQRReconstructor::reportTime(std::string msg)
{
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - timepoint);
    LOGW << io::xprintf("%s: %ds", msg.c_str(), duration.count());
    setTimepoint();
}

int GLSQRReconstructor::reconstruct(std::shared_ptr<io::DenProjectionMatrixReader> matrices,
                                    uint32_t maxIterations,
                                    float errCondition)
{
    LOGI << io::xprintf("WELCOME TO GLSQR");
    reportTime("GLSQR INIT");
    if(reportProgress)
    {
        // writeProjections(*b_buf, io::xprintf("%sb.den", progressBeginPath.c_str()));
        // writeVolume(*x_buf, io::xprintf("%sx_0.den", progressBeginPath.c_str()));
    }
    std::vector<matrix::ProjectionMatrix> PM = encodeProjectionMatrices(matrices);
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
    std::shared_ptr<cl::Buffer> x_jacobi_preconditioner;
    x_jacobi_preconditioner = xj_buf;
    reportTime("Precomputing preconditioner");
    precomputeJacobiPreconditioner(*x_jacobi_preconditioner, PM, scalingFactors);
    writeVolume(*x_jacobi_preconditioner,
                io::xprintf("%sPreconditioner.den", progressBeginPath.c_str()));
    // Try to use identity
    // cl_float fillones = 2.0;
    // Q->enqueueFillBuffer<cl_float>(*x_jacobi_preconditioner, fillones, 0, XDIM * sizeof(float));
    // END
    // Anything might be supplied here, but we will do standard initialization first
    v_next = xa_buf;
    Q->enqueueFillBuffer<cl_float>(*v_next, FLOATZERO, 0, XDIM * sizeof(float));
    backproject(*b_buf, *v_next, *x_jacobi_preconditioner, PM, scalingFactors);
    double vnextnorm = std::sqrt(normXBuffer_barier_double(*v_next));
    LOGI << io::xprintf("vnextnorm=%f", vnextnorm);
    scaleFloatVector(*v_next, float(1.0 / vnextnorm), XDIM);
    bool initializedByScaledBackprojectedRightSide = true;

    double d = 0.0;

    u_cur = ba_buf;
    Q->enqueueFillBuffer<cl_float>(*u_cur, FLOATZERO, 0, BDIM * sizeof(float));

    v_cur = xb_buf;
    Q->enqueueFillBuffer<cl_float>(*v_cur, FLOATZERO, 0, XDIM * sizeof(float));

    double varphi_hat = NB0;

    u_next = bb_buf;
    Q->enqueueFillBuffer<cl_float>(*u_next, FLOATZERO, 0, BDIM * sizeof(float));
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
    v_prev = xf_buf;
    x_prev = xg_buf;
    w_prev_prev = xh_buf;
    XZ = xi_buf;
    BZ = bd_buf;
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

        backproject(*u_cur, *XZ, *x_jacobi_preconditioner, PM, scalingFactors);
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

        project(*v_cur, *x_jacobi_preconditioner, *BZ, PM, scalingFactors);
        tau_prev = scalarProductBBuffer_barier_double(*BZ, *u_prev);
        addIntoFirstVectorSecondVectorScaled(*BZ, *u_prev, float(-tau_prev), BDIM);

        tau_cur = scalarProductBBuffer_barier_double(*BZ, *u_cur);
        addIntoFirstVectorSecondVectorScaled(*BZ, *u_cur, float(-tau_cur), BDIM);
        tau_next = std::sqrt(normBBuffer_barier_double(*BZ));
        LOGE << io::xprintf("tau_prev=%f, tau_cur=%f, tau_next=%f", tau_prev, tau_cur, tau_next);

        if(tau_next != 0)
        {
            Q->enqueueFillBuffer<cl_float>(*u_next, FLOATZERO, 0, BDIM * sizeof(float));
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
        if(reportProgress)
        {
            LOGD << io::xprintf("Writing file %sx_%d.den", progressBeginPath.c_str(), iteration);
            writeVolume(*x_cur, io::xprintf("%sx_%d.den", progressBeginPath.c_str(), iteration));
        }
    }
    cl::EnqueueArgs eargs(*Q, cl::NDRange(XDIM));
    (*FLOAT_multiply_vectors_into_first_vector)(eargs, *x_cur, *x_jacobi_preconditioner).wait();
    Q->enqueueReadBuffer(*x_cur, CL_TRUE, 0, sizeof(float) * XDIM, x);
    return 0;
}

} // namespace KCT
