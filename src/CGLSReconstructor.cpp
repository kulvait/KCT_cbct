#include "CGLSReconstructor.hpp"

namespace CTL {

int CGLSReconstructor::initializeOpenCL(uint32_t platformId)
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
    // clFile = io::xprintf("%s/opencl/centerVoxelProjector.cl", this->xpath.c_str());
    clFile = io::xprintf("%s/opencl/projector.cl", this->xpath.c_str());
    clFile = io::xprintf("%s/opencl/utils.cl", this->xpath.c_str());
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
    FLOAT_NormSquare = std::make_shared<cl::make_kernel<cl::Buffer&, cl::Buffer&, int&>>(
        cl::Kernel(program, "FLOATvector_NormSquarePartial"));
    FLOAT_SumPartial = std::make_shared<cl::make_kernel<cl::Buffer&, cl::Buffer&, int&>>(
        cl::Kernel(program, "FLOATvector_SumPartial"));
    FLOAT_NormSquare_barier
        = std::make_shared<cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::LocalSpaceArg&, int&>>(
            cl::Kernel(program, "FLOATvector_NormSquarePartial_barier"));
    Q = std::make_shared<cl::CommandQueue>(*context, *device);
    return 0;
}

int CGLSReconstructor::initializeVectors(float* projections, float* volume)
{
    this->b = projections;
    this->x = volume;
    cl_int err;

    // Initialize buffers x_buf, v_buf and v_buf by zeros
    x_buf
        = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                       sizeof(float) * vdimx * vdimy * vdimz, (void*)volume, &err);
    v_buf
        = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                       sizeof(float) * vdimx * vdimy * vdimz, (void*)volume, &err);
    w_buf
        = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                       sizeof(float) * vdimx * vdimy * vdimz, (void*)volume, &err);

    b_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         sizeof(float) * pdimx * pdimy * pdimz, (void*)projections,
                                         &err);
    // Initialize buffer c by projections data
    c_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                         sizeof(float) * pdimx * pdimy * pdimz, (void*)projections,
                                         &err);

    d_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                         sizeof(float) * pdimx * pdimy * pdimz, (void*)projections,
                                         &err);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful initialization of Image3D with error code %d!", err);
        return -1;
    }
    return 0;
}

int CGLSReconstructor::reconstruct(std::shared_ptr<io::DenProjectionMatrixReader> matrices)
{
    cl_float FLOATZERO(0.0);
    float bnorm = 0.0;
    cl_int err;
    cl::Buffer bnorm_buf_part(*context, CL_MEM_READ_WRITE, sizeof(float) * pdimz, nullptr, &err);
    cl::Buffer bnorm_buf(*context, CL_MEM_READ_WRITE, sizeof(float), nullptr, &err);

    Q->enqueueFillBuffer<cl_float>(*x_buf, FLOATZERO, 0, vdimx * vdimy * vdimz);
    Q->enqueueFillBuffer<cl_float>(*v_buf, FLOATZERO, 0, vdimx * vdimy * vdimz);
    Q->enqueueFillBuffer<cl_float>(*w_buf, FLOATZERO, 0, vdimx * vdimy * vdimz);
    Q->enqueueFillBuffer<cl_float>(*c_buf, FLOATZERO, 0, pdimx * pdimy * pdimz);
    Q->enqueueFillBuffer<cl_float>(*d_buf, FLOATZERO, 0, pdimx * pdimy * pdimz);
    cl::EnqueueArgs eargs(*Q, cl::NDRange(pdimz));
    int size = pdimx * pdimy;
    (*FLOAT_NormSquare)(eargs, *b_buf, bnorm_buf_part, size);
    cl::EnqueueArgs eargs2(*Q, cl::NDRange(1));
    size = 1;
    (*FLOAT_NormSquare)(eargs2, bnorm_buf_part, bnorm_buf, size);

    int ptotal, dimcompute;
    ptotal = pdimx * pdimy * pdimz;
    dimcompute = ptotal + (256 - ptotal % 256) % 256;
    cl::Buffer bnorm_reduced(*context, CL_MEM_READ_WRITE, sizeof(float) * dimcompute / 256, nullptr,
                             &err);
    cl::EnqueueArgs eargs_local(*Q, cl::NDRange(dimcompute), cl::NDRange(256));
    cl::LocalSpaceArg localsize = cl::Local(256);
    (*FLOAT_NormSquare_barier)(eargs_local, *b_buf, bnorm_reduced,localsize ,ptotal);
	int frameC = dimcompute / 256;   
 (*FLOAT_SumPartial)(eargs2, bnorm_reduced, bnorm_buf, frameC);

    Q->enqueueReadBuffer(bnorm_buf, CL_TRUE, 0, sizeof(float), &bnorm);
    LOGI << io::xprintf("Norm of b is %f", bnorm);
    return 0;
}

} // namespace CTL
