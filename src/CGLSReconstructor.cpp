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
    FLOATcutting_voxel_project
        = std::make_shared<cl::make_kernel<cl::Buffer&, cl::Buffer&, cl_double16&, cl_double4&,
                                           cl_double4&, cl_int4&, cl_double4&, cl_int2&, float&>>(
            cl::Kernel(program, "FLOATcutting_voxel_project"));
    Q = std::make_shared<cl::CommandQueue>(*context, *device);
    return 0;
}

int CGLSReconstructor::initializeVectors(float* projections, float* volume)
{
    this->b = projections;
    this->x = volume;
    cl_int err;
    x_buf
        = std::make_shared<cl::Buffer>(*context, CL_MEM_COPY_HOST_PTR,
                                       sizeof(float) * vdimx * vdimy * vdimz, (void*)volume, &err);
    y_buf
        = std::make_shared<cl::Buffer>(*context, CL_MEM_COPY_HOST_PTR,
                                       sizeof(float) * vdimx * vdimy * vdimz, (void*)volume, &err);
    z_buf
        = std::make_shared<cl::Buffer>(*context, CL_MEM_COPY_HOST_PTR,
                                       sizeof(float) * vdimx * vdimy * vdimz, (void*)volume, &err);
    b_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_READ_ONLY,
                                         sizeof(float) * pdimx * pdimy * pdimz, (void*)projections,
                                         &err);
    c_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_COPY_HOST_PTR,
                                         sizeof(float) * pdimx * pdimy * pdimz, (void*)projections,
                                         &err);
    d_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_COPY_HOST_PTR,
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
    cl::EnqueueArgs eargs(*Q, cl::NDRange(vdimz, vdimy, vdimx));

    Q->enqueueReadBuffer(*x_buf, CL_TRUE, 0, sizeof(float) * vdimx * vdimy * vdimz, x);
    return 0;
}

} // namespace CTL
