#include "CuttingVoxelProjector.hpp"

namespace CTL {

int CuttingVoxelProjector::initializeOpenCL(uint32_t platformId)
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
    cl::Program::Sources sources;
    std::string kernel_code
        = "   void kernel simple_add(global const int* A, global const int* B, global int* C){     "
          "  "
          "       C[get_global_id(0)]=A[get_global_id(0)]+B[get_global_id(0)];                 "
          "   }   ";
    sources.push_back({ kernel_code.c_str(), kernel_code.length() });

    cl::Program program(*context, sources);
    if(program.build({ *device }) != CL_SUCCESS)
    {
        LOGE << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(*device);
        return -3;
    }
    // OpenCL 1.2 got rid of KernelFunctor
    // https://forums.khronos.org/showthread.php/8317-cl-hpp-KernelFunctor-gone-replaced-with-KernelFunctorGlobal
    // https://stackoverflow.com/questions/23992369/what-should-i-use-instead-of-clkernelfunctor/54344990#54344990
    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer> simple_add(
        cl::Kernel(program, "simple_add"));
    // Create buffers on the device
    cl::Buffer buffer_A(*context, CL_MEM_READ_WRITE, sizeof(int) * 10);
    cl::Buffer buffer_B(*context, CL_MEM_READ_WRITE, sizeof(int) * 10);
    cl::Buffer buffer_C(*context, CL_MEM_READ_WRITE, sizeof(int) * 10);

    int A[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    int B[] = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 0 };

    // create queue to which we will push commands for the device.
    Q = std::make_shared<cl::CommandQueue>(*context, *device);
    // write arrays A and B to the device
    Q->enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int) * 10, A);
    Q->enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int) * 10, B);
    cl::EnqueueArgs eargs(*Q, cl::NullRange, cl::NDRange(10), cl::NullRange);
    simple_add(eargs, buffer_A, buffer_B, buffer_C).wait();

    int C[10];
    // read result C from the device to array C
    Q->enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int) * 10, C);

    std::cout << " result: \n";
    for(int i = 0; i < 10; i++)
    {
        std::cout << C[i] << " ";
    }
    return 0;
}

int CuttingVoxelProjector::initializeVolumeImage()
{
    cl::ImageFormat f(CL_INTENSITY, CL_FLOAT);
    cl_int err;
    volumeBuffer
        = std::make_shared<cl::Buffer>(*context, CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_READ_ONLY,
                                       sizeof(float) * vdimx * vdimy * vdimz, (void*)volume, &err);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful initialization of Image3D with error code %d!", err);
        return -1;
    }
    return 0;
}

int CuttingVoxelProjector::project(float* projection,
                                   uint32_t pdimx,
                                   uint32_t pdimy,
                                   matrix::ProjectionMatrix matrix,
                                   float scalingFactor)
{
    // FIXME
    // https://software.intel.com/en-us/openclsdk-devguide-enabling-debugging-in-opencl-runtime
    std::string projectorSource
        = io::fileToString(
            io::xprintf("%s/opencl/projector.cl", this->xpath.c_str()));
    cl::Program program(*context, projectorSource);
    if(program.build({ *device }, "-g") != CL_SUCCESS)
    {
        LOGE << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(*device);
        return -1;
    }
    /*
    if(program.build({ *device }, "-g") != CL_SUCCESS)
        {
            LOGE << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(*device);
            return -1;
        }
    */
    double* P = matrix.getPtr();
    std::array<double, 3> sourcePosition = matrix.sourcePosition();
    std::array<double, 3> normalToDetector = matrix.normalToDetector();
    //    cl::Buffer buffer_P(*context, CL_MEM_COPY_HOST_PTR, sizeof(double) * 12, (void*)P);
    cl::Buffer buffer_projection(*context, CL_MEM_COPY_HOST_PTR, sizeof(float) * pdimx * pdimy,
                                 (void*)projection);

    // OpenCL 1.2 got rid of KernelFunctor
    // https://forums.khronos.org/showthread.php/8317-cl-hpp-KernelFunctor-gone-replaced-with-KernelFunctorGlobal
    // https://stackoverflow.com/questions/23992369/what-should-i-use-instead-of-clkernelfunctor/54344990#54344990
    cl::make_kernel<cl::Buffer&, cl::Buffer&, cl_double16&, cl_double4&, cl_double4&, cl_int4&,
                    cl_double4&, cl_int2&, float&>
        FLOATcutting_voxel_project(cl::Kernel(program, "FLOATcutting_voxel_project"));
    cl_double16 PM({ P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10], P[11], 0.0,
                     0.0, 0.0, 0.0 });
    cl_double4 SOURCEPOSITION({ sourcePosition[0], sourcePosition[1], sourcePosition[2], 0.0 });
    cl_double4 NORMALTODETECTOR(
        { normalToDetector[0], normalToDetector[1], normalToDetector[2], 0.0 });
    cl::EnqueueArgs eargs(*Q, cl::NDRange(vdimz, vdimy, vdimx));
    cl_int4 vdims({ int(vdimx), int(vdimy), int(vdimz), 0 });
    cl_double4 voxelSizes({ 1.0, 1.0, 1.0, 0.0 });
    cl_int2 pdims({ int(pdimx), int(pdimy) });
    FLOATcutting_voxel_project(eargs, *volumeBuffer, buffer_projection, PM, SOURCEPOSITION,
                               NORMALTODETECTOR, vdims, voxelSizes, pdims, scalingFactor)
        .wait();

    Q->enqueueReadBuffer(buffer_projection, CL_TRUE, 0, sizeof(float) * pdimx * pdimy, projection);
    return 0;
}

} // namespace CTL
