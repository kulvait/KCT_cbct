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

    // Debug info
    // https://software.intel.com/en-us/openclsdk-devguide-enabling-debugging-in-opencl-runtime
    std::string clFile;
    if(centerVoxelProjector)
    {
        clFile = io::xprintf("%s/opencl/centerVoxelProjector.cl", this->xpath.c_str());
    } else
    {
        clFile = io::xprintf("%s/opencl/allsources.cl", this->xpath.c_str());
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
    LOGI << "Build sucesfull";
    // OpenCL 1.2 got rid of KernelFunctor
    // https://forums.khronos.org/showthread.php/8317-cl-hpp-KernelFunctor-gone-replaced-with-KernelFunctorGlobal
    // https://stackoverflow.com/questions/23992369/what-should-i-use-instead-of-clkernelfunctor/54344990#54344990
    if(centerVoxelProjector)
    {
        projector = std::make_shared<
            cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&, cl_double16&, cl_double4&,
                            cl_double4&, cl_int4&, cl_double4&, cl_int2&, float&>>(
            cl::Kernel(program, "FLOATcenter_voxel_project"));

    } else
    {
        projector = std::make_shared<
            cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&, cl_double16&, cl_double3&,
                            cl_double3&, cl_int3&, cl_double3&, cl_int2&, float&>>(
            cl::Kernel(program, "FLOATcutting_voxel_project"));
        FLOAT_addIntoFirstVectorSecondVectorScaled
            = std::make_shared<cl::make_kernel<cl::Buffer&, cl::Buffer&, float&>>(
                cl::Kernel(program, "FLOAT_add_into_first_vector_second_vector_scaled"));
        NormSquare = std::make_shared<cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&>>(
            cl::Kernel(program, "vector_NormSquarePartial"));
    }
    Q = std::make_shared<cl::CommandQueue>(*context, *device);
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
double CuttingVoxelProjector::normSquare(float* v, uint32_t pdimx, uint32_t pdimy)
{
    size_t vecsize = sizeof(float) * pdimx * pdimy;
    if(tmpBuffer == nullptr || vecsize != tmpBuffer_size)
    {
        tmpBuffer_size = vecsize;
        tmpBuffer = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                 projectionBuffer_size, (void*)v);
    } else
    {
        Q->enqueueWriteBuffer(*tmpBuffer, CL_TRUE, 0, tmpBuffer_size, (void*)v);
    }
    cl::Buffer onedouble(*context, CL_MEM_READ_WRITE, sizeof(double), nullptr);
    double sum;
    uint32_t framesize = pdimx * pdimy;
    cl::EnqueueArgs eargs(*Q, cl::NDRange(1));
    (*NormSquare)(eargs, *tmpBuffer, onedouble, framesize).wait();
    Q->enqueueReadBuffer(onedouble, CL_TRUE, 0, sizeof(double), &sum);
    return sum;
}

double CuttingVoxelProjector::normSquareDifference(float* v, uint32_t pdimx, uint32_t pdimy)
{
    size_t vecsize = sizeof(float) * pdimx * pdimy;
    if(projectionBuffer == nullptr)
    {
        std::string msg = "Comparing to empty buffer is not possible.";
        LOGE << msg;
        throw new std::runtime_error(msg);
    }
    if(vecsize != projectionBuffer_size)
    {
        std::string msg = "Ca not compare buffers of incompatible dimensions.";
        throw new std::runtime_error(msg);
    }

    if(tmpBuffer == nullptr || vecsize != tmpBuffer_size)
    {
        tmpBuffer_size = vecsize;
        tmpBuffer = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                 projectionBuffer_size, (void*)v);
    } else
    {
        Q->enqueueWriteBuffer(*tmpBuffer, CL_TRUE, 0, tmpBuffer_size, (void*)v);
    }
    uint32_t framesize = pdimx * pdimy;
    cl::EnqueueArgs eargs(*Q, cl::NDRange(framesize));
    float factor = -1.0;
    (*FLOAT_addIntoFirstVectorSecondVectorScaled)(eargs, *tmpBuffer, *projectionBuffer, factor)
        .wait();
    cl::Buffer onedouble(*context, CL_MEM_READ_WRITE, sizeof(double), nullptr);
    double sum;
    cl::EnqueueArgs ear(*Q, cl::NDRange(1));
    (*NormSquare)(ear, *tmpBuffer, onedouble, framesize).wait();
    Q->enqueueReadBuffer(onedouble, CL_TRUE, 0, sizeof(double), &sum);
    return sum;
}

int CuttingVoxelProjector::project(float* projection,
                                   uint32_t pdimx,
                                   uint32_t pdimy,
                                   matrix::ProjectionMatrix matrix,
                                   float scalingFactor)
{
    double* P = matrix.getPtr();
    std::array<double, 3> sourcePosition = matrix.sourcePosition();
    std::array<double, 3> normalToDetector = matrix.normalToDetector();
    //    cl::Buffer buffer_P(*context, CL_MEM_COPY_HOST_PTR, sizeof(double) * 12, (void*)P);
    size_t projectionSize = sizeof(float) * pdimx * pdimy;
    if(projectionBuffer == nullptr || projectionSize != projectionBuffer_size)
    {
        projectionBuffer_size = projectionSize;
        projectionBuffer = std::make_shared<cl::Buffer>(*context, CL_MEM_COPY_HOST_PTR,
                                                        projectionBuffer_size, (void*)projection);
    } else
    {
        Q->enqueueFillBuffer<cl_float>(*projectionBuffer, FLOATZERO, 0, projectionBuffer_size);
    }

    cl_double16 PM({ P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10], P[11], 0.0,
                     0.0, 0.0, 0.0 });
    cl_double3 SOURCEPOSITION({ sourcePosition[0], sourcePosition[1], sourcePosition[2] });
    cl_double3 NORMALTODETECTOR({ normalToDetector[0], normalToDetector[1], normalToDetector[2] });
    cl::EnqueueArgs eargs(*Q, cl::NDRange(vdimz, vdimy, vdimx));
    cl_int3 vdims({ int(vdimx), int(vdimy), int(vdimz) });
    cl_double3 voxelSizes({ 1.0, 1.0, 1.0 });
    cl_int2 pdims({ int(pdimx), int(pdimy) });
    unsigned int offset = 0;
    (*projector)(eargs, *volumeBuffer, *projectionBuffer, offset, PM, SOURCEPOSITION,
                 NORMALTODETECTOR, vdims, voxelSizes, pdims, scalingFactor)
        .wait();

    Q->enqueueReadBuffer(*projectionBuffer, CL_TRUE, 0, sizeof(float) * pdimx * pdimy, projection);
    return 0;
}

} // namespace CTL
