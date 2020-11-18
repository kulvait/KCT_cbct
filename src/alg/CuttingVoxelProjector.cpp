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
    clFile = io::xprintf("%s/opencl/allsources.cl", this->xpath.c_str());
    if(centerVoxelProjector)
    {
        io::concatenateTextFiles(
            clFile, true,
            { io::xprintf("%s/opencl/utils.cl", this->xpath.c_str()),
              io::xprintf("%s/opencl/centerVoxelProjector.cl", this->xpath.c_str()) });

    } else
    {
        io::concatenateTextFiles(
            clFile, true,
            {
                io::xprintf("%s/opencl/utils.cl", this->xpath.c_str()),
                io::xprintf("%s/opencl/projector_sidon.cl", this->xpath.c_str()),
                io::xprintf("%s/opencl/projector.cl", this->xpath.c_str()),
                io::xprintf("%s/opencl/projector_tt.cl", this->xpath.c_str()),
                io::xprintf("%s/opencl/backprojector.cl", this->xpath.c_str()),
                io::xprintf("%s/opencl/backprojector_tt.cl", this->xpath.c_str()),
                io::xprintf("%s/opencl/backprojector_sidon.cl", this->xpath.c_str()),
                io::xprintf("%s/opencl/backprojector_minmax.cl", this->xpath.c_str()),

            });
    }
    std::string projectorSource = io::fileToString(clFile);
    cl::Program program(*context, projectorSource);
    LOGI << io::xprintf("Building file %s.", clFile.c_str());
    if(debug)
    {
        std::string options = io::xprintf("-g -s \"%s\" -Werror", clFile.c_str());
        LOGD << io::xprintf("Parsing DEBUG options %s.", options.c_str());
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
        // Projectors
        projector = std::make_shared<
            cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&, cl_double16&, cl_double3&,
                            cl_double3&, cl_int3&, cl_double3&, cl_int2&, float&>>(
            cl::Kernel(program, "FLOATcutting_voxel_project"));
        projector_ta3 = std::make_shared<
            cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&, cl_double16&, cl_double3&,
                            cl_double3&, cl_int3&, cl_double3&, cl_int2&, float&>>(
            cl::Kernel(program, "FLOATtta3_project"));
        projector_sidon = std::make_shared<
            cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&, cl_double16&, cl_double3&,
                            cl_double3&, cl_int3&, cl_double3&, cl_int2&, float&, cl_uint2&>>(
            cl::Kernel(program, "FLOATsidon_project"));
        // Backprojectors
        FLOATbackprojector_sidon = std::make_shared<
            cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&, cl_double16&, cl_double3&,
                            cl_double3&, cl_int3&, cl_double3&, cl_int2&, float&, cl_uint2&>>(
            cl::Kernel(program, "FLOATsidon_backproject"));
        FLOATta3_backproject = std::make_shared<
            cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&, cl_double16&, cl_double3&,
                            cl_double3&, cl_int3&, cl_double3&, cl_int2&, float&>>(
            cl::Kernel(program, "FLOATtta3_backproject"));
        FLOATcutting_voxel_backproject = std::make_shared<
            cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&, cl_double16&, cl_double3&,
                            cl_double3&, cl_int3&, cl_double3&, cl_int2&, float&>>(
            cl::Kernel(program, "FLOATcutting_voxel_backproject"));
        FLOATcutting_voxel_minmaxbackproject = std::make_shared<
            cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&, cl_double16&, cl_double3&,
                            cl_double3&, cl_int3&, cl_double3&, cl_int2&, float&>>(
            cl::Kernel(program, "FLOATcutting_voxel_minmaxbackproject"));
        // Utils
        scalingProjectionsCos
            = std::make_shared<cl::make_kernel<cl::Buffer&, unsigned int&, cl_double16&,
                                               cl_double3&, cl_double3&, cl_uint2&, float&>>(
                cl::Kernel(program, "FLOATrescale_projections_cos"));
        scalingProjectionsExact
            = std::make_shared<cl::make_kernel<cl::Buffer&, unsigned int&, cl_uint2&, cl_double2&,
                                               cl_double2&, double&>>(
                cl::Kernel(program, "FLOATrescale_projections_exact"));
        scalingBackprojectionsExact
            = std::make_shared<cl::make_kernel<cl::Buffer&, unsigned int&, cl_uint2&, cl_double2&,
                                               cl_double2&, double&>>(
                cl::Kernel(program, "FLOATrescale_backprojections_exact"));
        FLOAT_CopyVector = std::make_shared<cl::make_kernel<cl::Buffer&, cl::Buffer&>>(
            cl::Kernel(program, "FLOAT_copy_vector"));
        FLOAT_addIntoFirstVectorSecondVectorScaled
            = std::make_shared<cl::make_kernel<cl::Buffer&, cl::Buffer&, float&>>(
                cl::Kernel(program, "FLOAT_add_into_first_vector_second_vector_scaled"));
        NormSquare = std::make_shared<cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&>>(
            cl::Kernel(program, "vector_NormSquarePartial"));
        SubstituteLowerThan = std::make_shared<cl::make_kernel<cl::Buffer&, float&, float&>>(
            cl::Kernel(program, "FLOAT_substitute_lower_than"));
        SubstituteGreaterThan = std::make_shared<cl::make_kernel<cl::Buffer&, float&, float&>>(
            cl::Kernel(program, "FLOAT_substitute_greater_than"));
        ZeroInfinity = std::make_shared<cl::make_kernel<cl::Buffer&>>(
            cl::Kernel(program, "FLOAT_zero_infinity"));
    }
    Q = std::make_shared<cl::CommandQueue>(*context, *device);
    return 0;
}

int CuttingVoxelProjector::initializeOrUpdateVolumeBuffer(uint32_t volumeSizeX,
                                                          uint32_t volumeSizeY,
                                                          uint32_t volumeSizeZ,
                                                          float* volumeArray)
{
    cl_int err = CL_SUCCESS;
    std::string msg;
    this->volumeSizeX = volumeSizeX;
    this->volumeSizeY = volumeSizeY;
    this->volumeSizeZ = volumeSizeZ;
    this->totalVolumeSize = volumeSizeX * volumeSizeY * volumeSizeZ;
    vdims = cl_int3({ int(volumeSizeX), int(volumeSizeY), int(volumeSizeZ) });
    if(volumeBuffer != nullptr)
    {
        if(this->totalVolumeBufferSize == sizeof(float) * this->totalVolumeSize)
        {
            if(volumeArray != nullptr)
            {
                err = Q->enqueueWriteBuffer(*volumeBuffer, CL_TRUE, 0, totalVolumeBufferSize,
                                            (void*)volumeArray);
            }
        } else
        {
            this->totalVolumeBufferSize = sizeof(float) * this->totalVolumeSize;
            if(volumeArray != nullptr)
            {
                volumeBuffer = std::make_shared<cl::Buffer>(
                    *context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, totalVolumeBufferSize,
                    (void*)volumeArray, &err);
            } else
            {
                volumeBuffer = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                            totalVolumeBufferSize, nullptr, &err);
            }
        }
    } else
    {
        this->totalVolumeBufferSize = sizeof(float) * this->totalVolumeSize;
        if(volumeArray != nullptr)
        {
            volumeBuffer
                = std::make_shared<cl::Buffer>(*context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE,
                                               totalVolumeBufferSize, (void*)volumeArray, &err);
        } else
        {
            volumeBuffer = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                        totalVolumeBufferSize, nullptr, &err);
        }
    }
    if(err != CL_SUCCESS)
    {
        msg = io::xprintf("Unsucessful initialization of Volume with error code %d!", err);
        LOGE << msg;
        throw std::runtime_error(msg);
    }
    return 0;
}

int CuttingVoxelProjector::initializeOrUpdateProjectionBuffer(uint32_t projectionSizeX,
                                                              uint32_t projectionSizeY,
                                                              uint32_t projectionSizeZ,
                                                              float* projectionArray)
{
    cl_int err = CL_SUCCESS;
    std::string msg;
    this->projectionSizeX = projectionSizeX;
    this->projectionSizeY = projectionSizeY;
    this->projectionSizeZ = projectionSizeZ;
    this->totalProjectionSize = projectionSizeX * projectionSizeY * projectionSizeZ;
    pdims = cl_int2({ int(projectionSizeX), int(projectionSizeY) });
    if(projectionBuffer != nullptr)
    {
        if(this->totalProjectionBufferSize == sizeof(float) * this->totalProjectionSize)
        {
            if(projectionArray != nullptr)
            {
                err = Q->enqueueWriteBuffer(*projectionBuffer, CL_TRUE, 0,
                                            totalProjectionBufferSize, (void*)projectionArray);
            }
        } else
        {
            this->totalProjectionBufferSize = sizeof(float) * this->totalProjectionSize;
            if(projectionArray != nullptr)
            {
                projectionBuffer = std::make_shared<cl::Buffer>(
                    *context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, totalProjectionBufferSize,
                    (void*)projectionArray, &err);
            } else
            {
                projectionBuffer = std::make_shared<cl::Buffer>(
                    *context, CL_MEM_READ_WRITE, totalProjectionBufferSize, nullptr, &err);
            }
        }
    } else
    {
        this->totalProjectionBufferSize = sizeof(float) * this->totalProjectionSize;
        if(projectionArray != nullptr)
        {
            projectionBuffer = std::make_shared<cl::Buffer>(
                *context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, totalProjectionBufferSize,
                (void*)projectionArray, &err);
        } else
        {
            projectionBuffer = std::make_shared<cl::Buffer>(
                *context, CL_MEM_READ_WRITE, totalProjectionBufferSize, nullptr, &err);
        }
    }
    if(err != CL_SUCCESS)
    {
        msg = io::xprintf("Unsucessful initialization of Projection with error code %d!", err);
        LOGE << msg;
        throw std::runtime_error(msg);
    }
    return 0;
}

int CuttingVoxelProjector::fillVolumeBufferByConstant(float constant)
{
    std::string msg;
    if(volumeBuffer == nullptr)
    {
        msg = io::xprintf(
            "Volume buffer is not yet initialized, call initializeVolumeBuffer first!");
        LOGE << msg;
        throw std::runtime_error(msg);
    }
    return Q->enqueueFillBuffer<cl_float>(*volumeBuffer, constant, 0, totalVolumeBufferSize);
}

int CuttingVoxelProjector::fillProjectionBufferByConstant(float constant)
{
    std::string msg;
    if(projectionBuffer == nullptr)
    {
        msg = io::xprintf(
            "Projection buffer is not yet initialized, call initializeProjectionBuffer first!");
        LOGE << msg;
        throw std::runtime_error(msg);
    }
    return Q->enqueueFillBuffer<cl_float>(*projectionBuffer, constant, 0,
                                          totalProjectionBufferSize);
}

double CuttingVoxelProjector::normSquare(float* v, uint32_t pdimx, uint32_t pdimy)
{
    size_t vecsize = sizeof(float) * pdimx * pdimy;
    if(tmpBuffer == nullptr || vecsize != tmpBuffer_size)
    {
        tmpBuffer_size = vecsize;
        tmpBuffer = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                 tmpBuffer_size, (void*)v);
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
    if(vecsize != totalProjectionBufferSize)
    {
        std::string msg = "Ca not compare buffers of incompatible dimensions.";
        throw new std::runtime_error(msg);
    }

    if(tmpBuffer == nullptr || vecsize != tmpBuffer_size)
    {
        tmpBuffer_size = vecsize;
        tmpBuffer = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                 tmpBuffer_size, (void*)v);
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
    if(exactProjectionScaling)
    {
        double normalProjectionX, normalProjectionY, projection45X, projection45Y, fX, fY;
        double sourceToDetector;
        std::array<double, 3> sourcePosition = matrix.sourcePosition();
        std::array<double, 3> normalToDetector = matrix.normalToDetector();
        std::array<double, 3> tangentToDetector = matrix.tangentToDetectorYDirection();
        matrix.project(sourcePosition[0] - normalToDetector[0] + tangentToDetector[0],
                       sourcePosition[1] - normalToDetector[1] + tangentToDetector[1],
                       sourcePosition[2] - normalToDetector[2] + tangentToDetector[2],
                       &projection45X, &projection45Y);
        matrix.project(
            sourcePosition[0] - normalToDetector[0], sourcePosition[1] - normalToDetector[1],
            sourcePosition[2] - normalToDetector[2], &normalProjectionX, &normalProjectionY);
        fX = (projection45X - normalProjectionX) * pixelSizeX;
        fY = (projection45Y - normalProjectionY) * pixelSizeY;
        sourceToDetector = std::sqrt(fX * fX + fY * fY);
        return projectExact(projection, pdimx, pdimy, normalProjectionX, normalProjectionY,
                            sourceToDetector, matrix);
    } else
    {
        return projectCos(projection, pdimx, pdimy, matrix, scalingFactor);
    }
}

int CuttingVoxelProjector::projectCos(float* projection,
                                      uint32_t pdimx,
                                      uint32_t pdimy,
                                      matrix::ProjectionMatrix matrix,
                                      float scalingFactor)
{
    double* P = matrix.getPtr();
    cl_int err;
    std::array<double, 3> sourcePosition = matrix.sourcePosition();
    std::array<double, 3> normalToDetector = matrix.normalToDetector();
    CTL::matrix::SquareMatrix CME(4,
                                  { P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9],
                                    P[10], P[11], sourcePosition[0], sourcePosition[1],
                                    sourcePosition[2], 1.0 });
    matrix::LUDoolittleForm lu = matrix::LUDoolittleForm::LUDecomposeDoolittle(CME, 0.001);
    matrix::SquareMatrix invertedCameraMatrix = lu.inverseMatrix();
    double* icm = invertedCameraMatrix.getPtr();
    //    cl::Buffer buffer_P(*context, CL_MEM_COPY_HOST_PTR, sizeof(double) * 12, (void*)P);
    initializeOrUpdateProjectionBuffer(pdimx, pdimy, 1);
    fillProjectionBufferByConstant(0.0f);

    cl_double16 PM({ P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10], P[11], 0.0,
                     0.0, 0.0, 0.0 });
    cl_double16 ICM({ icm[0], icm[1], icm[2], icm[3], icm[4], icm[5], icm[6], icm[7], icm[8],
                      icm[9], icm[10], icm[11], icm[12], icm[13], icm[14], icm[15] });
    cl_double3 SOURCEPOSITION({ sourcePosition[0], sourcePosition[1], sourcePosition[2] });
    cl_double3 NORMALTODETECTOR({ normalToDetector[0], normalToDetector[1], normalToDetector[2] });
    cl::EnqueueArgs eargs(*Q, cl::NDRange(volumeSizeZ, volumeSizeY, volumeSizeX));
    cl_int2 pdims({ int(pdimx), int(pdimy) });
    cl_uint2 pdims_uint({ pdimx, pdimy });
    unsigned int offset = 0;
    float scalingOne = 1.0;
    (*projector)(eargs, *volumeBuffer, *projectionBuffer, offset, PM, SOURCEPOSITION,
                 NORMALTODETECTOR, vdims, voxelSizes, pdims, scalingOne)
        .wait();
    cl::EnqueueArgs eargs2(*Q, cl::NDRange(pdimx, pdimy));
    (*scalingProjectionsCos)(eargs2, *projectionBuffer, offset, ICM, SOURCEPOSITION,
                             NORMALTODETECTOR, pdims_uint, scalingFactor)
        .wait();

    err = Q->enqueueReadBuffer(*projectionBuffer, CL_TRUE, 0, sizeof(float) * pdimx * pdimy,
                               projection);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful writte buffer to the projection variable, code %d!", err);
        return -1;
    }
    return 0;
}

int CuttingVoxelProjector::projectorWithoutScaling(float* projection,
                                                   uint32_t pdimx,
                                                   uint32_t pdimy,
                                                   double normalProjectionX,
                                                   double normalProjectionY,
                                                   double sourceToDetector,
                                                   matrix::ProjectionMatrix matrix)
{
    cl::EnqueueArgs eargs(*Q, cl::NDRange(volumeSizeZ, volumeSizeY, volumeSizeX));
    cl::EnqueueArgs eargs2(*Q, cl::NDRange(pdimx, pdimy));
    cl_int err;
    initializeOrUpdateProjectionBuffer(pdimx, pdimy, 1);
    fillProjectionBufferByConstant(0.0f);
    unsigned int offset = 0;
    double* P = matrix.getPtr();
    cl_double16 PM({ P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10], P[11], 0.0,
                     0.0, 0.0, 0.0 });
    std::array<double, 3> sourcePosition = matrix.sourcePosition();
    cl_double3 SOURCEPOSITION({ sourcePosition[0], sourcePosition[1], sourcePosition[2] });
    std::array<double, 3> normalToDetector = matrix.normalToDetector();
    cl_double3 NORMALTODETECTOR({ normalToDetector[0], normalToDetector[1], normalToDetector[2] });
    cl_int2 pdims({ int(pdimx), int(pdimy) });
    cl_uint2 pdims_uint({ pdimx, pdimy });
    cl_double2 normalProjection({ normalProjectionX, normalProjectionY });
    float scalingOne = 1.0;
    (*projector)(eargs, *volumeBuffer, *projectionBuffer, offset, PM, SOURCEPOSITION,
                 NORMALTODETECTOR, vdims, voxelSizes, pdims, scalingOne)
        .wait();

    err = Q->enqueueReadBuffer(*projectionBuffer, CL_TRUE, 0, sizeof(float) * pdimx * pdimy,
                               projection);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful writte buffer to the projection variable, code %d!", err);
        return -1;
    }
    return 0;
}

int CuttingVoxelProjector::projectExact(float* projection,
                                        uint32_t pdimx,
                                        uint32_t pdimy,
                                        double normalProjectionX,
                                        double normalProjectionY,
                                        double sourceToDetector,
                                        matrix::ProjectionMatrix matrix)
{
    cl::EnqueueArgs eargs(*Q, cl::NDRange(volumeSizeZ, volumeSizeY, volumeSizeX));
    cl::EnqueueArgs eargs2(*Q, cl::NDRange(pdimx, pdimy));
    cl_int err;
    initializeOrUpdateProjectionBuffer(pdimx, pdimy, 1);
    fillProjectionBufferByConstant(0.0f);
    unsigned int offset = 0;
    double* P = matrix.getPtr();
    cl_double16 PM({ P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10], P[11], 0.0,
                     0.0, 0.0, 0.0 });
    std::array<double, 3> sourcePosition = matrix.sourcePosition();
    cl_double3 SOURCEPOSITION({ sourcePosition[0], sourcePosition[1], sourcePosition[2] });
    std::array<double, 3> normalToDetector = matrix.normalToDetector();
    cl_double3 NORMALTODETECTOR({ normalToDetector[0], normalToDetector[1], normalToDetector[2] });
    cl_int2 pdims({ int(pdimx), int(pdimy) });
    cl_uint2 pdims_uint({ pdimx, pdimy });
    cl_double2 normalProjection({ normalProjectionX, normalProjectionY });
    float scalingOne = 1.0;
    (*projector)(eargs, *volumeBuffer, *projectionBuffer, offset, PM, SOURCEPOSITION,
                 NORMALTODETECTOR, vdims, voxelSizes, pdims, scalingOne)
        .wait();
    (*scalingProjectionsExact)(eargs2, *projectionBuffer, offset, pdims_uint, normalProjection,
                               pixelSizes, sourceToDetector)
        .wait();

    err = Q->enqueueReadBuffer(*projectionBuffer, CL_TRUE, 0, sizeof(float) * pdimx * pdimy,
                               projection);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful writte buffer to the projection variable, code %d!", err);
        return -1;
    }
    return 0;
}

int CuttingVoxelProjector::projectSiddon(float* projection,
                                         uint32_t pdimx,
                                         uint32_t pdimy,
                                         matrix::ProjectionMatrix matrix,
                                         float scalingFactor,
                                         uint32_t probesPerEdge)
{
    double* P = matrix.getPtr();
    cl_int err;
    initializeOrUpdateProjectionBuffer(pdimx, pdimy, 1);
    fillProjectionBufferByConstant(0.0f);
    std::array<double, 3> sourcePosition = matrix.sourcePosition();
    std::array<double, 3> normalToDetector = matrix.normalToDetector();
    CTL::matrix::SquareMatrix CME(4,
                                  { P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9],
                                    P[10], P[11], sourcePosition[0], sourcePosition[1],
                                    sourcePosition[2], 1.0 });
    matrix::LUDoolittleForm lu = matrix::LUDoolittleForm::LUDecomposeDoolittle(CME, 0.001);
    matrix::SquareMatrix invertedCameraMatrix = lu.inverseMatrix();
    double* icm = invertedCameraMatrix.getPtr();
    //    cl::Buffer buffer_P(*context, CL_MEM_COPY_HOST_PTR, sizeof(double) * 12, (void*)P);
    cl_double16 PM({ P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10], P[11], 0.0,
                     0.0, 0.0, 0.0 });
    cl_double16 ICM({ icm[0], icm[1], icm[2], icm[3], icm[4], icm[5], icm[6], icm[7], icm[8],
                      icm[9], icm[10], icm[11], icm[12], icm[13], icm[14], icm[15] });
    cl_double3 SOURCEPOSITION({ sourcePosition[0], sourcePosition[1], sourcePosition[2] });
    cl_double3 NORMALTODETECTOR({ normalToDetector[0], normalToDetector[1], normalToDetector[2] });
    cl_int2 pdims({ int(pdimx), int(pdimy) });
    cl_uint2 pixelGranularity({ probesPerEdge, probesPerEdge });
    unsigned int offset = 0;
    float scalingOne = 1.0;
    cl::EnqueueArgs eargs(*Q, cl::NDRange(pdimx, pdimy));
    (*projector_sidon)(eargs, *volumeBuffer, *projectionBuffer, offset, ICM, SOURCEPOSITION,
                       NORMALTODETECTOR, vdims, voxelSizes, pdims, scalingOne, pixelGranularity)
        .wait();

    err = Q->enqueueReadBuffer(*projectionBuffer, CL_TRUE, 0, sizeof(float) * pdimx * pdimy,
                               projection);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful writte buffer to the projection variable, code %d!", err);
        return -1;
    }
    return 0;
}

int CuttingVoxelProjector::projectTA3(float* projection,
                                      uint32_t pdimx,
                                      uint32_t pdimy,
                                      double normalProjectionX,
                                      double normalProjectionY,
                                      double sourceToDetector,
                                      matrix::ProjectionMatrix matrix)
{
    cl::EnqueueArgs eargs(*Q, cl::NDRange(volumeSizeZ, volumeSizeY, volumeSizeX));
    cl::EnqueueArgs eargs2(*Q, cl::NDRange(pdimx, pdimy));
    cl_int err;
    initializeOrUpdateProjectionBuffer(pdimx, pdimy, 1);
    fillProjectionBufferByConstant(0.0f);
    unsigned int offset = 0;
    double* P = matrix.getPtr();
    cl_double16 PM({ P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10], P[11], 0.0,
                     0.0, 0.0, 0.0 });
    std::array<double, 3> sourcePosition = matrix.sourcePosition();
    cl_double3 SOURCEPOSITION({ sourcePosition[0], sourcePosition[1], sourcePosition[2] });
    std::array<double, 3> normalToDetector = matrix.normalToDetector();
    cl_double3 NORMALTODETECTOR({ normalToDetector[0], normalToDetector[1], normalToDetector[2] });
    cl_int2 pdims({ int(pdimx), int(pdimy) });
    cl_uint2 pdims_uint({ pdimx, pdimy });
    cl_double2 normalProjection({ normalProjectionX, normalProjectionY });
    float scalingOne = 1.0;
    (*projector_ta3)(eargs, *volumeBuffer, *projectionBuffer, offset, PM, SOURCEPOSITION,
                     NORMALTODETECTOR, vdims, voxelSizes, pdims, scalingOne)
        .wait();

    err = Q->enqueueReadBuffer(*projectionBuffer, CL_TRUE, 0, sizeof(float) * pdimx * pdimy,
                               projection);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful writte buffer to the projection variable, code %d!", err);
        return -1;
    }
    return 0;
}

std::vector<cl_double16>
CuttingVoxelProjector::invertProjectionMatrices(std::vector<matrix::ProjectionMatrix> CM)
{
    std::vector<cl_double16> inverseProjectionMatrices;
    for(std::size_t i = 0; i != CM.size(); i++)
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
CuttingVoxelProjector::computeScalingFactors(std::vector<matrix::ProjectionMatrix> PM)
{
    std::vector<float> scalingFactors;
    double xoveryspacing = pixelSizeX / pixelSizeY;
    double yoverxspacing = pixelSizeY / pixelSizeX;
    for(std::size_t i = 0; i != PM.size(); i++)
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

int CuttingVoxelProjector::backproject(float* volume,
                                       uint32_t vdimx,
                                       uint32_t vdimy,
                                       uint32_t vdimz,
                                       std::vector<matrix::ProjectionMatrix>& CMS,
                                       uint64_t baseOffset)
{
    std::vector<cl_double16> invertedProjectionMatrices = invertProjectionMatrices(CMS);
    std::vector<float> scalingFactors = computeScalingFactors(CMS);
    initializeOrUpdateVolumeBuffer(vdimx, vdimy, vdimz);
    fillVolumeBufferByConstant(0.0f);
    cl::EnqueueArgs eargs(*Q, cl::NDRange(volumeSizeZ, volumeSizeY, volumeSizeX));
    cl::EnqueueArgs eargs2(*Q, cl::NDRange(projectionSizeX, projectionSizeY));
    cl_uint2 pdims_uint({ projectionSizeX, projectionSizeY });
    unsigned int frameSize = projectionSizeX * projectionSizeY;
    if(tmpBuffer == nullptr || totalProjectionBufferSize > tmpBuffer_size)
    {
        tmpBuffer = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                 totalProjectionBufferSize, nullptr);
        tmpBuffer_size = totalProjectionBufferSize;
    }
    copyFloatVector(*projectionBuffer, *tmpBuffer, totalProjectionSize);
    double normalProjectionX, normalProjectionY, projection45X, projection45Y, fX, fY;
    double sourceToDetector;
    for(std::size_t i = 0; i != projectionSizeZ; i++)
    {
        matrix::ProjectionMatrix mat = CMS[i];
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
        fX = (projection45X - normalProjectionX) * pixelSizeX;
        fY = (projection45Y - normalProjectionY) * pixelSizeY;
        sourceToDetector = std::sqrt(fX * fX + fY * fY);
        cl_double2 normalProjection({ normalProjectionX, normalProjectionY });
        double* P = mat.getPtr();
        cl_double16 PM({ P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10], P[11],
                         0.0, 0.0, 0.0, 0.0 });
        cl_double16 ICM = invertedProjectionMatrices[i];
        cl_double3 SOURCEPOSITION({ sourcePosition[0], sourcePosition[1], sourcePosition[2] });
        cl_double3 NORMALTODETECTOR(
            { normalToDetector[0], normalToDetector[1], normalToDetector[2] });
        unsigned int offset = baseOffset + i * frameSize;
        if(useSidonProjector)
        {
            (*FLOATbackprojector_sidon)(eargs2, *volumeBuffer, *tmpBuffer, offset, ICM,
                                        SOURCEPOSITION, NORMALTODETECTOR, vdims, voxelSizes, pdims,
                                        FLOATONE, pixelGranularity);
        } else if(useTTProjector)
        {
            (*FLOATta3_backproject)(eargs, *volumeBuffer, *tmpBuffer, offset, PM, SOURCEPOSITION,
                                    NORMALTODETECTOR, vdims, voxelSizes, pdims, FLOATONE);
        } else
        {
            if(exactProjectionScaling)
            {
                (*scalingProjectionsExact)(eargs2, *tmpBuffer, offset, pdims_uint, normalProjection,
                                           pixelSizes, sourceToDetector);
            } else
            {
                (*scalingProjectionsCos)(eargs2, *tmpBuffer, offset, ICM, SOURCEPOSITION,
                                         NORMALTODETECTOR, pdims_uint, scalingFactor);
            }
            (*FLOATcutting_voxel_backproject)(eargs, *volumeBuffer, *tmpBuffer, offset, PM,
                                              SOURCEPOSITION, NORMALTODETECTOR, vdims, voxelSizes,
                                              pdims, FLOATONE);
        }
    }
    cl_int err = Q->enqueueReadBuffer(*volumeBuffer, CL_TRUE, 0, totalVolumeBufferSize, volume);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful writte buffer to the projection variable, code %d!", err);
        return -1;
    }
    return 0;
}

int CuttingVoxelProjector::backproject_minmax(float* volume,
                                              uint32_t vdimx,
                                              uint32_t vdimy,
                                              uint32_t vdimz,
                                              std::vector<matrix::ProjectionMatrix>& CMS,
                                              uint64_t baseOffset)
{
    std::vector<cl_double16> invertedProjectionMatrices = invertProjectionMatrices(CMS);
    std::vector<float> scalingFactors = computeScalingFactors(CMS);
    initializeOrUpdateVolumeBuffer(vdimx, vdimy, vdimz);
    fillVolumeBufferByConstant(std::numeric_limits<float>::infinity());
    cl::EnqueueArgs eargs(*Q, cl::NDRange(volumeSizeZ, volumeSizeY, volumeSizeX));
    cl::EnqueueArgs eargs2(*Q, cl::NDRange(projectionSizeX, projectionSizeY));
    cl::EnqueueArgs eargsVolumeFlat(*Q, cl::NDRange(totalVolumeSize));
    cl_uint2 pdims_uint({ projectionSizeX, projectionSizeY });
    unsigned int frameSize = projectionSizeX * projectionSizeY;
    if(tmpBuffer == nullptr || totalProjectionBufferSize > tmpBuffer_size)
    {
        tmpBuffer = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                 totalProjectionBufferSize, nullptr);
        tmpBuffer_size = totalProjectionBufferSize;
    }
    copyFloatVector(*projectionBuffer, *tmpBuffer, totalProjectionSize);
    double normalProjectionX, normalProjectionY, projection45X, projection45Y, fX, fY;
    double sourceToDetector;
    for(std::size_t i = 0; i != projectionSizeZ; i++)
    {
        matrix::ProjectionMatrix mat = CMS[i];
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
        fX = (projection45X - normalProjectionX) * pixelSizeX;
        fY = (projection45Y - normalProjectionY) * pixelSizeY;
        sourceToDetector = std::sqrt(fX * fX + fY * fY);
        cl_double2 normalProjection({ normalProjectionX, normalProjectionY });
        double* P = mat.getPtr();
        cl_double16 PM({ P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10], P[11],
                         0.0, 0.0, 0.0, 0.0 });
        cl_double3 SOURCEPOSITION({ sourcePosition[0], sourcePosition[1], sourcePosition[2] });
        cl_double3 NORMALTODETECTOR(
            { normalToDetector[0], normalToDetector[1], normalToDetector[2] });
        unsigned int offset = baseOffset + i * frameSize;
        (*scalingBackprojectionsExact)(eargs2, *tmpBuffer, offset, pdims_uint, normalProjection,
                                       pixelSizes, sourceToDetector);
        (*FLOATcutting_voxel_minmaxbackproject)(eargs, *volumeBuffer, *tmpBuffer, offset, PM,
                                                SOURCEPOSITION, NORMALTODETECTOR, vdims, voxelSizes,
                                                pdims, FLOATONE);
    }
    (*ZeroInfinity)(eargsVolumeFlat, *volumeBuffer);
    cl_int err = Q->enqueueReadBuffer(*volumeBuffer, CL_TRUE, 0, totalVolumeBufferSize, volume);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful writte buffer to the projection variable, code %d!", err);
        return -1;
    }
    return 0;
}

int CuttingVoxelProjector::copyFloatVector(cl::Buffer& from, cl::Buffer& to, unsigned int size)
{
    cl::EnqueueArgs eargs(*Q, cl::NDRange(size));
    (*FLOAT_CopyVector)(eargs, from, to).wait();
    return 0;
}

} // namespace CTL
