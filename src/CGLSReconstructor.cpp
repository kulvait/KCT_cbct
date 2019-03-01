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
    std::string sourceText;
    // clFile = io::xprintf("%s/opencl/centerVoxelProjector.cl", this->xpath.c_str());
    clFile = io::xprintf("%s/opencl/allsources.cl", this->xpath.c_str());
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
    FLOATcutting_voxel_project = std::make_shared<
        cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&, cl_double16&, cl_double4&,
                        cl_double4&, cl_int4&, cl_double4&, cl_int2&, float&>>(
        cl::Kernel(program, "FLOATcutting_voxel_project"));
    FLOATcutting_voxel_backproject = std::make_shared<
        cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&, cl_double16&, cl_double4&,
                        cl_double4&, cl_int4&, cl_double4&, cl_int2&, float&>>(
        cl::Kernel(program, "FLOATcutting_voxel_backproject"));
    Q = std::make_shared<cl::CommandQueue>(*context, *device);
    return 0;
}

int CGLSReconstructor::initializeVectors(float* projections, float* volume)
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
    v_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * XDIM, nullptr,
                                         &err);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
        return -1;
    }
    w_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * XDIM, nullptr,
                                         &err);

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
    // Initialize buffer c by projections data
    c_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                         sizeof(float) * BDIM, (void*)projections, &err);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
        return -1;
    }

    d_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * BDIM, nullptr,
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
double CGLSReconstructor::normXBuffer_frame_double(cl::Buffer& X)
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
double CGLSReconstructor::normXBuffer_barier_double(cl::Buffer& X)
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
 * Funcion computes the norm of the Buffer that has pdimx * pdimy * pdimz elements.
 *
 * @param X
 *
 * @return
 */
double CGLSReconstructor::normBBuffer_frame_double(cl::Buffer& B)
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
double CGLSReconstructor::normBBuffer_barier_double(cl::Buffer& B)
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
float CGLSReconstructor::normXBuffer_frame(cl::Buffer& X)
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
float CGLSReconstructor::normXBuffer_barier(cl::Buffer& X)
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
float CGLSReconstructor::normBBuffer_frame(cl::Buffer& B)
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
float CGLSReconstructor::normBBuffer_barier(cl::Buffer& B)
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

int CGLSReconstructor::backproject(cl::Buffer& B,
                                   cl::Buffer& X,
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
        cl_int4 vdims({ int(vdimx), int(vdimy), int(vdimz), 0 });
        cl_double3 voxelSizes({ 1.0, 1.0, 1.0 });
        cl_int2 pdims({ int(pdimx), int(pdimy) });
        cl::EnqueueArgs eargs(*Q, cl::NDRange(vdimz, vdimy, vdimx));
        unsigned int offset = i * frameSize;
        (*FLOATcutting_voxel_backproject)(eargs, X, B, offset, PM, SOURCEPOSITION, NORMALTODETECTOR,
                                          vdims, voxelSizes, pdims, scalingFactor)
            .wait();
    }
    return 0;
}

int CGLSReconstructor::project(cl::Buffer& X,
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
        cl_int4 vdims({ int(vdimx), int(vdimy), int(vdimz), 0 });
        cl_double3 voxelSizes({ 1.0, 1.0, 1.0 });
        cl_int2 pdims({ int(pdimx), int(pdimy) });
        unsigned int offset = i * frameSize;
        cl::EnqueueArgs eargs(*Q, cl::NDRange(vdimz, vdimy, vdimx));
        (*FLOATcutting_voxel_project)(eargs, X, B, offset, PM, SOURCEPOSITION, NORMALTODETECTOR,
                                      vdims, voxelSizes, pdims, scalingFactor)
            .wait();
    }
    return 0;
}

int CGLSReconstructor::copyFloatVector(cl::Buffer& from, cl::Buffer& to, unsigned int size)
{
    cl::EnqueueArgs eargs(*Q, cl::NDRange(size));
    (*FLOAT_CopyVector)(eargs, from, to).wait();
    return 0;
}

int CGLSReconstructor::addIntoFirstVectorSecondVectorScaled(cl::Buffer& a,
                                                            cl::Buffer& b,
                                                            float f,
                                                            unsigned int size)
{
    cl::EnqueueArgs eargs(*Q, cl::NDRange(size));
    (*FLOAT_addIntoFirstVectorSecondVectorScaled)(eargs, a, b, f).wait();
    return 0;
}

int CGLSReconstructor::addIntoFirstVectorScaledSecondVector(cl::Buffer& a,
                                                            cl::Buffer& b,
                                                            float f,
                                                            unsigned int size)
{
    cl::EnqueueArgs eargs(*Q, cl::NDRange(size));
    (*FLOAT_addIntoFirstVectorScaledSecondVector)(eargs, a, b, f).wait();
    return 0;
}

std::vector<matrix::ProjectionMatrix>
CGLSReconstructor::encodeProjectionMatrices(std::shared_ptr<io::DenProjectionMatrixReader> pm)
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
CGLSReconstructor::computeScalingFactors(std::vector<matrix::ProjectionMatrix> PM)
{
    std::vector<float> scalingFactors;
    for(std::size_t i = 0; i != pdimz; i++)
    {
        double x1, x2, y1, y2;
        matrix::ProjectionMatrix pm = PM[i];
        std::array<double, 3> sourcePosition = pm.sourcePosition();
        std::array<double, 3> normalToDetector = pm.normalToDetector();
        pm.project(sourcePosition[0] + normalToDetector[0], sourcePosition[1] + normalToDetector[1],
                   sourcePosition[2] + normalToDetector[2], &x1, &y1);
        pm.project(100.0, 100.0, 100.0, &x2, &y2);
        double xspacing2 = pixelSpacingX * pixelSpacingX;
        double yspacing2 = pixelSpacingY * pixelSpacingY;
        double distance
            = std::sqrt((x1 - x2) * (x1 - x2) * xspacing2 + (y1 - y2) * (y1 - y2) * yspacing2);
        double x = 100.0 - sourcePosition[0];
        double y = 100.0 - sourcePosition[1];
        double z = 100.0 - sourcePosition[2];
        double norma = std::sqrt(x * x + y * y + z * z);
        x /= norma;
        y /= norma;
        z /= norma;
        double cos = normalToDetector[0] * x + normalToDetector[1] * y + normalToDetector[2] * z;
        double theta = std::acos(cos);
        double distToDetector = std::abs(distance / std::tan(theta));
        double scalingFactor = distToDetector * distToDetector / pixelSpacingX / pixelSpacingY;
        scalingFactors.push_back(scalingFactor);
    }
    return scalingFactors;
}

void CGLSReconstructor::writeVolume(cl::Buffer& X, std::string path)
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

void CGLSReconstructor::writeProjections(cl::Buffer& B, std::string path)
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

void CGLSReconstructor::setTimepoint() { timepoint = std::chrono::steady_clock::now(); }

void CGLSReconstructor::reportTime(std::string msg)
{
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - timepoint);
    LOGW << io::xprintf("%s: %ds", msg.c_str(), duration.count());
    setTimepoint();
}

int CGLSReconstructor::reconstruct(std::shared_ptr<io::DenProjectionMatrixReader> matrices)
{
    LOGI << io::xprintf("WELCOME TO CGLS");
    reportTime("CGLS INIT");
    writeProjections(*b_buf, io::xprintf("/tmp/cgls/b.den"));
    writeVolume(*x_buf, io::xprintf("/tmp/cgls/x_0.den"));
    std::vector<matrix::ProjectionMatrix> PM = encodeProjectionMatrices(matrices);
    std::vector<float> scalingFactors = computeScalingFactors(PM);
    uint32_t iteration = 0;
    double norm, vnorm2_old, vnorm2_now, dnorm2_old, alpha, beta;
    double NB0 = std::sqrt(normBBuffer_barier_double(*b_buf));
    norm = NB0;
    project(*x_buf, *tmp_b_buf, PM, scalingFactors);
    reportTime("X_0 projection");
    addIntoFirstVectorSecondVectorScaled(*tmp_b_buf, *b_buf, -1.0, BDIM);
    norm = std::sqrt(normBBuffer_barier_double(*tmp_b_buf));
    LOGI << io::xprintf("Initial norm of b is %f and initial |Ax-b| is %f.", NB0, norm);
    // INITIALIZATION x_0 is initialized typically by zeros but in general by supplied array
    // c_0 is filled by b
    // v_0=w_0=BACKPROJECT(c_0)
    // writeProjections(*c_buf, io::xprintf("/tmp/cgls/c_0.den"));
    backproject(*c_buf, *v_buf, PM, scalingFactors);
    reportTime("v_0 backprojection");
    writeVolume(*v_buf, io::xprintf("/tmp/cgls/v_0.den"));
    vnorm2_old = normXBuffer_barier_double(*v_buf);
    copyFloatVector(*v_buf, *w_buf, XDIM);
    // writeVolume(*w_buf, io::xprintf("/tmp/cgls/w_0.den"));
    project(*w_buf, *d_buf, PM, scalingFactors);
    reportTime("d_0 projection");
    writeProjections(*d_buf, "/tmp/cgls/d_0.den");
    dnorm2_old = normBBuffer_barier_double(*d_buf);
    while(norm / NB0 > 0.01)
    {
        // Iteration
        iteration = iteration + 1;
        alpha = vnorm2_old / dnorm2_old;
        LOGI << io::xprintf("After %d iteration |v|^2=%E, |d|^2=%E, alpha=%E", iteration - 1,
                            vnorm2_old, dnorm2_old, float(alpha));
        addIntoFirstVectorSecondVectorScaled(*x_buf, *w_buf, alpha, XDIM);
        reportTime(io::xprintf("x=x+alpha w %d iteration", iteration));
        // writeVolume(*x_buf, io::xprintf("/tmp/cgls/x_%d.den", iteration));
        addIntoFirstVectorSecondVectorScaled(*c_buf, *d_buf, -alpha, BDIM);
        reportTime(io::xprintf("c=c-alpha d %d iteration", iteration));
        // writeProjections(*c_buf, io::xprintf("/tmp/cgls/c_%d.den", iteration));
        backproject(*c_buf, *v_buf, PM, scalingFactors);
        reportTime(io::xprintf("v_%d backprojection", iteration));
        // writeVolume(*v_buf, io::xprintf("/tmp/cgls/v_%d.den", iteration));
        vnorm2_now = normXBuffer_barier_double(*v_buf);
        beta = vnorm2_now / vnorm2_old;
        LOGI << io::xprintf("In %d iteration |v_now|^2=%E, |v_old|^2=%E, beta=%0.2f", iteration,
                            vnorm2_now, vnorm2_old, beta);
        vnorm2_old = vnorm2_now;
        addIntoFirstVectorScaledSecondVector(*w_buf, *v_buf, beta, XDIM);
        // writeVolume(*w_buf, io::xprintf("/tmp/cgls/w_%d.den", iteration));
        project(*w_buf, *d_buf, PM, scalingFactors);
        reportTime(io::xprintf("d_%d projection", iteration));
        // writeProjections(*d_buf, io::xprintf("/tmp/cgls/d_%d.den", iteration));
        dnorm2_old = normBBuffer_barier_double(*d_buf);

        project(*x_buf, *tmp_b_buf, PM, scalingFactors);
        reportTime(io::xprintf("x_%d projection", iteration));
        addIntoFirstVectorSecondVectorScaled(*tmp_b_buf, *b_buf, -1.0, BDIM);
        norm = std::sqrt(normBBuffer_barier_double(*tmp_b_buf));
        LOGE << io::xprintf("After %d iteration, the norm of |Ax-b| is %f that is %0.2f%% of NB0.",
                            iteration, norm, 100.0 * norm / NB0);
    }
    // Optionally write even more converged solution
    alpha = vnorm2_old / dnorm2_old;
    LOGI << io::xprintf("Finally |v|^2=%E, |d|^2=%E, alpha=%E", iteration - 1, vnorm2_old,
                        dnorm2_old, float(alpha));
    addIntoFirstVectorSecondVectorScaled(*x_buf, *w_buf, alpha, XDIM);
    Q->enqueueReadBuffer(*x_buf, CL_TRUE, 0, sizeof(float) * XDIM, x);
    return 0;
}

} // namespace CTL
