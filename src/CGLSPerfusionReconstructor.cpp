#include "CGLSPerfusionReconstructor.hpp"

namespace CTL {

int CGLSPerfusionReconstructor::initializeOpenCL(uint32_t platformId)
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

int CGLSPerfusionReconstructor::initializeData(std::vector<float*> projections,
                                               std::vector<float*> basisVectorValues,
                                               std::vector<float*> volumes)
{
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
        v_buf.push_back(std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                     sizeof(float) * XDIM, nullptr, &err));
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
            return -1;
        }
        w_buf.push_back(std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                     sizeof(float) * XDIM, nullptr, &err));

        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
            return -1;
        }
    }
    for(std::size_t i = 0; i != b.size(); i++)
    {
        b_buf.push_back(
            std::make_shared<cl::Buffer>(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         sizeof(float) * BDIM, (void*)projections[i], &err));
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
            return -1;
        }
        // Initialize buffer c by projections data
        c_buf.push_back(
            std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                         sizeof(float) * BDIM, (void*)projections[i], &err));
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
            return -1;
        }

        d_buf.push_back(std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
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
double CGLSPerfusionReconstructor::normXBuffer_frame_double(cl::Buffer& X)
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
double CGLSPerfusionReconstructor::normXBuffer_barier_double(cl::Buffer& X)
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
 * Funcion computes the norm of the Buffer that has vdimx * vdimy * vdimz elements.
 *
 * @param X
 *
 * @return
 */
double
CGLSPerfusionReconstructor::normXBuffer_barier_double(std::vector<std::shared_ptr<cl::Buffer>>& X)
{

    double sum = 0;
    for(std::size_t i = 0; i != X.size(); i++)
    {
        sum += normXBuffer_barier_double(*X[i]);
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
double CGLSPerfusionReconstructor::normBBuffer_frame_double(cl::Buffer& B)
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
double CGLSPerfusionReconstructor::normBBuffer_barier_double(cl::Buffer& B)
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
 * Funcion computes the norm of the Buffer that has pdimx * pdimy * pdimz elements.
 *
 * @param X
 *
 * @return
 */
double
CGLSPerfusionReconstructor::normBBuffer_barier_double(std::vector<std::shared_ptr<cl::Buffer>>& B)
{ // Use workGroupSize that is private constant default to 256
    double sum = 0;
    for(std::size_t i = 0; i != B.size(); i++)
    {
        sum += normBBuffer_barier_double(*B[i]);
    }
    return sum;
}

/**
 * Funcion computes the norm of the Buffer that has vdimx * vdimy * vdimz elements.
 *
 * @param X
 *
 * @return
 */
float CGLSPerfusionReconstructor::normXBuffer_frame(cl::Buffer& X)
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
float CGLSPerfusionReconstructor::normXBuffer_barier(cl::Buffer& X)
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
float CGLSPerfusionReconstructor::normBBuffer_frame(cl::Buffer& B)
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
float CGLSPerfusionReconstructor::normBBuffer_barier(cl::Buffer& B)
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

int CGLSPerfusionReconstructor::backproject(std::vector<std::shared_ptr<cl::Buffer>>& B,
                                            std::vector<std::shared_ptr<cl::Buffer>>& X,
                                            std::vector<matrix::ProjectionMatrix>& V,
                                            std::vector<float>& scalingFactors)
{
    for(std::size_t k = 0; k != X.size(); k++)
    {
        Q->enqueueFillBuffer<cl_float>(*X[k], FLOATZERO, 0, XDIM * sizeof(float));
    }
    unsigned int frameSize = pdimx * pdimy;
    for(std::size_t k = 0; k != X.size(); k++)
    {
        for(std::size_t i = 0; i != pdimz; i++)
        {
            for(std::size_t j = 0; j != B.size(); j++)
            {
                matrix::ProjectionMatrix mat = V[i];
                float scalingFactor = scalingFactors[i] * basisFunctionsValues[k][j * pdimz + i];
                std::array<double, 3> sourcePosition = mat.sourcePosition();
                std::array<double, 3> normalToDetector = mat.normalToDetector();
                double* P = mat.getPtr();
                cl_double16 PM({ P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10],
                                 P[11], 0.0, 0.0, 0.0, 0.0 });
                cl_double3 SOURCEPOSITION(
                    { sourcePosition[0], sourcePosition[1], sourcePosition[2] });
                cl_double3 NORMALTODETECTOR(
                    { normalToDetector[0], normalToDetector[1], normalToDetector[2] });
                cl_int4 vdims({ int(vdimx), int(vdimy), int(vdimz), 0 });
                cl_double3 voxelSizes({ 1.0, 1.0, 1.0 });
                cl_int2 pdims({ int(pdimx), int(pdimy) });
                cl::EnqueueArgs eargs(*Q, cl::NDRange(vdimz, vdimy, vdimx));
                unsigned int offset = i * frameSize;
                (*FLOATcutting_voxel_backproject)(eargs, *X[k], *B[j], offset, PM, SOURCEPOSITION,
                                                  NORMALTODETECTOR, vdims, voxelSizes, pdims,
                                                  scalingFactor)
                    .wait();
            }
        }
    }
    return 0;
}

int CGLSPerfusionReconstructor::project(std::vector<std::shared_ptr<cl::Buffer>>& X,
                                        std::vector<std::shared_ptr<cl::Buffer>>& B,
                                        std::vector<matrix::ProjectionMatrix>& V,
                                        std::vector<float>& scalingFactors)
{
    for(std::size_t j = 0; j != B.size(); j++)
    {
        Q->enqueueFillBuffer<cl_float>(*B[j], FLOATZERO, 0, BDIM * sizeof(float));
    }
    unsigned int frameSize = pdimx * pdimy;
    for(std::size_t k = 0; k != X.size(); k++)
    {
        for(std::size_t i = 0; i != pdimz; i++)
        {
            for(std::size_t j = 0; j != B.size(); j++)
            {
                matrix::ProjectionMatrix mat = V[i];
                float scalingFactor = scalingFactors[i] * basisFunctionsValues[k][j * pdimz + i];
                std::array<double, 3> sourcePosition = mat.sourcePosition();
                std::array<double, 3> normalToDetector = mat.normalToDetector();
                double* P = mat.getPtr();
                cl_double16 PM({ P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10],
                                 P[11], 0.0, 0.0, 0.0, 0.0 });
                cl_double3 SOURCEPOSITION(
                    { sourcePosition[0], sourcePosition[1], sourcePosition[2] });
                cl_double3 NORMALTODETECTOR(
                    { normalToDetector[0], normalToDetector[1], normalToDetector[2] });
                cl_int4 vdims({ int(vdimx), int(vdimy), int(vdimz), 0 });
                cl_double3 voxelSizes({ 1.0, 1.0, 1.0 });
                cl_int2 pdims({ int(pdimx), int(pdimy) });
                unsigned int offset = i * frameSize;
                cl::EnqueueArgs eargs(*Q, cl::NDRange(vdimz, vdimy, vdimx));
                (*FLOATcutting_voxel_project)(eargs, *X[k], *B[j], offset, PM, SOURCEPOSITION,
                                              NORMALTODETECTOR, vdims, voxelSizes, pdims,
                                              scalingFactor)
                    .wait();
            }
        }
    }
    return 0;
}

int CGLSPerfusionReconstructor::copyFloatVector(cl::Buffer& from, cl::Buffer& to, unsigned int size)
{
    cl::EnqueueArgs eargs(*Q, cl::NDRange(size));
    (*FLOAT_CopyVector)(eargs, from, to).wait();
    return 0;
}

int CGLSPerfusionReconstructor::copyFloatVector(std::vector<std::shared_ptr<cl::Buffer>>& from, std::vector<std::shared_ptr<cl::Buffer>>& to, unsigned int size)
{
	for(std::size_t i = 0; i != from.size(); i++)
	{
		copyFloatVector(*from[i], *to[i], size);
	} 
    return 0;
}

int CGLSPerfusionReconstructor::addIntoFirstVectorSecondVectorScaled(cl::Buffer& a,
                                                                     cl::Buffer& b,
                                                                     float f,
                                                                     unsigned int size)
{
    cl::EnqueueArgs eargs(*Q, cl::NDRange(size));
    (*FLOAT_addIntoFirstVectorSecondVectorScaled)(eargs, a, b, f).wait();
    return 0;
}

int CGLSPerfusionReconstructor::addIntoFirstVectorSecondVectorScaled(
    std::vector<std::shared_ptr<cl::Buffer>>& a,
    std::vector<std::shared_ptr<cl::Buffer>>& b,
    float f,
    unsigned int size)
{
    for(std::size_t i = 0; i != a.size(); i++)
    {
        addIntoFirstVectorSecondVectorScaled(*a[i], *b[i], f, size);
    }
    return 0;
}

int CGLSPerfusionReconstructor::addIntoFirstVectorScaledSecondVector(cl::Buffer& a,
                                                                     cl::Buffer& b,
                                                                     float f,
                                                                     unsigned int size)
{
    cl::EnqueueArgs eargs(*Q, cl::NDRange(size));
    (*FLOAT_addIntoFirstVectorScaledSecondVector)(eargs, a, b, f).wait();
    return 0;
}

int CGLSPerfusionReconstructor::addIntoFirstVectorScaledSecondVector(
    std::vector<std::shared_ptr<cl::Buffer>>& a,
    std::vector<std::shared_ptr<cl::Buffer>>& b,
    float f,
    unsigned int size)
{
    for(std::size_t i = 0; i != a.size(); i++)
    {
        addIntoFirstVectorScaledSecondVector(*a[i], *b[i], f, size);
    }
    return 0;
}

std::vector<matrix::ProjectionMatrix> CGLSPerfusionReconstructor::encodeProjectionMatrices(
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

std::vector<float>
CGLSPerfusionReconstructor::computeScalingFactors(std::vector<matrix::ProjectionMatrix> PM)
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

void CGLSPerfusionReconstructor::writeVolume(std::vector<std::shared_ptr<cl::Buffer>>& X,
                                             std::string path)
{
    uint16_t buf[3];
    buf[0] = vdimy;
    buf[1] = vdimx;
    buf[2] = vdimz;
    for(std::size_t i = 0; i != x.size(); i++)
    {
        std::string newpath = io::xprintf("%s_elm%d", path.c_str(), i);
        io::createEmptyFile(newpath, 0, true); // Try if this is faster
        io::appendBytes(newpath, (uint8_t*)buf, 6);
        Q->enqueueReadBuffer(*X[i], CL_TRUE, 0, sizeof(float) * XDIM, x[i]);
        io::appendBytes(newpath, (uint8_t*)x[i], XDIM * sizeof(float));
    }
}

void CGLSPerfusionReconstructor::writeProjections(std::vector<std::shared_ptr<cl::Buffer>>& B,
                                                  std::string path)
{
    uint16_t buf[3];
    buf[0] = pdimy;
    buf[1] = pdimx;
    buf[2] = pdimz;
    for(std::size_t i = 0; i != b.size(); i++)
    {
        std::string newpath = io::xprintf("%s_proj%d", path.c_str(), i);
        io::createEmptyFile(newpath, 0, true); // Try if this is faster
        io::appendBytes(newpath, (uint8_t*)buf, 6);
        Q->enqueueReadBuffer(*B[i], CL_TRUE, 0, sizeof(float) * BDIM, b[i]);
        io::appendBytes(newpath, (uint8_t*)b[i], BDIM * sizeof(float));
    }
}

void CGLSPerfusionReconstructor::setTimepoint() { timepoint = std::chrono::steady_clock::now(); }

void CGLSPerfusionReconstructor::reportTime(std::string msg)
{
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - timepoint);
    LOGW << io::xprintf("%s: %ds", msg.c_str(), duration.count());
    setTimepoint();
}

int CGLSPerfusionReconstructor::reconstruct(std::shared_ptr<io::DenProjectionMatrixReader> matrices,
                                            uint32_t maxIterations,
                                            float errCondition)
{
    LOGI << io::xprintf("WELCOME TO CGLS");
    reportTime("CGLS INIT");
    if(reportProgress)
    {
        writeProjections(b_buf, io::xprintf("%sb.den", progressBeginPath.c_str()));
        writeVolume(x_buf, io::xprintf("%sx_0.den", progressBeginPath.c_str()));
    }
    std::vector<matrix::ProjectionMatrix> PM = encodeProjectionMatrices(matrices);
    std::vector<float> scalingFactors = computeScalingFactors(PM);
    uint32_t iteration = 0;
    double norm, vnorm2_old, vnorm2_now, dnorm2_old, alpha, beta;
    double NB0 = std::sqrt(normBBuffer_barier_double(b_buf));
    norm = NB0;
    project(x_buf, tmp_b_buf, PM, scalingFactors);
    reportTime("X_0 projection");
    addIntoFirstVectorSecondVectorScaled(tmp_b_buf, b_buf, -1.0, BDIM);
    norm = std::sqrt(normBBuffer_barier_double(tmp_b_buf));
    LOGI << io::xprintf("Initial norm of b is %f and initial |Ax-b| is %f.", NB0, norm);
    // INITIALIZATION x_0 is initialized typically by zeros but in general by supplied array
    // c_0 is filled by b
    // v_0=w_0=BACKPROJECT(c_0)
    // writeProjections(*c_buf, io::xprintf("/tmp/cgls/c_0.den"));
    backproject(c_buf, v_buf, PM, scalingFactors);
    reportTime("v_0 backprojection");
    if(reportProgress)
    {
       // writeVolume(v_buf, io::xprintf("%sv_0.den", progressBeginPath.c_str()));
    }
    vnorm2_old = normXBuffer_barier_double(v_buf);
    copyFloatVector(v_buf, w_buf, XDIM);
    if(reportProgress)
    {
       // writeVolume(w_buf, io::xprintf("%sw_0.den", progressBeginPath.c_str()));
    }
    project(w_buf, d_buf, PM, scalingFactors);
    reportTime("d_0 projection");
    if(reportProgress)
    {
       // writeProjections(d_buf, io::xprintf("%sd_0.den", progressBeginPath.c_str()));
    }
    dnorm2_old = normBBuffer_barier_double(d_buf);
    while(norm / NB0 > errCondition && iteration < maxIterations)
    {
        // Iteration
        iteration = iteration + 1;
        alpha = vnorm2_old / dnorm2_old;
        LOGI << io::xprintf("After iteration %d, |v|^2=%E, |d|^2=%E, alpha=%E", iteration - 1,
                            vnorm2_old, dnorm2_old, float(alpha));
        addIntoFirstVectorSecondVectorScaled(x_buf, w_buf, alpha, XDIM);
        if(reportProgress)
        {
            writeVolume(x_buf, io::xprintf("%sx_%d.den", progressBeginPath.c_str(), iteration));
        }
        addIntoFirstVectorSecondVectorScaled(c_buf, d_buf, -alpha, BDIM);
        if(reportProgress)
        {
        //    writeProjections(c_buf,
        //                     io::xprintf("%sc_%d.den", progressBeginPath.c_str(), iteration));
        }
        backproject(c_buf, v_buf, PM, scalingFactors);
        reportTime(io::xprintf("v_%d backprojection", iteration));

        if(reportProgress)
        {
      //      writeVolume(v_buf, io::xprintf("%sv_%d.den", progressBeginPath.c_str(), iteration));
        }
        vnorm2_now = normXBuffer_barier_double(v_buf);
        beta = vnorm2_now / vnorm2_old;
        LOGI << io::xprintf("In iteration %d, |v_now|^2=%E, |v_old|^2=%E, beta=%0.2f", iteration,
                            vnorm2_now, vnorm2_old, beta);
        vnorm2_old = vnorm2_now;
        addIntoFirstVectorScaledSecondVector(w_buf, v_buf, beta, XDIM);
        if(reportProgress)
        {
    //        writeVolume(w_buf, io::xprintf("%sw_%d.den", progressBeginPath.c_str(), iteration));
        }
        addIntoFirstVectorSecondVectorScaled(tmp_b_buf, d_buf, alpha, BDIM);
        project(w_buf, d_buf, PM, scalingFactors);
        reportTime(io::xprintf("d_%d projection", iteration));
        if(reportProgress)
        {
      //      writeProjections(d_buf,
      //                       io::xprintf("%sd_%d.den", progressBeginPath.c_str(), iteration));
        }
        dnorm2_old = normBBuffer_barier_double(d_buf);

        norm = std::sqrt(normBBuffer_barier_double(tmp_b_buf));
        LOGE << io::xprintf("After iteration %d, the norm of |Ax-b| is %f that is %0.2f%% of NB0.",
                            iteration, norm, 100.0 * norm / NB0);
    }
    // Optionally write even more converged solution
    alpha = vnorm2_old / dnorm2_old;
    LOGI << io::xprintf("Finally |v|^2=%E, |d|^2=%E, alpha=%E", iteration - 1, vnorm2_old,
                        dnorm2_old, float(alpha));
    addIntoFirstVectorSecondVectorScaled(x_buf, w_buf, alpha, XDIM);
    for(std::size_t i = 0; i != x.size(); i++)
    {
        Q->enqueueReadBuffer(*x_buf[i], CL_TRUE, 0, sizeof(float) * XDIM, x[i]);
    }
    return 0;
}

} // namespace CTL
