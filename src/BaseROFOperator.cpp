#include "BaseROFOperator.hpp"

// Namespace for your PDHG operator implementation
namespace KCT {

// OpenCL initialization
int BaseROFOperator::initializeOpenCL(uint32_t platformID,
                                      uint32_t* deviceIds,
                                      uint32_t deviceIdsLength,
                                      std::string xpath,
                                      bool debug,
                                      bool relaxed)
{
    int val
        = Kniha::initializeOpenCL(platformID, deviceIds, deviceIdsLength, xpath, debug, relaxed);
    if(val == 0)
    {
        return 0;
    } else
    {
        std::string ERR = io::xprintf("Wrong initialization of OpenCL with code %d!", val);
        KCTERR(ERR);
    }
}

// Initializes the volume and sets up the required buffers
int BaseROFOperator::initializeVolume(float* volume)
{
    this->x = volume;
    cl_int err;
    // Initialize volume buffer
    x_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                         sizeof(float) * XDIM, (void*)volume, &err);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
        return -1;
    }
    return 0;
}

// Sets up the problem with voxel dimensions or other problem-specific parameters
int BaseROFOperator::problemSetup(double voxelSpacingX, double voxelSpacingY, double voxelSpacingZ)
{
    this->voxelSpacingX = voxelSpacingX;
    this->voxelSpacingY = voxelSpacingY;
    this->voxelSpacingZ = voxelSpacingZ;
    voxelSizes = cl_double3({ voxelSpacingX, voxelSpacingY, voxelSpacingZ });
    voxelSizesF = cl_float3({ static_cast<float>(voxelSpacingX), static_cast<float>(voxelSpacingY),
                              static_cast<float>(voxelSpacingZ) });
    initReductionBuffers();
    return 0;
}

int BaseROFOperator::allocateXBuffers(uint32_t xBufferCount)
{
    cl_int err;
    std::shared_ptr<cl::Buffer> xbufptr;
    while(this->x_buffers.size() < xBufferCount)
    {
        xbufptr = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * XDIM,
                                               nullptr, &err);
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of X buffer with error code %d!", err);
            return -1;
        } else
        {
            x_buffers.push_back(xbufptr);
        }
    }
    return 0;
}

std::shared_ptr<cl::Buffer> BaseROFOperator::getXBuffer(uint32_t i)
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

void BaseROFOperator::setReportingParameters(bool verbose,
                                             uint32_t reportKthIteration,
                                             std::string intermediatePrefix)
{
    this->verbose = verbose;
    this->reportKthIteration = reportKthIteration;
    this->intermediatePrefix = intermediatePrefix;
}

void BaseROFOperator::initializeVolumeConvolution()
{
    if(!isOpenCLInitialized())
    {
        CLINCLUDEconvolution();
    } else
    {
        std::string err
            = "Could not initialize volume convolution when OpenCL was already initialized.";
        KCTERR(err);
    }
}

void BaseROFOperator::initializeProximal()
{
    if(!isOpenCLInitialized())
    {
        CLINCLUDEproximal();
    } else
    {
        std::string err
            = "Could not initialize volume convolution when OpenCL was already initialized.";
        KCTERR(err);
    }
}

void BaseROFOperator::initializeGradient()
{
    if(!isOpenCLInitialized())
    {
        CLINCLUDEgradient();
        gradientInitialized = true;
    } else
    {
        std::string err
            = "Could not initialize volume convolution when OpenCL was already initialized.";
        KCTERR(err);
    }
}

void BaseROFOperator::setGradientType(GradientType type)
{
    useGradientType = type;
    std::string gradientTypeString = GradientTypeToString(useGradientType);
    LOGD << io::xprintf("Setting gradient computation method to %s", gradientTypeString.c_str());
}

int BaseROFOperator::volume_gradient2D(cl::Buffer& F, cl::Buffer& GX, cl::Buffer& GY)
{
    if(!gradientInitialized)
    {
        KCTERR("Gradient not initialized, call initializeGradient() before initializing OpenCL.");
    }
    cl::NDRange globalRangeGradient(vdims.x, vdims.y, vdims.z);
    cl::NDRange localRangeGradient = cl::NullRange;

    switch(useGradientType)
    {
    case GradientType::CentralDifference3Point:
        algFLOATvector_Gradient2D_centralDifference_3point(F, GX, GY, vdims, voxelSizesF,
                                                           globalRangeGradient, localRangeGradient);
        break;

    case GradientType::CentralDifference5Point:
        algFLOATvector_Gradient2D_centralDifference_5point(F, GX, GY, vdims, voxelSizesF,
                                                           globalRangeGradient, localRangeGradient);
        break;

    case GradientType::ForwardDifference2Point:
        algFLOATvector_Gradient2D_forwardDifference_2point(F, GX, GY, vdims, voxelSizesF,
                                                           globalRangeGradient, localRangeGradient);
        break;

    case GradientType::ForwardDifference3Point:
        algFLOATvector_Gradient2D_forwardDifference_3point(F, GX, GY, vdims, voxelSizesF,
                                                           globalRangeGradient, localRangeGradient);
        break;
    case GradientType::ForwardDifference4Point:
        algFLOATvector_Gradient2D_forwardDifference_4point(F, GX, GY, vdims, voxelSizesF,
                                                           globalRangeGradient, localRangeGradient);
        break;
    case GradientType::ForwardDifference5Point:
        algFLOATvector_Gradient2D_forwardDifference_5point(F, GX, GY, vdims, voxelSizesF,
                                                           globalRangeGradient, localRangeGradient);
        break;
    case GradientType::ForwardDifference6Point:
        algFLOATvector_Gradient2D_forwardDifference_6point(F, GX, GY, vdims, voxelSizesF,
                                                           globalRangeGradient, localRangeGradient);
        break;
    case GradientType::ForwardDifference7Point:
        algFLOATvector_Gradient2D_forwardDifference_7point(F, GX, GY, vdims, voxelSizesF,
                                                           globalRangeGradient, localRangeGradient);
        break;

    default:
        KCTERR("Unknown GradientType");
    }
    return 0;
}

int BaseROFOperator::volume_gradient2D_adjoint(cl::Buffer& GX, cl::Buffer& GY, cl::Buffer& D)
{
    if(!gradientInitialized)
    {
        KCTERR("Gradient not initialized, call initializeGradient() before initializing OpenCL.");
    }
    cl::NDRange globalRangeGradient(vdims.x, vdims.y, vdims.z);
    cl::NDRange localRangeGradient = cl::NullRange;

    switch(useGradientType)
    {
    case GradientType::CentralDifference3Point:
        algFLOATvector_Gradient2D_centralDifference_3point_adjoint(
            GX, GY, D, vdims, voxelSizesF, globalRangeGradient, localRangeGradient);
        break;

    case GradientType::CentralDifference5Point:
        algFLOATvector_Gradient2D_centralDifference_5point_adjoint(
            GX, GY, D, vdims, voxelSizesF, globalRangeGradient, localRangeGradient);
        break;

    case GradientType::ForwardDifference2Point:
        algFLOATvector_Gradient2D_forwardDifference_2point_adjoint(
            GX, GY, D, vdims, voxelSizesF, globalRangeGradient, localRangeGradient);
        break;

    case GradientType::ForwardDifference3Point:
        algFLOATvector_Gradient2D_forwardDifference_3point_adjoint(
            GX, GY, D, vdims, voxelSizesF, globalRangeGradient, localRangeGradient);
        break;
    case GradientType::ForwardDifference4Point:
        algFLOATvector_Gradient2D_forwardDifference_4point_adjoint(
            GX, GY, D, vdims, voxelSizesF, globalRangeGradient, localRangeGradient);
        break;
    case GradientType::ForwardDifference5Point:
        algFLOATvector_Gradient2D_forwardDifference_5point_adjoint(
            GX, GY, D, vdims, voxelSizesF, globalRangeGradient, localRangeGradient);
        break;
    case GradientType::ForwardDifference6Point:
        algFLOATvector_Gradient2D_forwardDifference_6point_adjoint(
            GX, GY, D, vdims, voxelSizesF, globalRangeGradient, localRangeGradient);
        break;
    case GradientType::ForwardDifference7Point:
        algFLOATvector_Gradient2D_forwardDifference_7point_adjoint(
            GX, GY, D, vdims, voxelSizesF, globalRangeGradient, localRangeGradient);
        break;

    default:
        KCTERR("Unknown GradientType");
    }
    return 0;
}

void BaseROFOperator::writeVolume(cl::Buffer& X, const std::string& path)
{
    bufferIntoArray(X, x, XDIM);
    bool arrayxmajor = true;
    bool outxmajor = true;
    io::DenFileInfo::create3DDenFileFromArray(x, arrayxmajor, path, io::DenSupportedType::FLOAT32,
                                              vdimx, vdimy, vdimz, outxmajor);
}

std::chrono::milliseconds BaseROFOperator::millisecondsFromTimestamp(bool setNewTimestamp)
{
    std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - timestamp);
    if(setNewTimestamp)
    {
        setTimestamp(false);
    }
    return ms;
}

void BaseROFOperator::setTimestamp(bool finishCommandQueue)
{
    if(finishCommandQueue)
    {
        Q[0]->finish();
    }
    timestamp = std::chrono::steady_clock::now();
}

std::string
BaseROFOperator::printTime(std::string msg, bool finishCommandQueue, bool setNewTimestamp)
{
    if(finishCommandQueue)
    {
        Q[0]->finish();
    }
    auto duration = millisecondsFromTimestamp(setNewTimestamp);
    return io::xprintf("%s: %0.2fs", msg.c_str(), duration.count() / 1000.0);
}

void BaseROFOperator::reportTime(std::string msg, bool finishCommandQueue, bool setNewTimestamp)
{
    if(finishCommandQueue)
    {
        Q[0]->finish();
    }
    auto duration = millisecondsFromTimestamp(setNewTimestamp);
    if(verbose)
    {
        LOGD << io::xprintf("%s: %0.2fs", msg.c_str(), duration.count() / 1000.0);
    }
}

void BaseROFOperator::setVerbose(bool verbose, std::string intermediatePrefix)
{
    this->verbose = verbose;
    this->intermediatePrefix = intermediatePrefix;
}

} // namespace KCT
