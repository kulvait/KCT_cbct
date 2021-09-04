#include "VolumeConvolutionOperator.hpp"

namespace KCT {
cl::NDRange VolumeConvolutionOperator::guessProjectionLocalNDRange(bool barrierCalls)
{
    cl::NDRange projectorLocalNDRange;
    if(barrierCalls)
    {

        if(vdimx % 64 == 0 && vdimy % 4 == 0)
        {
            projectorLocalNDRange = cl::NDRange(64, 4, 1); // 9.45 Barrier
        } else
        {
            projectorLocalNDRange = cl::NDRange();
        }
    } else
    {
        if(vdimz % 4 == 0 && vdimy % 64 == 0)
        {
            projectorLocalNDRange = cl::NDRange(4, 64, 1); // 23.23 RELAXED
        } else
        {
            projectorLocalNDRange = cl::NDRange();
        }
    }
    return projectorLocalNDRange;
}

void VolumeConvolutionOperator::initializeConvolution()
{
    if(!isOpenCLInitialized())
    {
        CLINCLUDEutils();
        CLINCLUDEinclude();
        CLINCLUDEprojector();
        CLINCLUDEbackprojector();
        CLINCLUDEprojector_tt();
        CLINCLUDEbackprojector_tt();
        CLINCLUDEconvolution();
    } else
    {
        std::string err = "Could not initialize projector when OpenCL was already initialized.";
        LOGE << err;
        throw std::runtime_error(err.c_str());
    }
}

void VolumeConvolutionOperator::initializeAllAlgorithms()
{
    if(!isOpenCLInitialized())
    {
        CLINCLUDEutils();
        CLINCLUDEinclude();
        CLINCLUDEprojector();
        CLINCLUDEbackprojector();
        CLINCLUDEprojector_sidon();
        CLINCLUDEbackprojector_sidon();
        CLINCLUDEprojector_tt();
        CLINCLUDEbackprojector_tt();
        CLINCLUDEbackprojector_minmax();
        CLINCLUDEconvolution();
    } else
    {
        std::string err = "Could not initialize projector when OpenCL was already initialized.";
        LOGE << err;
        throw std::runtime_error(err.c_str());
    }
}
int VolumeConvolutionOperator::problemSetup(double voxelSizeX, double voxelSizeY, double voxelSizeZ)
{
    voxelSizes = cl_double3({ voxelSizeX, voxelSizeY, voxelSizeZ });
    return 0;
}

int VolumeConvolutionOperator::initializeOrUpdateVolumeBuffer(float* volumeArray)
{
    return initializeOrUpdateVolumeBuffer(vdimx, vdimy, vdimz, volumeArray);
}

int VolumeConvolutionOperator::initializeOrUpdateVolumeBuffer(uint32_t vdimx,
                                                              uint32_t vdimy,
                                                              uint32_t vdimz,
                                                              float* volumeArray)
{
    cl_int err = CL_SUCCESS;
    std::string msg;
    this->vdimx = vdimx;
    this->vdimy = vdimy;
    this->vdimz = vdimz;
    this->totalVoxelNum = vdimx * vdimy * vdimz;
    vdims = cl_int3({ int(vdimx), int(vdimy), int(vdimz) });
    if(volumeBuffer != nullptr)
    {
        if(this->totalVolumeBufferSize == sizeof(float) * this->totalVoxelNum)
        {
            if(volumeArray != nullptr)
            {
                err = Q[0]->enqueueWriteBuffer(*volumeBuffer, CL_TRUE, 0, totalVolumeBufferSize,
                                               (void*)volumeArray);
            }
        } else
        {
            this->totalVolumeBufferSize = sizeof(float) * this->totalVoxelNum;
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
        this->totalVolumeBufferSize = sizeof(float) * this->totalVoxelNum;
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

int VolumeConvolutionOperator::initializeOrUpdateOutputBuffer()
{
    cl_int err = CL_SUCCESS;
    std::string ERR;
    if(volumeBuffer == nullptr)
    {
        ERR = io::xprintf(
            "Volume buffer must be initialized before calling initializeOrUpdateOutputBuffer!");
        LOGE << ERR;
        throw std::runtime_error(ERR);
    }
    if(outputBuffer != nullptr)
    {
        if(this->totalOutputBufferSize != this->totalVolumeBufferSize)
        {
            this->totalOutputBufferSize = this->totalVolumeBufferSize;
            outputBuffer = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                        totalVolumeBufferSize, nullptr, &err);
        }
    } else
    {
        this->totalOutputBufferSize = this->totalVolumeBufferSize;
        outputBuffer = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                    totalVolumeBufferSize, nullptr, &err);
    }
    if(err != CL_SUCCESS)
    {
        ERR = io::xprintf("Unsucessful initialization of Volume with error code %d!", err);
        LOGE << ERR;
        throw std::runtime_error(ERR);
    }
    return 0;
}

int VolumeConvolutionOperator::initializeOrUpdateGradientOutputBuffers()
{
    cl_int err = CL_SUCCESS;
    std::string ERR;
    if(volumeBuffer == nullptr)
    {
        ERR = io::xprintf(
            "Volume buffer must be initialized before calling initializeOrUpdateOutputBuffer!");
        LOGE << ERR;
        throw std::runtime_error(ERR);
    }
    if(outputGradientX != nullptr)
    {
        if(this->totalOutputGradientBuffersSize != this->totalVolumeBufferSize)
        {
            this->totalOutputGradientBuffersSize = this->totalVolumeBufferSize;
            outputGradientX = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                           totalVolumeBufferSize, nullptr, &err);
            outputGradientY = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                           totalVolumeBufferSize, nullptr, &err);
            outputGradientZ = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                           totalVolumeBufferSize, nullptr, &err);
        }
    } else
    {
        this->totalOutputGradientBuffersSize = this->totalVolumeBufferSize;
        outputGradientX = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                       totalVolumeBufferSize, nullptr, &err);
        outputGradientY = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                       totalVolumeBufferSize, nullptr, &err);
        outputGradientZ = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                       totalVolumeBufferSize, nullptr, &err);
    }
    if(err != CL_SUCCESS)
    {
        ERR = io::xprintf("Unsucessful initialization of Volume with error code %d!", err);
        LOGE << ERR;
        throw std::runtime_error(ERR);
    }
    return 0;
}

int VolumeConvolutionOperator::fillVolumeBufferByConstant(float constant)
{
    std::string msg;
    if(volumeBuffer == nullptr)
    {
        msg = io::xprintf(
            "Volume buffer is not yet initialized, call initializeVolumeBuffer first!");
        LOGE << msg;
        throw std::runtime_error(msg);
    }
    return Q[0]->enqueueFillBuffer<cl_float>(*volumeBuffer, constant, 0, totalVolumeBufferSize);
}

int VolumeConvolutionOperator::convolve(std::string kernelName, float* outputVolume)
{
    cl::NDRange globalRange(vdimx, vdimy, vdimz);
    std::shared_ptr<cl::NDRange> localRange = std::make_shared<cl::NDRange>(projectorLocalNDRange);
    initializeOrUpdateOutputBuffer();
    cl_float16 convolutionKernel = { 0.25f, 0.5f, 0.25f, 0.5f, -3.0f, 0.5f, 0.25f, 0.5f,
                                     0.25f, 0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f };
    algFLOATvector_2Dconvolution3x3(*volumeBuffer, *outputBuffer, vdims, convolutionKernel,
                                    globalRange, localRange);
    cl_int err
        = Q[0]->enqueueReadBuffer(*outputBuffer, CL_TRUE, 0, totalVolumeBufferSize, outputVolume);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful writte buffer to the projection variable, code %d!", err);
        return -1;
    }
    return 0;
}

int VolumeConvolutionOperator::laplace3D(cl_float3 voxelSizes, float* outputVolume)
{
    cl::NDRange globalRange(vdimx, vdimy, vdimz);
    std::shared_ptr<cl::NDRange> localRange = std::make_shared<cl::NDRange>(projectorLocalNDRange);
    initializeOrUpdateOutputBuffer();
    algFLOATvector_3DconvolutionLaplaceZeroBoundary(*volumeBuffer, *outputBuffer, vdims, voxelSizes,
                                                    globalRange, localRange);
    cl_int err
        = Q[0]->enqueueReadBuffer(*outputBuffer, CL_TRUE, 0, totalVolumeBufferSize, outputVolume);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful writte buffer to the projection variable, code %d!", err);
        return -1;
    }
    return 0;
}

int VolumeConvolutionOperator::faridGradient3D(
    cl_float3 voxelSizes, float* outputX, float* outputY, float* outputZ, bool reflectionBoundary)
{
    cl::NDRange globalRange(vdimx, vdimy, vdimz);
    std::shared_ptr<cl::NDRange> localRange = std::make_shared<cl::NDRange>(projectorLocalNDRange);
    localRange = nullptr;
    initializeOrUpdateGradientOutputBuffers();
    algFLOATvector_3DconvolutionGradientFarid5x5x5(
        *volumeBuffer, *outputGradientX, *outputGradientY, *outputGradientZ, vdims, voxelSizes,
        (int)reflectionBoundary, globalRange, localRange);
    cl_int err
        = Q[0]->enqueueReadBuffer(*outputGradientX, CL_TRUE, 0, totalVolumeBufferSize, outputX);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful writte outputGradientX to the variable outputX, code %d!",
                            err);
        return -1;
    }
    err = Q[0]->enqueueReadBuffer(*outputGradientY, CL_TRUE, 0, totalVolumeBufferSize, outputY);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful writte outputGradientY to the variable outputY, code %d!",
                            err);
        return -1;
    }
    err = Q[0]->enqueueReadBuffer(*outputGradientZ, CL_TRUE, 0, totalVolumeBufferSize, outputZ);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful writte outputGradientZ to the variable outputZ, code %d!",
                            err);
        return -1;
    }
    return 0;
}

int VolumeConvolutionOperator::sobelGradient3D(cl_float3 voxelSizes,
                                               float* outputX,
                                               float* outputY,
                                               float* outputZ,
                                               bool reflectionBoundaryConditions)
{
    cl::NDRange globalRange(vdimx, vdimy, vdimz);
    std::shared_ptr<cl::NDRange> localRange = std::make_shared<cl::NDRange>(projectorLocalNDRange);
    localRange = nullptr;
    initializeOrUpdateGradientOutputBuffers();
    if(reflectionBoundaryConditions)
    {
        algFLOATvector_3DconvolutionGradientSobelFeldmanReflectionBoundary(
            *volumeBuffer, *outputGradientX, *outputGradientY, *outputGradientZ, vdims, voxelSizes,
            globalRange, localRange);
    } else
    {
        algFLOATvector_3DconvolutionGradientSobelFeldmanZeroBoundary(
            *volumeBuffer, *outputGradientX, *outputGradientY, *outputGradientZ, vdims, voxelSizes,
            globalRange, localRange);
    }
    cl_int err
        = Q[0]->enqueueReadBuffer(*outputGradientX, CL_TRUE, 0, totalVolumeBufferSize, outputX);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful writte outputGradientX to the variable outputX, code %d!",
                            err);
        return -1;
    }
    err = Q[0]->enqueueReadBuffer(*outputGradientY, CL_TRUE, 0, totalVolumeBufferSize, outputY);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful writte outputGradientY to the variable outputY, code %d!",
                            err);
        return -1;
    }
    err = Q[0]->enqueueReadBuffer(*outputGradientZ, CL_TRUE, 0, totalVolumeBufferSize, outputZ);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful writte outputGradientZ to the variable outputZ, code %d!",
                            err);
        return -1;
    }
    return 0;
}

int VolumeConvolutionOperator::isotropicGradient3D(cl_float3 voxelSizes,
                                                   float* outputX,
                                                   float* outputY,
                                                   float* outputZ)
{
    cl::NDRange globalRange(vdimx, vdimy, vdimz);
    std::shared_ptr<cl::NDRange> localRange = std::make_shared<cl::NDRange>(projectorLocalNDRange);
    localRange = nullptr;
    initializeOrUpdateGradientOutputBuffers();
    algFLOATvector_3DisotropicGradient(*volumeBuffer, *outputGradientX, *outputGradientY,
                                       *outputGradientZ, vdims, voxelSizes, globalRange,
                                       localRange);
    cl_int err
        = Q[0]->enqueueReadBuffer(*outputGradientX, CL_TRUE, 0, totalVolumeBufferSize, outputX);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful writte outputGradientX to the variable outputX, code %d!",
                            err);
        return -1;
    }
    err = Q[0]->enqueueReadBuffer(*outputGradientY, CL_TRUE, 0, totalVolumeBufferSize, outputY);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful writte outputGradientY to the variable outputY, code %d!",
                            err);
        return -1;
    }
    err = Q[0]->enqueueReadBuffer(*outputGradientZ, CL_TRUE, 0, totalVolumeBufferSize, outputZ);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful writte outputGradientZ to the variable outputZ, code %d!",
                            err);
        return -1;
    }
    return 0;
}

} // namespace KCT
