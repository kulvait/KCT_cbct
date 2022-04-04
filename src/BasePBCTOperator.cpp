#include "BasePBCTOperator.hpp"

namespace KCT {

void BasePBCTOperator::initializeCVPProjector(bool useBarrierCalls, uint32_t LOCALARRAYSIZE)
{
    if(!isOpenCLInitialized())
    {
        this->useCVPProjector = true;
        this->useBarrierImplementation = useBarrierCalls;
        this->useSidonProjector = false;
        this->useTTProjector = false;
        CLINCLUDEutils();
        CLINCLUDEinclude();
        if(useBarrierCalls)
        {
            addOptString(io::xprintf("-DLOCALARRAYSIZE=%d", LOCALARRAYSIZE));
            this->LOCALARRAYSIZE = LOCALARRAYSIZE;
            KCTERR("No barrier implementation of PBCVP yet.");
        } else
        {
            CLINCLUDEpbct_cvp();
        }
        CLINCLUDEbackprojector();
        CLINCLUDErescaleProjections();
    } else
    {
        KCTERR("Could not initialize projector when OpenCL was already initialized.");
    }
}

void BasePBCTOperator::initializeSidonProjector(uint32_t probesPerEdgeX, uint32_t probesPerEdgeY)
{
    if(!isOpenCLInitialized())
    {
        this->useSidonProjector = true;
        this->pixelGranularity = { probesPerEdgeX, probesPerEdgeY };
        this->useCVPProjector = false;
        this->useTTProjector = false;
        CLINCLUDEutils();
        CLINCLUDEinclude();
        CLINCLUDEprojector_sidon();
        CLINCLUDEbackprojector_sidon();
    } else
    {
        KCTERR("Could not initialize projector when OpenCL was already initialized.");
    }
}

void BasePBCTOperator::initializeTTProjector()
{
    if(!isOpenCLInitialized())
    {
        this->useTTProjector = true;
        this->useCVPProjector = false;
        this->useSidonProjector = false;
        CLINCLUDEutils();
        CLINCLUDEinclude();
        CLINCLUDEprojector();
        CLINCLUDEbackprojector();
        CLINCLUDEprojector_tt();
        CLINCLUDEbackprojector_tt();
    } else
    {
        KCTERR("Could not initialize projector when OpenCL was already initialized.");
    }
}

void BasePBCTOperator::initializeVolumeConvolution()
{
    if(!isOpenCLInitialized())
    {
        CLINCLUDEconvolution();
    } else
    {
        std::string err
            = "Could not initialize volume convolution when OpenCL was already initialized.";
        LOGE << err;
        throw std::runtime_error(err.c_str());
    }
}

void BasePBCTOperator::useJacobiVectorCLCode()
{
    if(!isOpenCLInitialized())
    {
        CLINCLUDEprecomputeJacobiPreconditioner();
    } else
    {
        std::string err = "Could not initialize projector when OpenCL was already initialized.";
        LOGE << err;
        throw std::runtime_error(err.c_str());
    }
}

int BasePBCTOperator::problemSetup(
    std::vector<std::shared_ptr<geometry::Geometry3DParallelI>> geometries,
    double voxelSpacingX,
    double voxelSpacingY,
    double voxelSpacingZ,
    double volumeCenterX,
    double volumeCenterY,
    double volumeCenterZ)
{
    if(geometries.size() != pdimz)
    {
        std::string ERR
            = io::xprintf("The pdimz=%d but the size of camera geometries vector is %d!", pdimz,
                          geometries.size());
        KCTERR(ERR);
    }
    this->geometries = geometries;
    PM8Vector.clear();
    scalingFactorVector.clear();
    cl_double8 CM;
    float pixelArea, detectorTilt, scalingFactor;
    for(uint32_t k = 0; k != geometries.size(); k++)
    {
        std::shared_ptr<geometry::Geometry3DParallelI> G = geometries[k];
        pixelArea = G->pixelArea();
        detectorTilt = G->detectorTilt();
        scalingFactor = 1.0f / (pixelArea * detectorTilt);
        G->projectionMatrixAsVector8((double*)&CM);
        PM8Vector.emplace_back(CM);
        scalingFactorVector.emplace_back(scalingFactor);
    }
    voxelSizes = cl_double3({ voxelSpacingX, voxelSpacingY, voxelSpacingZ });
    volumeCenter = cl_double3({ volumeCenterX, volumeCenterY, volumeCenterZ });
    initializeAlgorithmsBuffers();
    return 0;
}

/**
 * @brief
 *
 * @param projections The b vector to invert.
 * @param volume Allocated memory to store x. Might contain the initial guess.
 *
 * @return
 */
int BasePBCTOperator::initializeVectors(float* projections,
                                        float* volume,
                                        bool useVolumeAsInitialX0)
{
    this->useVolumeAsInitialX0 = useVolumeAsInitialX0;
    this->b = projections;
    this->x = volume;
    cl_int err;

    // Initialize buffers
    if(useVolumeAsInitialX0)
    {
        x_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                             sizeof(float) * XDIM, (void*)volume, &err);
    } else
    {
        x_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * XDIM,
                                             nullptr, &err);
    }
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

    tmp_b_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * BDIM,
                                             nullptr, &err);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
        return -1;
    }

    return 0;
}

int BasePBCTOperator::allocateXBuffers(uint32_t xBufferCount)
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
        }
        x_buffers.push_back(xbufptr);
    }
    return 0;
}

int BasePBCTOperator::allocateBBuffers(uint32_t bBufferCount)
{
    cl_int err;
    std::shared_ptr<cl::Buffer> bbufptr;
    while(this->b_buffers.size() < bBufferCount)
    {
        bbufptr = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * BDIM,
                                               nullptr, &err);
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of B buffer with error code %d!", err);
            return -1;
        }
        b_buffers.push_back(bbufptr);
    }
    return 0;
}

int BasePBCTOperator::allocateTmpXBuffers(uint32_t xBufferCount)
{
    cl_int err;
    std::shared_ptr<cl::Buffer> xbufptr;
    while(this->tmp_x_buffers.size() < xBufferCount)
    {
        xbufptr = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * XDIM,
                                               nullptr, &err);
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of X buffer with error code %d!", err);
            return -1;
        }
        tmp_x_buffers.push_back(xbufptr);
    }
    return 0;
}

int BasePBCTOperator::allocateTmpBBuffers(uint32_t bBufferCount)
{
    cl_int err;
    std::shared_ptr<cl::Buffer> bbufptr;
    while(this->tmp_b_buffers.size() < bBufferCount)
    {
        bbufptr = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * BDIM,
                                               nullptr, &err);
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of B buffer with error code %d!", err);
            return -1;
        }
        tmp_b_buffers.push_back(bbufptr);
    }
    return 0;
}

std::shared_ptr<cl::Buffer> BasePBCTOperator::getXBuffer(uint32_t i)
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

std::shared_ptr<cl::Buffer> BasePBCTOperator::getBBuffer(uint32_t i)
{
    if(i < b_buffers.size())
    {
        return b_buffers[i];
    } else
    {
        std::string err = io::xprintf(
            "Index %d is out of range of the b_buffers vector of size %d!", i, b_buffers.size());
        LOGE << err;
        throw std::runtime_error(err);
    }
}

std::shared_ptr<cl::Buffer> BasePBCTOperator::getTmpXBuffer(uint32_t i)
{
    if(i < tmp_x_buffers.size())
    {
        return tmp_x_buffers[i];
    } else
    {
        std::string err
            = io::xprintf("Index %d is out of range of the tmp_x_buffers vector of size %d!", i,
                          tmp_x_buffers.size());
        LOGE << err;
        throw std::runtime_error(err);
    }
}

std::shared_ptr<cl::Buffer> BasePBCTOperator::getTmpBBuffer(uint32_t i)
{
    if(i < tmp_b_buffers.size())
    {
        return tmp_b_buffers[i];
    } else
    {
        std::string err
            = io::xprintf("Index %d is out of range of the tmp_b_buffers vector of size %d!", i,
                          tmp_b_buffers.size());
        LOGE << err;
        throw std::runtime_error(err);
    }
}

/**
 * @param initialProjectionIndex For OS SART 0 by default
 * @param projectionIncrement For OS SART 1 by default
 *
 */
int BasePBCTOperator::backproject(cl::Buffer& B,
                                  cl::Buffer& X,
                                  uint32_t initialProjectionIndex,
                                  uint32_t projectionIncrement)
{
    Q[0]->enqueueFillBuffer<cl_float>(X, FLOATZERO, 0, XDIM * sizeof(float));
    cl::NDRange globalRange(vdimx, vdimy, vdimz);
    std::shared_ptr<cl::NDRange> localRange
        = std::make_shared<cl::NDRange>(backprojectorLocalNDRange);
    cl_double8 CM;
    float scalingFactor;
    unsigned long frameSize = pdimx * pdimy;
    unsigned long offset;
    for(std::size_t i = initialProjectionIndex; i < pdimz; i += projectionIncrement)
    {
        CM = PM8Vector[i];
        scalingFactor = scalingFactorVector[i];
        offset = i * frameSize;
        if(useSidonProjector)
        {
            KCTERR("Siddon operators are not yet implemented for PBCT.");
        } else if(useTTProjector)
        {
            KCTERR("Footprint operators are not yet implemented for PBCT.");
        } else
        {
            algFLOAT_pbct_cutting_voxel_backproject(X, B, offset, CM, vdims, voxelSizes,
                                                    volumeCenter, pdims, scalingFactor, globalRange,
                                                    localRange);
        }
    }
    return 0;
}

/**
 * @param initialProjectionIndex For OS SART, 0 by default
 * @param projectionIncrement For OS SART, 1 by default
 *
 */
int BasePBCTOperator::project(cl::Buffer& X,
                              cl::Buffer& B,
                              uint32_t initialProjectionIndex,
                              uint32_t projectionIncrement)
{
    Q[0]->enqueueFillBuffer<cl_float>(B, FLOATZERO, 0, BDIM * sizeof(float));
    cl::NDRange globalRange(vdimz, vdimy, vdimx);
    std::shared_ptr<cl::NDRange> localRange = std::make_shared<cl::NDRange>(projectorLocalNDRange);
    // clang-format off
    // cl::NDRange barrierGlobalRange = cl::NDRange(vdimx, vdimy, vdimz);
    // std::shared_ptr<cl::NDRange> barrierLocalRange
    //    = std::make_shared<cl::NDRange>(projectorLocalNDRangeBarrier);
    // clang-format on
    cl_double8 CM;
    float scalingFactor;
    unsigned long frameSize = pdimx * pdimy;
    unsigned long offset;
    for(std::size_t i = initialProjectionIndex; i < pdimz; i += projectionIncrement)
    {
        CM = PM8Vector[i];
        scalingFactor = scalingFactorVector[i];
        offset = i * frameSize;
        if(useSidonProjector)
        {
            KCTERR("Siddon operators are not yet implemented for PBCT.");
        } else if(useTTProjector)
        {
            KCTERR("Footprint operators are not yet implemented for PBCT.");
        } else
        {
            if(useBarrierImplementation)
            {
                KCTERR("Barrier implementation of CVP projector is not yet implemented for PBCT.");
            } else
            {
                algFLOAT_pbct_cutting_voxel_project(X, B, offset, CM, vdims, voxelSizes,
                                                    volumeCenter, pdims, scalingFactor, globalRange,
                                                    localRange);
            }
        }
    }
    return 0;
}

std::vector<std::shared_ptr<CameraI>>
BasePBCTOperator::encodeProjectionMatrices(std::shared_ptr<io::DenProjectionMatrixReader> pm)
{
    std::vector<std::shared_ptr<CameraI>> v;
    std::shared_ptr<CameraI> P;
    for(std::size_t i = 0; i != pm->count(); i++)
    {
        P = std::make_shared<LightProjectionMatrix>(pm->readMatrix(i));
        v.emplace_back(P);
    }
    return v;
}

// Scaling factor is a expression f*f/(px*py), where f is source to detector distance and pixel
// sizes are (px and py)  Focal length http://ksimek.github.io/2013/08/13/intrinsic/
std::vector<float> BasePBCTOperator::computeScalingFactors()
{
    std::vector<float> scalingFactors;
    float pixelArea, detectorTilt, scalingFactor;
    for(uint32_t k = 0; k != pdimz; k++)
    {
        std::shared_ptr<geometry::Geometry3DParallelI> G = geometries[k];
        pixelArea = G->pixelArea();
        detectorTilt = G->detectorTilt();
        scalingFactor = 1.0f / (pixelArea * detectorTilt);
        scalingFactors.emplace_back(scalingFactor);
    }
    return scalingFactors;
}

void BasePBCTOperator::writeVolume(cl::Buffer& X, std::string path)
{
    uint16_t buf[3];
    buf[0] = vdimy;
    buf[1] = vdimx;
    buf[2] = vdimz;
    io::createEmptyFile(path, 0, true); // Try if this is faster
    io::appendBytes(path, (uint8_t*)buf, 6);
    Q[0]->enqueueReadBuffer(X, CL_TRUE, 0, sizeof(float) * XDIM, x);
    io::appendBytes(path, (uint8_t*)x, XDIM * sizeof(float));
}

void BasePBCTOperator::writeProjections(cl::Buffer& B, std::string path)
{
    Q[0]->enqueueReadBuffer(B, CL_TRUE, 0, sizeof(float) * BDIM, b);
    io::DenAsyncFrame2DWritter<float> projectionWritter(path, pdimx, pdimy, pdimz);
    uint64_t frameSize = pdimx * pdimy;
    for(uint32_t k = 0; k != pdimz; k++)
    {
        io::BufferedFrame2D<float> transposedFrame(b + k * frameSize, pdimy, pdimx);
        std::shared_ptr<io::Frame2DI<float>> frame = transposedFrame.transposed();
        projectionWritter.writeFrame(*frame, k);
    }
}

void BasePBCTOperator::setTimestamp(bool finishCommandQueue)
{
    if(finishCommandQueue)
    {
        Q[0]->finish();
    }
    timestamp = std::chrono::steady_clock::now();
}
std::chrono::milliseconds BasePBCTOperator::millisecondsFromTimestamp(bool setNewTimestamp)
{
    std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - timestamp);
    if(setNewTimestamp)
    {
        setTimestamp(false);
    }
    return ms;
}

std::string
BasePBCTOperator::printTime(std::string msg, bool finishCommandQueue, bool setNewTimestamp)
{
    if(finishCommandQueue)
    {
        Q[0]->finish();
    }
    auto duration = millisecondsFromTimestamp(setNewTimestamp);
    return io::xprintf("%s: %0.2fs", msg.c_str(), duration.count() / 1000.0);
}

void BasePBCTOperator::reportTime(std::string msg, bool finishCommandQueue, bool setNewTimestamp)
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

void BasePBCTOperator::setReportingParameters(bool verbose,
                                              uint32_t reportKthIteration,
                                              std::string intermediatePrefix)
{
    this->verbose = verbose;
    this->reportKthIteration = reportKthIteration;
    this->intermediatePrefix = intermediatePrefix;
}

double BasePBCTOperator::adjointProductTest()
{
    std::shared_ptr<cl::Buffer> xa_buf; // X buffers
    allocateXBuffers(1);
    xa_buf = getXBuffer(0);
    allocateBBuffers(1);
    std::shared_ptr<cl::Buffer> ba_buf; // B buffers
    ba_buf = getBBuffer(0);
    project(*x_buf, *ba_buf);
    backproject(*b_buf, *xa_buf);
    double bdotAx = scalarProductBBuffer_barrier_double(*b_buf, *ba_buf);
    double ATbdotx = scalarProductXBuffer_barrier_double(*x_buf, *xa_buf);
    return (bdotAx / ATbdotx);
}

} // namespace KCT
