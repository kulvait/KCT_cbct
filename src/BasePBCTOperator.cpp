#include "BasePBCTOperator.hpp"

namespace KCT {

bool BasePBCTOperator::isLocalRangeAdmissible(cl::NDRange& localRange)
{
    size_t dim = localRange.dimensions();
    if(dim == 0)
    {
        return false;
    } else if(dim == 3)
    {
        uint64_t totalSize = localRange[0] * localRange[1] * localRange[2];
        if(totalSize > maxWorkGroupSize)
        {
            return false;
        }
        if(localRange[0] == 0 || vdimx % localRange[0] != 0)
        {
            return false;
        }
        if(localRange[1] == 0 || vdimy % localRange[1] != 0)
        {
            return false;
        }
        if(localRange[2] == 0 || vdimz % localRange[2] != 0)
        {
            return false;
        }
        return true;
    } else
    {
        return false;
    }
}

void BasePBCTOperator::checkLocalRange(cl::NDRange& localRange, std::string name)
{
    size_t dim = localRange.dimensions();
    std::string ERR;
    if(dim == 0)
    {
        LOGD << io::xprintf("%s = cl::NDRange()", name.c_str());
    } else if(dim == 3)
    {
        uint64_t totalSize = localRange[0] * localRange[1] * localRange[2];
        if(totalSize > maxWorkGroupSize)
        {
            ERR = io::xprintf("%s has total size %d exceeding maxWorkGroupSize=%d!", name.c_str(),
                              totalSize, maxWorkGroupSize);
            KCTERR(ERR);
        }
        if(totalSize == 0)
        {
            ERR = io::xprintf("There is 0 in %s definition!", name.c_str());
            KCTERR(ERR);
        }
        if(vdimx % localRange[0] != 0)
        {
            ERR = io::xprintf("%s vdimx %% localRange[0] != 0 %d %% %d!=0", name.c_str(), vdimx,
                              localRange[0]);
            KCTERR(ERR);
        }
        if(vdimy % localRange[1] != 0)
        {
            ERR = io::xprintf("%s vdimy %% localRange[1] != 0 %d %% %d!=0", name.c_str(), vdimy,
                              localRange[1]);
            KCTERR(ERR);
        }
        if(vdimz % localRange[2] != 0)
        {
            ERR = io::xprintf("%s vdimz %% localRange[2] != 0 %d %% %d!=0", name.c_str(), vdimz,
                              localRange[2]);
            KCTERR(ERR);
        }
        LOGD << io::xprintf("%s = cl::NDRange(%d, %d, %d)", name.c_str(), localRange[0],
                            localRange[1], localRange[2]);
    } else
    {
        ERR = io::xprintf("%s has dimension %d but it shall be 3!", name.c_str(), dim);
        KCTERR(ERR);
    }
}

int BasePBCTOperator::initializeOpenCL(uint32_t platformID,
                                       uint32_t* deviceIds,
                                       uint32_t deviceIdsLength,
                                       std::string xpath,
                                       bool debug,
                                       bool relaxed,
                                       cl::NDRange projectorLocalNDRange,
                                       cl::NDRange backprojectorLocalNDRange)
{
    int val
        = Kniha::initializeOpenCL(platformID, deviceIds, deviceIdsLength, xpath, debug, relaxed);
    if(val == 0)
    {
        std::size_t dim = projectorLocalNDRange.dimensions();
        if(dim == 0
           || (dim == 3 && projectorLocalNDRange[0] == 0 && projectorLocalNDRange[1] == 1
               && projectorLocalNDRange[2] == 1)) // guess
        {
            this->projectorLocalNDRange = guessProjectionLocalNDRange(false);
            this->projectorLocalNDRangeBarrier = guessProjectionLocalNDRange(true);
        } else if(isLocalRangeAdmissible(projectorLocalNDRange))
        {
            this->projectorLocalNDRange = projectorLocalNDRange;
            this->projectorLocalNDRangeBarrier = projectorLocalNDRange;
        } else
        {
            this->projectorLocalNDRange = cl::NullRange;
            this->projectorLocalNDRangeBarrier = cl::NullRange;
        }
        dim = backprojectorLocalNDRange.dimensions();
        if(dim == 0
           || (dim == 3 && backprojectorLocalNDRange[0] == 0 && backprojectorLocalNDRange[1] == 1
               && backprojectorLocalNDRange[2] == 1)) // guess
        {
            this->backprojectorLocalNDRange = guessBackprojectorLocalNDRange();
        } else if(isLocalRangeAdmissible(backprojectorLocalNDRange))
        {
            this->backprojectorLocalNDRange = backprojectorLocalNDRange;
        } else
        {
            this->backprojectorLocalNDRange = cl::NullRange;
        }
        checkLocalRange(this->projectorLocalNDRange, "projectorLocalNDRange");
        checkLocalRange(this->projectorLocalNDRangeBarrier, "projectorLocalNDRangeBarrier");
        checkLocalRange(this->backprojectorLocalNDRange, "backprojectorLocalNDRange");
        return 0;
    } else
    {
        std::string ERR = io::xprintf("Wrong initialization of OpenCL with code %d!", val);
        KCTERR(ERR);
    }
}

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
            CLINCLUDEpbct_cvp();
            CLINCLUDEpbct_cvp_barrier();
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
    initReductionBuffers();
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
        } else
        {
            x_buffers.push_back(xbufptr);
        }
    }
    return 0;
}

int BasePBCTOperator::allocateBBuffers(uint32_t bBufferCount)
{
    cl_int err;
    std::shared_ptr<cl::Buffer> bbufptr;
    while(this->b_buffers.size() < bBufferCount)
    {
        uint64_t size = uint64_t(sizeof(float)) * uint64_t(BDIM);
        bbufptr = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, size, nullptr, &err);
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
    cl::NDRange localRange = backprojectorLocalNDRange;
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
    cl::NDRange localRange;
    if(useBarrierImplementation)
    {
        localRange = projectorLocalNDRangeBarrier;
    } else
    {
        localRange = projectorLocalNDRange;
    }
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
                cl::NDRange globalRange(vdimx, vdimy, vdimz);
                algFLOAT_pbct_cutting_voxel_project_barrier(
                    X, B, offset, CM, vdims, voxelSizes, volumeCenter, pdims, scalingFactor,
                    LOCALARRAYSIZE, globalRange, localRange);
            } else
            {
                cl::NDRange globalRange(vdimz, vdimy, vdimx);
                localRange = cl::NDRange(projectorLocalNDRange[2], projectorLocalNDRange[1],
                              projectorLocalNDRange[0]);
                algFLOAT_pbct_cutting_voxel_project(X, B, offset, CM, vdims, voxelSizes,
                                                    volumeCenter, pdims, scalingFactor, globalRange,
                                                    localRange);
            }
        }
    }
    return 0;
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

void BasePBCTOperator::writeVolume(cl::Buffer& X, float* x, std::string path)
{
    bufferIntoArray(X, x, XDIM);
    bool arrayxmajor = true;
    bool outxmajor = true;
    io::DenFileInfo::create3DDenFileFromArray(x, arrayxmajor, path, io::DenSupportedType::FLOAT32,
                                              vdimx, vdimy, vdimz, outxmajor);
}

void BasePBCTOperator::writeProjections(cl::Buffer& B, float* b, std::string path)
{
    bufferIntoArray(B, b, BDIM);
    bool arrayxmajor = false;
    bool outxmajor = true;
    io::DenFileInfo::create3DDenFileFromArray(b, arrayxmajor, path, io::DenSupportedType::FLOAT32,
                                              pdimx, pdimy, pdimz, outxmajor);
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

void BasePBCTOperator::setVerbose(bool verbose, std::string intermediatePrefix)
{
    this->verbose = verbose;
    this->intermediatePrefix = intermediatePrefix;
}

double BasePBCTOperator::adjointProductTest(float* x, float* b)
{
    std::shared_ptr<cl::Buffer> x_buf, xa_buf; // X buffers
    allocateXBuffers(2);
    x_buf = getXBuffer(0);
    arrayIntoBuffer(x, *x_buf, XDIM);
    xa_buf = getXBuffer(1);
    allocateBBuffers(2);
    std::shared_ptr<cl::Buffer> b_buf, ba_buf; // B buffers
    b_buf = getBBuffer(0);
    arrayIntoBuffer(b, *b_buf, BDIM);
    ba_buf = getBBuffer(1);
    project(*x_buf, *ba_buf);
    backproject(*b_buf, *xa_buf);
    double bdotAx = scalarProductBBuffer_barrier_double(*b_buf, *ba_buf);
    double ATbdotx = scalarProductXBuffer_barrier_double(*x_buf, *xa_buf);
    return (bdotAx / ATbdotx);
}

cl::NDRange BasePBCTOperator::guessProjectionLocalNDRange(bool barrierCalls)
{
    cl::NDRange projectorLocalNDRange;
    LOGI << io::xprintf("vdimx=%d vdimy=%d vdimz=%d maxWorkGroupSize=%d", vdimx, vdimy, vdimz,
                        maxWorkGroupSize);
    if(barrierCalls)
    {
        if(vdimx % 64 == 0 && vdimy % 4 == 0 && 64 * 4 <= maxWorkGroupSize)
        {
            projectorLocalNDRange = cl::NDRange(64, 4, 1); // 9.45 Barrier
        } else if(vdimx % 16 == 0 && vdimy % 16 == 0 && 16 * 16 <= maxWorkGroupSize)
        {
            projectorLocalNDRange = cl::NDRange(16, 16, 1); // 3.8 Barrier
        } else if(vdimx % 8 == 0 && vdimy % 8 == 0 && 8 * 8 <= maxWorkGroupSize)
        {
            projectorLocalNDRange = cl::NDRange(8, 8, 1); // 10.9 Barrier
        } else if(vdimx % 4 == 0 && vdimy % 4 == 0 && 4 * 4 <= maxWorkGroupSize)
        {
            projectorLocalNDRange = cl::NDRange(4, 4, 1); // 32.5 Barrier
        } else
        {
            projectorLocalNDRange = cl::NullRange;
        }
    } else
    {
        if(vdimx % 4 == 0 && vdimy % 64 == 0 && 4 * 64 <= maxWorkGroupSize)
        {
            projectorLocalNDRange = cl::NDRange(4, 64, 1); // 23.23 RELAXED
        } else if(vdimx % 16 == 0 && vdimy % 16 == 0 && 16 * 16 <= maxWorkGroupSize)
        {
            projectorLocalNDRange = cl::NDRange(16, 16, 1); // 3.8 Barrier
        } else if(vdimx % 8 == 0 && vdimy % 8 == 0 && 8 * 8 <= maxWorkGroupSize)
        {
            projectorLocalNDRange = cl::NDRange(8, 8, 1); // 10.9 Barrier
        } else if(vdimx % 4 == 0 && vdimy % 4 == 0 && 4 * 4 <= maxWorkGroupSize)
        {
            projectorLocalNDRange = cl::NDRange(4, 4, 1); // 32.5 Barrier
        } else
        {
            projectorLocalNDRange = cl::NullRange;
        }
    }
    return projectorLocalNDRange;
}

cl::NDRange BasePBCTOperator::guessBackprojectorLocalNDRange()
{
    cl::NDRange backprojectorLocalNDRange;
    if(vdimx % 4 == 0 && vdimy % 16 == 0 && 4 * 16 <= maxWorkGroupSize)
    {
        backprojectorLocalNDRange = cl::NDRange(4, 16, 1); // 4.05 RELAXED
    } else
    {
        backprojectorLocalNDRange = cl::NullRange;
    }
    return backprojectorLocalNDRange;
}

} // namespace KCT
