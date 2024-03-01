#include "PartialPBCTOperator.hpp"

namespace KCT {

int PartialPBCTOperator::initializeOpenCL(uint32_t platformID,
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
        PBCTLocalNDRangeFactory localRangeFactory(vdimx, vdimy, vzblock_maxsize, maxWorkGroupSize);
        bool verbose = true;
        this->projectorLocalNDRange
            = localRangeFactory.getProjectorLocalNDRange(projectorLocalNDRange, verbose);
        this->projectorLocalNDRangeBarrier
            = localRangeFactory.getProjectorBarrierLocalNDRange(projectorLocalNDRange, verbose);
        this->backprojectorLocalNDRange
            = localRangeFactory.getBackprojectorLocalNDRange(backprojectorLocalNDRange, verbose);
        return 0;
    } else
    {
        std::string ERR = io::xprintf("Wrong initialization of OpenCL with code %d!", val);
        KCTERR(ERR);
    }
}

void PartialPBCTOperator::initializeCVPProjector(bool useBarrierCalls, uint32_t LOCALARRAYSIZE)
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
    } else
    {
        KCTERR("Could not initialize projector when OpenCL was already initialized.");
    }
}

void PartialPBCTOperator::initializeSidonProjector(uint32_t probesPerEdgeX, uint32_t probesPerEdgeY)
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

void PartialPBCTOperator::initializeTTProjector()
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

void PartialPBCTOperator::initializeVolumeConvolution()
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

void PartialPBCTOperator::useJacobiVectorCLCode()
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

int PartialPBCTOperator::problemSetup(
    std::vector<std::shared_ptr<geometry::Geometry3DParallelI>> geometries,
    double voxelSpacingX,
    double voxelSpacingY,
    double voxelSpacingZ,
    double volumeCenterX,
    double volumeCenterY,
    double volumeCenterZ)
{
    if(geometries.size() != pzblock_maxsize)
    {
        std::string ERR
            = io::xprintf("The pzblock_maxsize=%d and the size of camera geometries vector is %d.",
                          pzblock_maxsize, geometries.size());
        if(geometries.size() < pzblock_maxsize)
        {
            KCTERR(ERR);
        }
        LOGI << ERR;
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

int PartialPBCTOperator::allocateXBuffers(uint32_t xBufferCount)
{
    cl_int err;
    std::shared_ptr<cl::Buffer> xbp;
    uint64_t bytesize = sizeof(float) * XDIM_maxsize;
    while(this->x_buffers.size() < xBufferCount)
    {
        xbp = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, bytesize, nullptr, &err);
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of X buffer with error code %d!", err);
            return -1;
        }
        x_buffers.push_back(xbp);
    }

    return 0;
}

int PartialPBCTOperator::allocateBBuffers(uint32_t bBufferCount)
{
    cl_int err;
    std::shared_ptr<cl::Buffer> bbp;
    uint64_t bytesize = sizeof(float) * BDIM_maxsize;
    while(this->b_buffers.size() < bBufferCount)
    {
        bbp = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, bytesize, nullptr, &err);
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of B buffer with error code %d!", err);
            return -1;
        }
        b_buffers.push_back(bbp);
    }
    return 0;
}

int PartialPBCTOperator::allocateTmpXBuffers(uint32_t xBufferCount)
{
    cl_int err;
    std::shared_ptr<cl::Buffer> xbp;
    uint64_t bytesize = sizeof(float) * XDIM_maxsize;
    while(this->tmp_x_buffers.size() < xBufferCount)
    {
        xbp = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, bytesize, nullptr, &err);
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of X buffer with error code %d!", err);
            return -1;
        }
        tmp_x_buffers.push_back(xbp);
    }
    return 0;
}

int PartialPBCTOperator::allocateTmpBBuffers(uint32_t bBufferCount)
{
    cl_int err;
    std::shared_ptr<cl::Buffer> bbp;
    uint64_t bytesize = sizeof(float) * BDIM_maxsize;
    while(this->tmp_b_buffers.size() < bBufferCount)
    {
        bbp = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, bytesize, nullptr, &err);
        if(err != CL_SUCCESS)
        {
            LOGE << io::xprintf("Unsucessful initialization of B buffer with error code %d!", err);
            return -1;
        }
        tmp_b_buffers.push_back(bbp);
    }
    return 0;
}

std::shared_ptr<cl::Buffer> PartialPBCTOperator::getXBuffer(uint32_t i)
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

std::shared_ptr<cl::Buffer> PartialPBCTOperator::getBBuffer(uint32_t i)
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

std::shared_ptr<cl::Buffer> PartialPBCTOperator::getTmpXBuffer(uint32_t i)
{
    if(i < tmp_x_buffers.size())
    {
        return tmp_x_buffers[i];
    } else
    {
        std::string err
            = io::xprintf("Index %d is out of range of the tmp_x_buffers vector of size %d!", i,
                          tmp_x_buffers.size());
        KCTERR(err);
    }
}

std::shared_ptr<cl::Buffer> PartialPBCTOperator::getTmpBBuffer(uint32_t i)
{
    if(i < tmp_b_buffers.size())
    {
        return tmp_b_buffers[i];
    } else
    {
        std::string err
            = io::xprintf("Index %d is out of range of the tmp_b_buffers vector of size %d!", i,
                          tmp_b_buffers.size());
        KCTERR(err);
    }
}

/**
 * @param initialProjectionIndex For OS SART 0 by default
 * @param projectionIncrement For OS SART 1 by default
 *
 */
int PartialPBCTOperator::backproject(cl::Buffer& B,
                                     cl::Buffer& X,
                                     uint32_t initialProjectionIndex,
                                     uint32_t projectionIncrement)
{
    std::string ERR = "Not yet implemented";
    KCTERR(ERR);
    /*
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
                                                        volumeCenter, pdims, scalingFactor,
       globalRange, localRange);
            }
        }
        return 0;*/
}

/**
 * @param initialProjectionIndex For OS SART, 0 by default
 * @param projectionIncrement For OS SART, 1 by default
 *
 */
int PartialPBCTOperator::project_partial(uint32_t QID,
                                         cl::Buffer& X,
                                         cl::Buffer& B,
                                         uint32_t vdimz_local,
                                         float voxelzCenterOffset,
                                         uint32_t geometries_from,
                                         uint32_t geometries_to,
                                         uint32_t initialProjectionIndex,
                                         uint32_t projectionIncrement)
{
    cl::NDRange localRange;
    if(useBarrierImplementation)
    {
        localRange = projectorLocalNDRangeBarrier;
        if(vdimz_local != vzblock_maxsize)
        {
            std::string err
                = io::xprintf("Barrier calls are not implemented for uneven partitioning of the "
                              "volume vdimz_local=%d vzblock_maxsize=%d.",
                              vdimz_local, vzblock_maxsize);
            KCTERR(err);
        }
    } else
    {
        localRange = projectorLocalNDRange;
    }
    cl_int3 vdims = cl_int3({ int(vdimx), int(vdimy), int(vdimz_local) });
    // clang-format off
    // cl::NDRange barrierGlobalRange = cl::NDRange(vdimx, vdimy, vdimz);
    // std::shared_ptr<cl::NDRange> barrierLocalRange
    //    = std::make_shared<cl::NDRange>(projectorLocalNDRangeBarrier);
    // clang-format on
    cl_double8 CM;
    float scalingFactor;
    unsigned long frameSize = pdimx * pdimy;
    unsigned long offset;
    cl_double3 volumeCenter_local = volumeCenter;
    volumeCenter_local.z -= (voxelzCenterOffset * voxelSizes.z);
    uint64_t IND;
    uint32_t i_start, i_stop;
    if(initialProjectionIndex == 0 && projectionIncrement == 1)
    {
        i_start = geometries_from;
        i_stop = geometries_to;
    } else
    {
        i_start = initialProjectionIndex;
        i_stop = PM8Vector.size();
    }

    for(std::size_t i = i_start; i < i_stop; i += projectionIncrement)
    {
        if(i >= geometries_from && i < geometries_to)
        {
            CM = PM8Vector[i];
            scalingFactor = scalingFactorVector[i];
            IND = i - geometries_from;
            offset = IND * frameSize;
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
                    cl::NDRange globalRange(vdimx, vdimy, vdimz_local);
                    algFLOAT_pbct_cutting_voxel_project_barrier(
                        X, B, offset, CM, vdims, voxelSizes, volumeCenter_local, pdims,
                        scalingFactor, LOCALARRAYSIZE, globalRange, localRange, false, QID);
                } else
                {
                    cl::NDRange globalRange(vdimz_local, vdimy, vdimx);
                    localRange = NDRangeHelper::flipNDRange(projectorLocalNDRange);
                    algFLOAT_pbct_cutting_voxel_project(X, B, offset, CM, vdims, voxelSizes,
                                                        volumeCenter_local, pdims, scalingFactor,
                                                        globalRange, localRange, false, QID);
                }
            }
        }
    }
    return 0;
}

// Scaling factor is a expression f*f/(px*py), where f is source to detector distance and pixel
// sizes are (px and py)  Focal length http://ksimek.github.io/2013/08/13/intrinsic/
std::vector<float> PartialPBCTOperator::computeScalingFactors()
{
    std::vector<float> scalingFactors;
    float pixelArea, detectorTilt, scalingFactor;
    for(uint32_t k = 0; k != geometries.size(); k++)
    {
        std::shared_ptr<geometry::Geometry3DParallelI> G = geometries[k];
        pixelArea = G->pixelArea();
        detectorTilt = G->detectorTilt();
        scalingFactor = 1.0f / (pixelArea * detectorTilt);
        scalingFactors.emplace_back(scalingFactor);
    }
    return scalingFactors;
}

void PartialPBCTOperator::writeVolume(cl::Buffer& X, float* x, std::string path)
{
    std::string ERR = "Not implemented";
    KCTERR(ERR);
    /*
         bufferIntoArray(X, x, XDIM);
         bool arrayxmajor = true;
         bool outxmajor = true;
         io::DenFileInfo::create3DDenFileFromArray(x, arrayxmajor, path,
       io::DenSupportedType::FLOAT32, vdimx, vdimy, vdimz, outxmajor);*/
}

void PartialPBCTOperator::writeProjections(cl::Buffer& B, float* b, std::string path)
{
    std::string ERR = "Not implemented";
    KCTERR(ERR);
    /*
        bufferIntoArray(B, b, BDIM);
        bool arrayxmajor = false;
        bool outxmajor = true;
        io::DenFileInfo::create3DDenFileFromArray(b, arrayxmajor, path,
       io::DenSupportedType::FLOAT32, pdimx, pdimy, pdimz, outxmajor);*/
}

void PartialPBCTOperator::setTimestamp(bool finishCommandQueue)
{
    if(finishCommandQueue)
    {
        Q[0]->finish();
    }
    timestamp = std::chrono::steady_clock::now();
}
std::chrono::milliseconds PartialPBCTOperator::millisecondsFromTimestamp(bool setNewTimestamp)
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
PartialPBCTOperator::printTime(std::string msg, bool finishCommandQueue, bool setNewTimestamp)
{
    if(finishCommandQueue)
    {
        Q[0]->finish();
    }
    auto duration = millisecondsFromTimestamp(setNewTimestamp);
    return io::xprintf("%s: %0.2fs", msg.c_str(), duration.count() / 1000.0);
}

void PartialPBCTOperator::reportTime(std::string msg, bool finishCommandQueue, bool setNewTimestamp)
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

void PartialPBCTOperator::setVerbose(bool verbose, std::string intermediatePrefix)
{
    this->verbose = verbose;
    this->intermediatePrefix = intermediatePrefix;
}

double PartialPBCTOperator::adjointProductTest(float* x, float* b)
{
    /*
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
    */
    LOGE << "Unimplemented";
    return 0;
}

cl::NDRange PartialPBCTOperator::guessProjectionLocalNDRange(bool barrierCalls)
{
    cl::NDRange projectorLocalNDRange;
    if(barrierCalls)
    {

        if(vdimx % 64 == 0 && vdimy % 4 == 0 && workGroupSize >= 256)
        {
            projectorLocalNDRange = cl::NDRange(64, 4, 1); // 9.45 Barrier
        } else
        {
            projectorLocalNDRange = cl::NullRange;
        }
    } else
    {
        if(vzblock_maxsize % 4 == 0 && vdimy % 64 == 0 && workGroupSize >= 256)
        {
            projectorLocalNDRange = cl::NDRange(4, 64, 1); // 23.23 RELAXED
        } else
        {
            projectorLocalNDRange = cl::NullRange;
        }
    }
    return projectorLocalNDRange;
}

cl::NDRange PartialPBCTOperator::guessBackprojectorLocalNDRange()
{
    cl::NDRange backprojectorLocalNDRange;
    if(vdimx % 4 == 0 && vdimy % 16 == 0 && workGroupSize >= 64)
    {
        backprojectorLocalNDRange = cl::NDRange(4, 16, 1); // 4.05 RELAXED
    } else
    {
        backprojectorLocalNDRange = cl::NullRange;
    }
    return backprojectorLocalNDRange;
}

} // namespace KCT
