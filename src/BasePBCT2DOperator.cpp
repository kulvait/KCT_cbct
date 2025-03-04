#include "BasePBCT2DOperator.hpp"

namespace KCT {

int BasePBCT2DOperator::initializeOpenCL(uint32_t platformID,
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
        PBCT2DLocalNDRangeFactory localRangeFactory(vdimx, vdimy, maxWorkGroupSize);
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

void BasePBCT2DOperator::initializeCVPProjector(bool useBarrierCalls, uint32_t LOCALARRAYSIZE)
{
    if(!isOpenCLInitialized())
    {
        this->useCVPProjector = true;
        this->useBarrierImplementation = useBarrierCalls;
        this->useSiddonProjector = false;
        this->useTTProjector = false;
        CLINCLUDEutils();
        CLINCLUDEinclude();
        if(useBarrierCalls)
        {
            addOptString(io::xprintf("-DLOCALARRAYSIZE=%d", LOCALARRAYSIZE));
            this->LOCALARRAYSIZE = LOCALARRAYSIZE;
            CLINCLUDEpbct2d_cvp();
            CLINCLUDEpbct2d_cvp_barrier();
        } else
        {
            CLINCLUDEpbct2d_cvp();
        }
        // CLINCLUDEbackprojector();
    } else
    {
        KCTERR("Could not initialize projector when OpenCL was already initialized.");
    }
}

void BasePBCT2DOperator::initializeSiddonProjector(uint32_t probesPerEdgeX, uint32_t probesPerEdgeY)
{
    if(!isOpenCLInitialized())
    {
        KCTERR("Siddon projector not yet implemented for PBCT2D!");
        this->useSiddonProjector = true;
        this->pixelGranularity = { probesPerEdgeX, probesPerEdgeY };
        this->useCVPProjector = false;
        this->useTTProjector = false;
        CLINCLUDEutils();
        CLINCLUDEinclude();
        CLINCLUDEprojector_siddon();
        CLINCLUDEbackprojector_siddon();
    } else
    {
        KCTERR("Could not initialize projector when OpenCL was already initialized.");
    }
}

void BasePBCT2DOperator::initializeTTProjector()
{
    if(!isOpenCLInitialized())
    {
        KCTERR("TT projector not yet implemented for PBCT2D!");
        this->useTTProjector = true;
        this->useCVPProjector = false;
        this->useSiddonProjector = false;
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

void BasePBCT2DOperator::initializeVolumeConvolution()
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

void BasePBCT2DOperator::initializeProximal()
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

void BasePBCT2DOperator::initializeGradient()
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

void BasePBCT2DOperator::useJacobiVectorCLCode()
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

int BasePBCT2DOperator::problemSetup(
    std::vector<std::shared_ptr<geometry::Geometry3DParallelI>> geometries,
    double geometryAtY,
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
    this->geometryAtY = geometryAtY;
    this->voxelSpacingX = voxelSpacingX;
    this->voxelSpacingY = voxelSpacingY;
    this->voxelSpacingZ = voxelSpacingZ;
    PM3Vector.clear();
    scalingFactorVector.clear();
    cl_double3 CM;
    float pixelArea, detectorTilt, scalingFactor;
    for(uint32_t k = 0; k != geometries.size(); k++)
    {
        std::shared_ptr<geometry::Geometry3DParallelI> G = geometries[k];
        pixelArea = G->pixelArea();
        detectorTilt = G->detectorTilt();
        scalingFactor = 1.0f / (pixelArea * detectorTilt);
        G->projectionMatrixPXAsVector3((double*)&CM, geometryAtY);
        PM3Vector.emplace_back(CM);
        scalingFactorVector.emplace_back(scalingFactor);
    }
    voxelSizes = cl_double3({ voxelSpacingX, voxelSpacingY, voxelSpacingZ });
    voxelSizesF = cl_float3({ static_cast<float>(voxelSpacingX), static_cast<float>(voxelSpacingY),
                              static_cast<float>(voxelSpacingZ) });
    volumeCenter = cl_double2({ volumeCenterX, volumeCenterY });
    initReductionBuffers();
    return 0;
}

int BasePBCT2DOperator::allocateXBuffers(uint32_t xBufferCount)
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

int BasePBCT2DOperator::allocateBBuffers(uint32_t bBufferCount)
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

int BasePBCT2DOperator::allocateTmpXBuffers(uint32_t xBufferCount)
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

int BasePBCT2DOperator::allocateTmpBBuffers(uint32_t bBufferCount)
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

std::shared_ptr<cl::Buffer> BasePBCT2DOperator::getXBuffer(uint32_t i)
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

std::shared_ptr<cl::Buffer> BasePBCT2DOperator::getBBuffer(uint32_t i)
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

std::shared_ptr<cl::Buffer> BasePBCT2DOperator::getTmpXBuffer(uint32_t i)
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

std::shared_ptr<cl::Buffer> BasePBCT2DOperator::getTmpBBuffer(uint32_t i)
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
int BasePBCT2DOperator::backproject(cl::Buffer& B,
                                    cl::Buffer& X,
                                    uint32_t initialProjectionIndex,
                                    uint32_t projectionIncrement,
                                    float additionalScaling)
{
    Q[0]->enqueueFillBuffer<cl_float>(X, FLOATZERO, 0, XDIM * sizeof(float));
    cl::NDRange globalRange(vdimx, vdimy);
    cl::NDRange localRange = backprojectorLocalNDRange;
    cl_double3 CM;
    float scalingFactor;
    unsigned long frameSize = pdimx * pdimy;
    unsigned long offset;
    int k_from = 0;
    int k_to = pdimy;
    for(std::size_t i = initialProjectionIndex; i < pdimz; i += projectionIncrement)
    {
        CM = PM3Vector[i];
        scalingFactor = scalingFactorVector[i] * additionalScaling;
        offset = i * frameSize;
        if(useSiddonProjector)
        {
            KCTERR("Siddon operators are not yet implemented for PBCT2D.");
        } else if(useTTProjector)
        {
            KCTERR("Footprint operators are not yet implemented for PBCT2D.");
        } else
        {
            algFLOAT_pbct2d_cutting_voxel_backproject(X, B, offset, CM, vdims, voxelSizes,
                                                      volumeCenter, pdims, scalingFactor, k_from,
                                                      k_to, globalRange, localRange);
        }
    }
    return 0;
}

int BasePBCT2DOperator::backproject_kaczmarz(cl::Buffer& B,
                                             cl::Buffer& X,
                                             uint32_t initialProjectionIndex,
                                             uint32_t projectionIncrement,
                                             float additionalScaling)
{
    Q[0]->enqueueFillBuffer<cl_float>(X, FLOATZERO, 0, XDIM * sizeof(float));
    cl::NDRange globalRange(vdimx, vdimy);
    cl::NDRange localRange = backprojectorLocalNDRange;
    cl_double3 CM;
    float scalingFactor;
    unsigned long frameSize = pdimx * pdimy;
    unsigned long offset;
    int k_from = 0;
    int k_to = pdimy;
    for(std::size_t i = initialProjectionIndex; i < pdimz; i += projectionIncrement)
    {
        CM = PM3Vector[i];
        scalingFactor = scalingFactorVector[i] * additionalScaling;
        offset = i * frameSize;
        if(useSiddonProjector)
        {
            KCTERR("Siddon operators are not yet implemented for PBCT2D.");
        } else if(useTTProjector)
        {
            KCTERR("Footprint operators are not yet implemented for PBCT2D.");
        } else
        {
            algFLOAT_pbct2d_cutting_voxel_backproject_kaczmarz(
                X, B, offset, CM, vdims, voxelSizes, volumeCenter, pdims, scalingFactor, k_from,
                k_to, globalRange, localRange);
        }
    }
    return 0;
}

/**
 * @param initialProjectionIndex For OS SART, 0 by default
 * @param projectionIncrement For OS SART, 1 by default
 *
 */
int BasePBCT2DOperator::project(cl::Buffer& X,
                                cl::Buffer& B,
                                uint32_t initialProjectionIndex,
                                uint32_t projectionIncrement,
                                float additionalScaling)
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
    cl_double3 CM;
    float scalingFactor;
    unsigned long frameSize = pdimx * pdimy;
    unsigned long offset;
    int k_from = 0;
    int k_count = pdimy;
    for(std::size_t i = initialProjectionIndex; i < pdimz; i += projectionIncrement)
    {
        CM = PM3Vector[i];
        scalingFactor = scalingFactorVector[i] * additionalScaling;
        offset = i * frameSize;
        if(useSiddonProjector)
        {
            KCTERR("Siddon operators are not yet implemented for PBCT.");
        } else if(useTTProjector)
        {
            KCTERR("Footprint operators are not yet implemented for PBCT.");
        } else
        {
            if(useBarrierImplementation)
            {
                cl::NDRange globalRange(vdimx, vdimy);
                algFLOAT_pbct2d_cutting_voxel_project_barrier(
                    X, B, offset, CM, vdims, voxelSizes, volumeCenter, pdims, scalingFactor, k_from,
                    k_count, LOCALARRAYSIZE, globalRange, localRange);
            } else
            {
                cl::NDRange globalRange(vdimx, vdimy);
                algFLOAT_pbct2d_cutting_voxel_project(X, B, offset, CM, vdims, voxelSizes,
                                                      volumeCenter, pdims, scalingFactor, k_from,
                                                      k_count, globalRange, localRange);
            }
        }
    }
    return 0;
}

int BasePBCT2DOperator::kaczmarz_product(cl::Buffer& K,
                                         uint32_t initialProjectionIndex,
                                         uint32_t projectionIncrement,
                                         float additionalScaling)
{
    Q[0]->enqueueFillBuffer<cl_float>(K, FLOATZERO, 0, BDIM * sizeof(float));
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
    cl_double3 CM;
    float scalingFactor;
    unsigned long frameSize = pdimx * pdimy;
    unsigned long offset;
    int k_from = 0;
    int k_count = pdimy;
    if(useBarrierImplementation)
    {
        LOGW << "Kraczmarz vector for PBCT2D operator barrier not yet implemented, using "
                "non barrier implementation.";
    }
    for(std::size_t i = initialProjectionIndex; i < pdimz; i += projectionIncrement)
    {
        CM = PM3Vector[i];
        scalingFactor = scalingFactorVector[i] * additionalScaling;
        // LOGD << io::xprintf("Scaling factor=%f square=%f",
        //                    scalingFactorVector[i] * additionalScaling, scalingFactor);
        offset = i * frameSize;
        if(useSiddonProjector)
        {
            KCTERR("Siddon operators are not yet implemented for PBCT.");
        } else if(useTTProjector)
        {
            KCTERR("Footprint operators are not yet implemented for PBCT.");
        } else
        {
            if(useBarrierImplementation)
            {
                // "Kraczmarz vector for PBCT2D operator barrier not yet implemented, using "
                //        "non barrier implementation.";
                // cl::NDRange globalRange(vdimx, vdimy);
                // algFLOAT_pbct2d_cutting_voxel_kraczmarz_product_barrier(
                //    K, offset, CM, vdims, voxelSizes, volumeCenter, pdims, scalingFactor, k_from,
                //    k_count, LOCALARRAYSIZE, globalRange, localRange);
                cl::NDRange globalRange(vdimx, vdimy);
                algFLOAT_pbct2d_cutting_voxel_kaczmarz_product(
                    K, offset, CM, vdims, voxelSizes, volumeCenter, pdims, scalingFactor, k_from,
                    k_count, globalRange, localRange);
            } else
            {
                cl::NDRange globalRange(vdimx, vdimy);
                algFLOAT_pbct2d_cutting_voxel_kaczmarz_product(
                    K, offset, CM, vdims, voxelSizes, volumeCenter, pdims, scalingFactor, k_from,
                    k_count, globalRange, localRange);
            }
        }
    }
    return 0;
}

void BasePBCT2DOperator::setGradientType(GradientType type)
{
    useGradientType = type;
    std::string gradientTypeString = GradientTypeToString(useGradientType);
    LOGD << io::xprintf("Setting gradient computation method to %s", gradientTypeString.c_str());
}

int BasePBCT2DOperator::volume_gradient2D(cl::Buffer& F, cl::Buffer& GX, cl::Buffer& GY)
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

int BasePBCT2DOperator::volume_gradient2D_adjoint(cl::Buffer& GX, cl::Buffer& GY, cl::Buffer& D)
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

// Scaling factor is a expression 1.0f / (pixelArea * detectorTilt), where f is source to
// detector distance and pixel sizes are (px and py)  Focal length
// http://ksimek.github.io/2013/08/13/intrinsic/
std::vector<float> BasePBCT2DOperator::computeScalingFactors()
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

void BasePBCT2DOperator::writeVolume(cl::Buffer& X, float* x, std::string path)
{
    bufferIntoArray(X, x, XDIM);
    bool arrayxmajor = true;
    bool outxmajor = true;
    io::DenFileInfo::create3DDenFileFromArray(x, arrayxmajor, path, io::DenSupportedType::FLOAT32,
                                              vdimx, vdimy, vdimz, outxmajor);
}

void BasePBCT2DOperator::writeProjections(cl::Buffer& B, float* b, std::string path)
{
    bufferIntoArray(B, b, BDIM);
    bool arrayxmajor = false;
    bool outxmajor = true;
    io::DenFileInfo::create3DDenFileFromArray(b, arrayxmajor, path, io::DenSupportedType::FLOAT32,
                                              pdimx, pdimy, pdimz, outxmajor);
}

void BasePBCT2DOperator::setTimestamp(bool finishCommandQueue)
{
    if(finishCommandQueue)
    {
        Q[0]->finish();
    }
    timestamp = std::chrono::steady_clock::now();
}
std::chrono::milliseconds BasePBCT2DOperator::millisecondsFromTimestamp(bool setNewTimestamp)
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
BasePBCT2DOperator::printTime(std::string msg, bool finishCommandQueue, bool setNewTimestamp)
{
    if(finishCommandQueue)
    {
        Q[0]->finish();
    }
    auto duration = millisecondsFromTimestamp(setNewTimestamp);
    return io::xprintf("%s: %0.2fs", msg.c_str(), duration.count() / 1000.0);
}

void BasePBCT2DOperator::reportTime(std::string msg, bool finishCommandQueue, bool setNewTimestamp)
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

void BasePBCT2DOperator::setVerbose(bool verbose, std::string intermediatePrefix)
{
    this->verbose = verbose;
    this->intermediatePrefix = intermediatePrefix;
}

double BasePBCT2DOperator::adjointProductTest(float* x, float* b)
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

} // namespace KCT
