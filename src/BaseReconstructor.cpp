#include "BaseReconstructor.hpp"

namespace KCT {

BaseReconstructor::~BaseReconstructor() = default;

void BaseReconstructor::initializeCVPProjector(bool useExactScaling,
                                               bool useElevationCorrection,
                                               bool useBarrierCalls,
                                               uint32_t LOCALARRAYSIZE)
{
    if(!isOpenCLInitialized())
    {
        this->useCVPProjector = true;
        this->useCVPExactProjectionsScaling = useExactScaling;
        this->useCVPElevationCorrection = useElevationCorrection;
        this->useBarrierImplementation = useBarrierCalls;
        this->useSidonProjector = false;
        this->useTTProjector = false;
        CLINCLUDEutils();
        CLINCLUDEinclude();
        if(useCVPElevationCorrection)
        {
            addOptString(io::xprintf("-DELEVATIONCORRECTION"));
        }
        if(useBarrierCalls)
        {
            addOptString(io::xprintf("-DLOCALARRAYSIZE=%d", LOCALARRAYSIZE));
            this->LOCALARRAYSIZE = LOCALARRAYSIZE;
            CLINCLUDEprojector_cvp_barrier();
        } else
        {
            CLINCLUDEprojector();
        }
        CLINCLUDEbackprojector();
        CLINCLUDErescaleProjections();
    } else
    {
        KCTERR("Could not initialize projector when OpenCL was already initialized.");
    }
}

void BaseReconstructor::initializeSidonProjector(uint32_t probesPerEdgeX, uint32_t probesPerEdgeY)
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

void BaseReconstructor::initializeTTProjector()
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

void BaseReconstructor::initializeVolumeConvolution()
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

void BaseReconstructor::useJacobiVectorCLCode()
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

int BaseReconstructor::vectorIntoBuffer(cl::Buffer X, float* v, std::size_t size)
{
    cl_int err = CL_SUCCESS;
    std::string e;
    std::size_t bufferSize;
    X.getInfo(CL_MEM_SIZE, &bufferSize);
    std::size_t totalSize = size * sizeof(float);
    e = io::xprintf("The buffer is %d bytes to represent vector of size %d that is %d bytes.",
                    bufferSize, size, totalSize);
    LOGE << e;
    if(bufferSize >= totalSize)
    {
        err = Q[0]->enqueueWriteBuffer(X, CL_TRUE, 0, totalSize, (void*)v);
        if(err != CL_SUCCESS)
        {
            e = io::xprintf("Unsucessful initialization of Volume with error code %d!", err);
            LOGE << e;
            throw std::runtime_error(e);
        }
    } else
    {
        e = io::xprintf(
            "The buffer is too small %d bytes to represent vector of size %d that is %d bytes.",
            bufferSize, size, totalSize);
        LOGE << e;
        throw std::runtime_error(e);
    }
    return 0;
}

int BaseReconstructor::problemSetup(float* projection,
                                    float* volume,
                                    bool volumeContainsX0,
                                    std::vector<std::shared_ptr<CameraI>> cameraVector,
                                    double voxelSpacingX,
                                    double voxelSpacingY,
                                    double voxelSpacingZ,
                                    double volumeCenterX,
                                    double volumeCenterY,
                                    double volumeCenterZ)
{
    if(cameraVector.size() != pdimz)
    {
        std::string err
            = io::xprintf("The pdimz=%d but the size of camera geometries vector is %d!");
        KCTERR(err);
    }
    this->cameraVector = cameraVector;
    PM12Vector.clear();
    ICM16Vector.clear();
    scalingFactorVector.clear();
    cl_double16 CM, ICM;
    std::array<double, 2> focalLength;
    for(uint32_t k = 0; k != cameraVector.size(); k++)
    {
        std::shared_ptr<CameraI> P = cameraVector[k];
        focalLength = P->focalLength();
        P->projectionMatrixAsVector12((double*)&CM);
        P->inverseProjectionMatrixAsVector16((double*)&ICM);
        PM12Vector.emplace_back(CM);
        ICM16Vector.emplace_back(ICM);
        scalingFactorVector.emplace_back(focalLength[0] * focalLength[1]);
    }
    voxelSizes = cl_double3({ voxelSpacingX, voxelSpacingY, voxelSpacingZ });
    volumeCenter = cl_double3({ volumeCenterX, volumeCenterY, volumeCenterZ });
    std::array<double, 3> centerGlobal = { volumeCenterX, volumeCenterY, volumeCenterZ };
    std::array<double, 3> offsetx = { voxelSpacingX * vdims.x * 0.5, 0.0, 0.0 };
    std::array<double, 3> offsety = { 0.0, voxelSpacingY * vdims.y * 0.5, 0.0 };
    std::array<double, 3> offsetz = { 0.0, 0.0, voxelSpacingZ * vdims.z * 0.5 };
    std::array<double, 3> A, B, C, D, E, F, G, H; // Corners of volume
    std::array<double, 3> center;
    std::array<double, 3> source;
    std::array<double, 3> VN;
    for(uint32_t k = 0; k != cameraVector.size(); k++)
    {
        std::shared_ptr<CameraI> P = cameraVector[k];
        source = P->sourcePosition();
        center = vectorDiff(centerGlobal, source);
        A = vectorDiff(vectorDiff(vectorDiff(center, offsetx), offsety), offsetz);
        B = vectorDiff(vectorDiff(vectorSum(center, offsetx), offsety), offsetz);
        C = vectorDiff(vectorSum(vectorDiff(center, offsetx), offsety), offsetz);
        D = vectorDiff(vectorSum(vectorSum(center, offsetx), offsety), offsetz);
        E = vectorSum(vectorDiff(vectorDiff(center, offsetx), offsety), offsetz);
        F = vectorSum(vectorDiff(vectorSum(center, offsetx), offsety), offsetz);
        G = vectorSum(vectorSum(vectorDiff(center, offsetx), offsety), offsetz);
        H = vectorSum(vectorSum(vectorSum(center, offsetx), offsety), offsetz);
        VN = P->directionVectorVN();
        if(vectorDotProduct(VN, center) < 0)
        {
            LOGW << io::xprintf("Apparently the volume is specified such that its center do not "
                                "belong to the half space orthogonal to the principal ray in %d-th "
                                "projection. VN=(%f,%f,%f), center=(%f,%f, %f), dot(VN,center)=%f.",
                                k, VN[0], VN[1], VN[2], center[0], center[1], center[2],
                                vectorDotProduct(VN, center));
        }
        if(vectorDotProduct(VN, A) < 0 || vectorDotProduct(VN, B) < 0 || vectorDotProduct(VN, C) < 0
           || vectorDotProduct(VN, D) < 0 || vectorDotProduct(VN, E) < 0
           || vectorDotProduct(VN, F) < 0 || vectorDotProduct(VN, G) < 0
           || vectorDotProduct(VN, H) < 0)
        {
            LOGW << io::xprintf(
                "Apparently the volume is so big that some its corners do not fit "
                "to the half space orthogonal to the principal ray in %d-th projection.",
                k);
        }

        // Test positions of corners relative to source
    }
    initReductionBuffers();
    return initializeVectors(projection, volume, volumeContainsX0);
}

/**
 * @brief
 *
 * @param projections The b vector to invert.
 * @param volume Allocated memory to store x. Might contain the initial guess.
 *
 * @return
 */
int BaseReconstructor::initializeVectors(float* projections,
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

int BaseReconstructor::allocateXBuffers(uint32_t xBufferCount)
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

int BaseReconstructor::allocateBBuffers(uint32_t bBufferCount)
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

int BaseReconstructor::allocateTmpXBuffers(uint32_t xBufferCount)
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

int BaseReconstructor::allocateTmpBBuffers(uint32_t bBufferCount)
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

std::shared_ptr<cl::Buffer> BaseReconstructor::getXBuffer(uint32_t i)
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

std::shared_ptr<cl::Buffer> BaseReconstructor::getBBuffer(uint32_t i)
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

std::shared_ptr<cl::Buffer> BaseReconstructor::getTmpXBuffer(uint32_t i)
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

std::shared_ptr<cl::Buffer> BaseReconstructor::getTmpBBuffer(uint32_t i)
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
int BaseReconstructor::backproject(cl::Buffer& B,
                                   cl::Buffer& X,
                                   uint32_t initialProjectionIndex,
                                   uint32_t projectionIncrement)
{
    Q[0]->enqueueFillBuffer<cl_float>(X, FLOATZERO, 0, XDIM * sizeof(float));
    uint32_t frameSize = rp->pFrameSize;
    algFLOATvector_copy(B, *tmp_b_buf, BDIM);
    // cl::EnqueueArgs eargs(*Q[0], cl::NDRange(vdimz, vdimy, vdimx));
    // cl::EnqueueArgs eargs(*Q[0], cl::NDRange(vdimz, vdimy, vdimx), localRangeBackprojection);
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(vdimx, vdimy, vdimz), backprojectorLocalNDRange);
    cl::EnqueueArgs eargs2(*Q[0], cl::NDRange(pdimx, pdimy));
    cl_double16 CM;
    cl_double16 ICM;
    cl_double3 SOURCEPOSITION, NORMALTODETECTOR;
    cl_double2 NORMALPROJECTION;
    cl_double2 VIRTUALPIXELSIZES;
    double VIRTUALDETECTORDISTANCE = 1.0;
    std::shared_ptr<CameraI> P;
    std::array<double, 2> focalLength;
    float scalingFactor;
    unsigned int offset;
    for(std::size_t i = initialProjectionIndex; i < pdimz; i += projectionIncrement)
    {
        P = cameraVector[i];
        focalLength = P->focalLength();
        // Kernel parameters
        scalingFactor = focalLength[0] * focalLength[1];
        P->projectionMatrixAsVector12((double*)&CM);
        P->inverseProjectionMatrixAsVector16((double*)&ICM);
        P->normalToDetector((double*)&NORMALTODETECTOR);
        P->principalRayProjection((double*)&NORMALPROJECTION);
        P->sourcePosition((double*)&SOURCEPOSITION);
        VIRTUALPIXELSIZES = { 1.0 / focalLength[0], 1.0 / focalLength[1] };
        offset = i * frameSize;
        if(useSidonProjector)
        {
            (*FLOATsidon_backproject)(eargs2, X, *tmp_b_buf, offset, ICM, SOURCEPOSITION,
                                      NORMALTODETECTOR, vdims, voxelSizes, volumeCenter, pdims,
                                      FLOATONE, pixelGranularity);
        } else if(useTTProjector)
        {
            (*FLOATta3_backproject)(eargs, X, *tmp_b_buf, offset, CM, SOURCEPOSITION,
                                    NORMALTODETECTOR, vdims, voxelSizes, volumeCenter, pdims,
                                    FLOATONE);
        } else
        {
            if(useCVPExactProjectionsScaling)
            {
                (*FLOATrescale_projections_exact)(eargs2, *tmp_b_buf, offset, pdims_uint,
                                                  NORMALPROJECTION, VIRTUALPIXELSIZES,
                                                  VIRTUALDETECTORDISTANCE);
            } else
            {
                (*FLOATrescale_projections_cos)(eargs2, *tmp_b_buf, offset, ICM, SOURCEPOSITION,
                                                NORMALTODETECTOR, pdims_uint, scalingFactor);
            }
            (*FLOATcutting_voxel_backproject)(eargs, X, *tmp_b_buf, offset, CM, SOURCEPOSITION,
                                              NORMALTODETECTOR, vdims, voxelSizes, volumeCenter,
                                              pdims, FLOATONE);
        }
    }
    return 0;
}

/**
 * @param initialProjectionIndex For OS SART 0 by default
 * @param projectionIncrement For OS SART 1 by default
 *
 */
int BaseReconstructor::backproject_minmax(cl::Buffer& B,
                                          cl::Buffer& X,
                                          uint32_t initialProjectionIndex,
                                          uint32_t projectionIncrement)
{
    Q[0]->enqueueFillBuffer<cl_float>(X, std::numeric_limits<float>::infinity(), 0,
                                      XDIM * sizeof(float));
    unsigned int frameSize = rp->pFrameSize;
    algFLOATvector_copy(B, *tmp_b_buf, BDIM);
    // cl::EnqueueArgs eargs(*Q[0], cl::NDRange(vdimz, vdimy, vdimx));
    cl::NDRange voxelRange(vdimx, vdimy, vdimz);
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(vdimx, vdimy, vdimz), backprojectorLocalNDRange);
    cl::EnqueueArgs eargs2(*Q[0], cl::NDRange(pdimx, pdimy));
    cl_double16 CM;
    cl_double16 ICM;
    cl_double3 SOURCEPOSITION, NORMALTODETECTOR;
    cl_double2 NORMALPROJECTION;
    cl_double2 VIRTUALPIXELSIZES;
    double VIRTUALDETECTORDISTANCE = 1.0;
    std::shared_ptr<CameraI> P;
    std::array<double, 2> focalLength;
    float scalingFactor;
    unsigned int offset;
    for(std::size_t i = initialProjectionIndex; i < pdimz; i += projectionIncrement)
    {
        P = cameraVector[i];
        focalLength = P->focalLength();
        // Kernel parameters
        scalingFactor = focalLength[0] * focalLength[1];
        P->projectionMatrixAsVector12((double*)&CM);
        P->inverseProjectionMatrixAsVector16((double*)&ICM);
        P->normalToDetector((double*)&NORMALTODETECTOR);
        P->principalRayProjection((double*)&NORMALPROJECTION);
        P->sourcePosition((double*)&SOURCEPOSITION);
        VIRTUALPIXELSIZES = { 1.0 / focalLength[0], 1.0 / focalLength[1] };
        offset = i * frameSize;
        if(useSidonProjector)
        {
            throw std::runtime_error("Minmax backprojector not implemented for Sidon projector.");
        } else if(useTTProjector)
        {
            throw std::runtime_error("Minmax backprojector not implemented for TT projector.");
        } else
        {
            if(useCVPExactProjectionsScaling)
            {
                (*FLOATrescale_projections_exact)(eargs2, *tmp_b_buf, offset, pdims_uint,
                                                  NORMALPROJECTION, VIRTUALPIXELSIZES,
                                                  VIRTUALDETECTORDISTANCE);
            } else
            {
                (*FLOATrescale_projections_cos)(eargs2, *tmp_b_buf, offset, ICM, SOURCEPOSITION,
                                                NORMALTODETECTOR, pdims_uint, scalingFactor);
            }
            algFLOATcutting_voxel_minmaxbackproject(X, *tmp_b_buf, offset, CM, SOURCEPOSITION,
                                                    NORMALTODETECTOR, vdims, voxelSizes,
                                                    volumeCenter, pdims, FLOATONE, voxelRange);
        }
        // TODO: Don't know if that is neccesary in current implementation
        algFLOATvector_zero_infinite_values(X, XDIM);
    }
    return 0;
}

/**
 * @param initialProjectionIndex For OS SART, 0 by default
 * @param projectionIncrement For OS SART, 1 by default
 *
 */
int BaseReconstructor::project(cl::Buffer& X,
                               cl::Buffer& B,
                               uint32_t initialProjectionIndex,
                               uint32_t projectionIncrement)
{
    Q[0]->enqueueFillBuffer<cl_float>(B, FLOATZERO, 0, BDIM * sizeof(float));
    unsigned int frameSize = rp->pFrameSize;
    // cl::EnqueueArgs eargs(*Q[0], cl::NDRange(vdimz, vdimy, vdimx));
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(vdimz, vdimy, vdimx), projectorLocalNDRange);
    cl::NDRange barrierGlobalRange = cl::NDRange(vdimx, vdimy, vdimz);
    cl::NDRange barrierLocalRange = projectorLocalNDRangeBarrier;
    cl::EnqueueArgs eargs2(*Q[0], cl::NDRange(pdimx, pdimy));
    cl_double16 CM;
    cl_double16 ICM;
    cl_double3 SOURCEPOSITION, NORMALTODETECTOR;
    cl_double2 NORMALPROJECTION;
    cl_double2 VIRTUALPIXELSIZES;
    double VIRTUALDETECTORDISTANCE = 1.0;
    std::shared_ptr<CameraI> P;
    std::array<double, 2> focalLength;
    float scalingFactor;
    unsigned int offset;
    for(std::size_t i = initialProjectionIndex; i < pdimz; i += projectionIncrement)
    {
        P = cameraVector[i];
        focalLength = P->focalLength();
        // Kernel parameters
        scalingFactor = focalLength[0] * focalLength[1];
        P->projectionMatrixAsVector12((double*)&CM);
        P->inverseProjectionMatrixAsVector16((double*)&ICM);
        P->normalToDetector((double*)&NORMALTODETECTOR);
        P->principalRayProjection((double*)&NORMALPROJECTION);
        P->sourcePosition((double*)&SOURCEPOSITION);
        VIRTUALPIXELSIZES = { 1.0 / focalLength[0], 1.0 / focalLength[1] };
        offset = i * frameSize;
        if(useSidonProjector)
        {
            (*FLOATsidon_project)(eargs2, X, B, offset, ICM, SOURCEPOSITION, NORMALTODETECTOR,
                                  vdims, voxelSizes, volumeCenter, pdims, FLOATONE,
                                  pixelGranularity);
        } else if(useTTProjector)
        {
            (*FLOATta3_project)(eargs, X, B, offset, CM, SOURCEPOSITION, NORMALTODETECTOR, vdims,
                                voxelSizes, volumeCenter, pdims, FLOATONE);
        } else
        {
            if(useBarrierImplementation)
            {
                algFLOATcutting_voxel_project_barrier(
                    X, B, offset, CM, SOURCEPOSITION, NORMALTODETECTOR, vdims, voxelSizes,
                    volumeCenter, pdims, FLOATONE, this->LOCALARRAYSIZE, barrierGlobalRange,
                    barrierLocalRange, false);
            } else
            {
                (*FLOATcutting_voxel_project)(eargs, X, B, offset, CM, SOURCEPOSITION,
                                              NORMALTODETECTOR, vdims, voxelSizes, volumeCenter,
                                              pdims, FLOATONE);
            }
            if(useCVPExactProjectionsScaling)
            {
                (*FLOATrescale_projections_exact)(eargs2, B, offset, pdims_uint, NORMALPROJECTION,
                                                  VIRTUALPIXELSIZES, VIRTUALDETECTORDISTANCE);
            } else
            {
                (*FLOATrescale_projections_cos)(eargs2, B, offset, ICM, SOURCEPOSITION,
                                                NORMALTODETECTOR, pdims_uint, scalingFactor);
            }
        }
    }
    return 0;
}

std::vector<std::shared_ptr<CameraI>>
BaseReconstructor::encodeProjectionMatrices(std::shared_ptr<io::DenProjectionMatrixReader> pm)
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
std::vector<float> BaseReconstructor::computeScalingFactors()
{
    std::vector<float> scalingFactors;
    std::array<double, 2> focalLength;
    for(std::size_t i = 0; i != pdimz; i++)
    {
        focalLength = cameraVector[i]->focalLength();
        scalingFactors.emplace_back(focalLength[0] * focalLength[1]);
    }
    return scalingFactors;
}

void BaseReconstructor::writeVolume(cl::Buffer& X, std::string path)
{
    bufferIntoArray(X, x, XDIM);
    bool arrayxmajor = true;
    bool outxmajor = true;
    io::DenFileInfo::create3DDenFileFromArray(x, arrayxmajor, path, io::DenSupportedType::FLOAT32,
                                              vdimx, vdimy, vdimz, outxmajor);
}

void BaseReconstructor::writeProjections(cl::Buffer& B, std::string path)
{
    bufferIntoArray(B, b, BDIM);
    bool arrayxmajor = false;
    bool outxmajor = true;
    io::DenFileInfo::create3DDenFileFromArray(b, arrayxmajor, path, io::DenSupportedType::FLOAT32,
                                              pdimx, pdimy, pdimz, outxmajor);
}

void BaseReconstructor::setTimestamp(bool finishCommandQueue)
{
    if(finishCommandQueue)
    {
        Q[0]->finish();
    }
    timestamp = std::chrono::steady_clock::now();
}
std::chrono::milliseconds BaseReconstructor::millisecondsFromTimestamp(bool setNewTimestamp)
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
BaseReconstructor::printTime(std::string msg, bool finishCommandQueue, bool setNewTimestamp)
{
    if(finishCommandQueue)
    {
        Q[0]->finish();
    }
    auto duration = millisecondsFromTimestamp(setNewTimestamp);
    return io::xprintf("%s: %0.2fs", msg.c_str(), duration.count() / 1000.0);
}

void BaseReconstructor::reportTime(std::string msg, bool finishCommandQueue, bool setNewTimestamp)
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

void BaseReconstructor::setReportingParameters(bool verbose,
                                               uint32_t reportKthIteration,
                                               std::string intermediatePrefix)
{
    this->verbose = verbose;
    this->reportKthIteration = reportKthIteration;
    this->intermediatePrefix = intermediatePrefix;
}

double BaseReconstructor::adjointProductTest()
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

cl::NDRange BaseReconstructor::guessProjectionLocalNDRange(bool barrierCalls)
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
        if(vdimz % 4 == 0 && vdimy % 64 == 0 && workGroupSize >= 256)
        {
            projectorLocalNDRange = cl::NDRange(4, 64, 1); // 23.23 RELAXED
        } else
        {
            projectorLocalNDRange = cl::NullRange;
        }
        /*
                    // ZYX
                    localRangeProjection = cl::NDRange(4, 64, 1); // 23.23 RELAXED
                    localRangeProjection = cl::NDRange(); // 27.58 RELAXED
                    localRangeProjection = cl::NDRange(256, 1, 1); // 31.4 RELAXED
                    localRangeProjection = cl::NDRange(1, 256, 1); // 42.27 RELAXED
                    localRangeProjection = cl::NDRange(1, 1, 256); // 42.52 RELAXED
                    localRangeProjection = cl::NDRange(128, 2, 1); // 27.52 RELAXED
                    localRangeProjection = cl::NDRange(128, 1, 2); // 38.1 RELAXED
                    localRangeProjection = cl::NDRange(2, 128, 1); // 24.53 RELAXED
                    localRangeProjection = cl::NDRange(1, 128, 2); // 30.5 RELAXED
                    localRangeProjection = cl::NDRange(1, 2, 128); // 36.17 RELAXED
                    localRangeProjection = cl::NDRange(2, 1, 128); // 26.57 RELAXED
                    localRangeProjection = cl::NDRange(64, 4, 1); // 33.59 RELAXED
                    localRangeProjection = cl::NDRange(64, 1, 4); // 30.79 RELAXED
                    localRangeProjection = cl::NDRange(4, 64, 1); // 23.29 RELAXED
                    localRangeProjection = cl::NDRange(1, 64, 4); // 31.22 RELAXED
                    localRangeProjection = cl::NDRange(4, 1, 64); // 25.24 RELAXED
                    localRangeProjection = cl::NDRange(1, 4, 64); // 43.15 RELAXED
                    localRangeProjection = cl::NDRange(64, 2, 2); // 42.27 RELAXED
                    localRangeProjection = cl::NDRange(2, 64, 2); // 26.31 RELAXED
                    localRangeProjection = cl::NDRange(2, 2, 64); // 30.83 RELAXED
                    localRangeProjection = cl::NDRange(1, 16, 16); // 42.28 RELAXED
                    localRangeProjection = cl::NDRange(32, 4, 2); // 43.86 RELAXED
                    localRangeProjection = cl::NDRange(32, 2, 4); // 31.32 RELAXED
                    localRangeProjection = cl::NDRange(4, 32, 2); // 24.13 RELAXED
                    localRangeProjection = cl::NDRange(2, 32, 4); // 25.48 RELAXED
                    localRangeProjection = cl::NDRange(2, 4, 32); // 34.53 RELAXED
                    localRangeProjection = cl::NDRange(4, 2, 32); // 27.54 RELAXED
                    localRangeProjection = cl::NDRange(32, 8, 1); // 36.83 RELAXED
                    localRangeProjection = cl::NDRange(32, 1, 8); // 26.3 RELAXED
                    localRangeProjection = cl::NDRange(8, 32, 1); // 25.82 RELAXED
                    localRangeProjection = cl::NDRange(1, 32, 8); // 36.59 RELAXED
                    localRangeProjection = cl::NDRange(1, 8, 32); // 47.2 RELAXED
                    localRangeProjection = cl::NDRange(8, 1, 32); // 25.1 RELAXED
                    localRangeProjection = cl::NDRange(16, 16, 1); // 31.18 RELAXED
                    localRangeProjection = cl::NDRange(16, 1, 16); // 26.12 RELAXED
                    localRangeProjection = cl::NDRange(1, 16, 16); // 42.28 RELAXED
                    localRangeProjection = cl::NDRange(16, 8, 2); // 40.83 RELAXED
                    localRangeProjection = cl::NDRange(16, 2, 8); // 27 RELAXED
                    localRangeProjection = cl::NDRange(8, 16, 2); // 29.07 RELAXED
                    localRangeProjection = cl::NDRange(2, 16, 8); // 29.98 RELAXED
                    localRangeProjection = cl::NDRange(8, 2, 16); // 26.1 RELAXED
                    localRangeProjection = cl::NDRange(2, 8, 16); // 33.99 RELAXED
                    localRangeProjection = cl::NDRange(16, 4, 4); // 30.98 RELAXED
                    localRangeProjection = cl::NDRange(4, 16, 4); // 25.22 RELAXED
                    localRangeProjection = cl::NDRange(4, 4, 16); // 28.43 RELAXED
                    localRangeProjection = cl::NDRange(8, 8, 4); // 26.01 RELAXED
                    localRangeProjection = cl::NDRange(8, 4, 8); // 25.94 RELAXED
                    localRangeProjection = cl::NDRange(8, 8, 4); // 25.99 RELAXED
                    localRangeProjection = cl::NDRange(128, 1, 1); // 31.41 RELAXED
                    localRangeProjection = cl::NDRange(1, 128, 1); // 29.77 RELAXED
                    localRangeProjection = cl::NDRange(1, 1, 128); // 32.65 RELAXED
                    localRangeProjection = cl::NDRange(64, 2, 1); // 36.02 RELAXED
                    localRangeProjection = cl::NDRange(64, 1, 2); // 48.74 RELAXED
                    localRangeProjection = cl::NDRange(1, 64, 2); // 31.02 RELAXED
                    localRangeProjection = cl::NDRange(2, 64, 1); // 25.5 RELAXED
                    localRangeProjection = cl::NDRange(2, 1, 64); // 28.37 RELAXED
                    localRangeProjection = cl::NDRange(1, 2, 64); // 37.47 RELAXED
                    localRangeProjection = cl::NDRange(32, 4, 1); // 70.6 RELAXED
                    localRangeProjection = cl::NDRange(32, 1, 4); // 36.1 RELAXED
                    localRangeProjection = cl::NDRange(1, 32, 4); // 33.84 RELAXED
                    localRangeProjection = cl::NDRange(4, 32, 1); // 27.14 RELAXED
                    localRangeProjection = cl::NDRange(4, 1, 32); // 26.39 RELAXED
                    localRangeProjection = cl::NDRange(1, 4, 32); // 42.66 RELAXED
                    localRangeProjection = cl::NDRange(32, 2, 2); // 52.77 RELAXED
                    localRangeProjection = cl::NDRange(2, 32, 2); // 25.63 RELAXED
                    localRangeProjection = cl::NDRange(2, 2, 32); // 31.02 RELAXED
                    localRangeProjection = cl::NDRange(16, 8, 1); // 62.91 RELAXED
                    localRangeProjection = cl::NDRange(16, 1, 8); // 30.92 RELAXED
                    localRangeProjection = cl::NDRange(8, 16, 1); // 35.56 RELAXED
                    localRangeProjection = cl::NDRange(1, 16, 8); // 41.01 RELAXED
                    localRangeProjection = cl::NDRange(1, 8, 16); // 44.49 RELAXED
                    localRangeProjection = cl::NDRange(8, 1, 16); // 27.51 RELAXED
                    localRangeProjection = cl::NDRange(16, 4, 2); // 52.41 RELAXED
                    localRangeProjection = cl::NDRange(16, 2, 4); // 35.46 RELAXED
                    localRangeProjection = cl::NDRange(2, 16, 4); // 28.74 RELAXED
                    localRangeProjection = cl::NDRange(4, 16, 2); // 38.59 RELAXED
                    localRangeProjection = cl::NDRange(2, 4, 16); // 33.57 RELAXED
                    localRangeProjection = cl::NDRange(4, 2, 16); // 28.13 RELAXED
                    localRangeProjection = cl::NDRange(8, 8, 2); // 42.77 RELAXED
                    localRangeProjection = cl::NDRange(8, 2, 8); // 29.21 RELAXED
                    localRangeProjection = cl::NDRange(2, 8, 8); // 32.44 RELAXED
                    localRangeProjection = cl::NDRange(8, 4, 4); // 33.73 RELAXED
                    localRangeProjection = cl::NDRange(4, 8, 4); // 28.29 RELAXED
                    localRangeProjection = cl::NDRange(4, 4, 8); // 28.96 RELAXED
                    localRangeProjection = cl::NDRange(64, 1, 1); // 73.33 RELAXED
                    localRangeProjection = cl::NDRange(1, 64, 1); // 30.72 RELAXED
                    localRangeProjection = cl::NDRange(1, 1, 64); // 34.66 RELAXED
                    localRangeProjection = cl::NDRange(32, 2, 1); // 83.75 RELAXED
                    localRangeProjection = cl::NDRange(32, 1, 2); // 60.47 RELAXED
                    localRangeProjection = cl::NDRange(1, 32, 2); // 33.54 RELAXED
                    localRangeProjection = cl::NDRange(2, 32, 1); // 28.32 RELAXED
                    localRangeProjection = cl::NDRange(1, 2, 32); // 37.56 RELAXED
                    localRangeProjection = cl::NDRange(2, 1, 32); // 29.73 RELAXED
                    localRangeProjection = cl::NDRange(16, 4, 1); // 88.21 RELAXED
                    localRangeProjection = cl::NDRange(16, 1, 4); // 43.81 RELAXED
                    localRangeProjection = cl::NDRange(4, 16, 1); // 45.73 RELAXED
                    localRangeProjection = cl::NDRange(1, 16, 4); // 38.55 RELAXED
                    localRangeProjection = cl::NDRange(4, 1, 16); // 30.69 RELAXED
                    localRangeProjection = cl::NDRange(1, 4, 16); // 41.65 RELAXED
                    localRangeProjection = cl::NDRange(16, 2, 2); // 61.03 RELAXED
                    localRangeProjection = cl::NDRange(2, 16, 2); // 40.26 RELAXED
                    localRangeProjection = cl::NDRange(2, 2, 16); // 32.01 RELAXED
                    localRangeProjection = cl::NDRange(8, 8, 1); // 84.33 RELAXED
                    localRangeProjection = cl::NDRange(8, 1, 8); // 35.4 RELAXED
                    localRangeProjection = cl::NDRange(1, 8, 8); // 42.66 RELAXED
                    localRangeProjection = cl::NDRange(8, 4, 2); // 57.54 RELAXED
                    localRangeProjection = cl::NDRange(8, 2, 4); // 40.99 RELAXED
                    localRangeProjection = cl::NDRange(4, 8, 2); // 41.55 RELAXED
                    localRangeProjection = cl::NDRange(2, 8, 4); // 34.14 RELAXED
                    localRangeProjection = cl::NDRange(4, 2, 8); // 33.25 RELAXED
                    localRangeProjection = cl::NDRange(2, 4, 8); // 33.94 RELAXED
                    localRangeProjection = cl::NDRange(4, 4, 4); // 37.98 RELAXED
                    localRangeProjection = cl::NDRange(32, 1, 1); // 115.24 RELAXED
                    localRangeProjection = cl::NDRange(1, 32, 1); // 33.92 RELAXED
                    localRangeProjection = cl::NDRange(1, 1, 32); // 39.21 RELAXED
                    localRangeProjection = cl::NDRange(16, 2, 1); // 114.62 RELAXED
                    localRangeProjection = cl::NDRange(16, 1, 2); // 77.93 RELAXED
                    localRangeProjection = cl::NDRange(2, 16, 1); // 75.46 RELAXED
                    localRangeProjection = cl::NDRange(1, 16, 2); // 45.32 RELAXED
                    localRangeProjection = cl::NDRange(2, 1, 16); // 43.31 RELAXED
                    localRangeProjection = cl::NDRange(1, 2, 16); // 41.63 RELAXED
                    localRangeProjection = cl::NDRange(8, 4, 1); // 110.89 RELAXED
                    localRangeProjection = cl::NDRange(8, 1, 4); // 62.35 RELAXED
                    localRangeProjection = cl::NDRange(4, 8, 1); // 81.61 RELAXED
                    localRangeProjection = cl::NDRange(1, 8, 4); // 42.95 RELAXED
                    localRangeProjection = cl::NDRange(4, 1, 8); // 51.45 RELAXED
                    localRangeProjection = cl::NDRange(1, 4, 8); // 43.01 RELAXED
                    localRangeProjection = cl::NDRange(8, 2, 2); // 73.64 RELAXED
                    localRangeProjection = cl::NDRange(2, 8, 2); // 51.65 RELAXED
                    localRangeProjection = cl::NDRange(2, 2, 8); // 45.86 RELAXED
                    localRangeProjection = cl::NDRange(4, 4, 2); // 68.42 RELAXED
                    localRangeProjection = cl::NDRange(4, 2, 4); // 56.71 RELAXED
                    localRangeProjection = cl::NDRange(2, 4, 4); // 50.84 RELAXED
                    localRangeProjection = cl::NDRange(16, 1, 1); // 143.31 RELAXED
                    localRangeProjection = cl::NDRange(1, 16, 1); // 82.41 RELAXED
                    localRangeProjection = cl::NDRange(1, 1, 16); // 63.43 RELAXED
                    localRangeProjection = cl::NDRange(8, 2, 1); // 137.76 RELAXED
                    localRangeProjection = cl::NDRange(8, 1, 2); // 114.3 RELAXED
                    localRangeProjection = cl::NDRange(2, 8, 1); // 90.78 RELAXED
                    localRangeProjection = cl::NDRange(1, 8, 2); // 71.05 RELAXED
                    localRangeProjection = cl::NDRange(2, 1, 8); // 77.34 RELAXED
                    localRangeProjection = cl::NDRange(1, 2, 8); // 67.66 RELAXED
                    localRangeProjection = cl::NDRange(4, 4, 1); // 132.09 RELAXED
                    localRangeProjection = cl::NDRange(4, 1, 4); // 93.69 RELAXED
                    localRangeProjection = cl::NDRange(1, 4, 4); // 71.06 RELAXED
                    localRangeProjection = cl::NDRange(4, 2, 2); // 105.82 RELAXED
                    localRangeProjection = cl::NDRange(2, 4, 2); // 94.46 RELAXED
                    localRangeProjection = cl::NDRange(2, 2, 4); // 82.65 RELAXED
                    localRangeProjection = cl::NDRange(8, 1, 1); // 214.05 RELAXED
                    localRangeProjection = cl::NDRange(1, 8, 1); // 127.61 RELAXED
                    localRangeProjection = cl::NDRange(1, 1, 8); // 115.48 RELAXED
                    localRangeProjection = cl::NDRange(4, 2, 1); // 196.41 RELAXED
                    localRangeProjection = cl::NDRange(4, 1, 2); // 171.96 RELAXED
                    localRangeProjection = cl::NDRange(2, 4, 1); // 171.29 RELAXED
                    localRangeProjection = cl::NDRange(1, 4, 2); // 126.08 RELAXED
                    localRangeProjection = cl::NDRange(2, 1, 4); // 142.78 RELAXED
                    localRangeProjection = cl::NDRange(1, 2, 4); // 121.82 RELAXED
                    localRangeProjection = cl::NDRange(2, 2, 2); // 152.92 RELAXED
                    localRangeProjection = cl::NDRange(4, 1, 1); // 313.17 RELAXED
                    localRangeProjection = cl::NDRange(1, 4, 1); // 224.94 RELAXED
                    localRangeProjection = cl::NDRange(1, 1, 4); // 211.77 RELAXED
                    localRangeProjection = cl::NDRange(2, 2, 1); // 272.84 RELAXED
                    localRangeProjection = cl::NDRange(2, 1, 2); // 260.53 RELAXED
                    localRangeProjection = cl::NDRange(1, 2, 2); // 219.3 RELAXED
                    localRangeProjection = cl::NDRange(2, 1, 1); // 476.79 RELAXED
                    localRangeProjection = cl::NDRange(1, 2, 1); // 385.66 RELAXED
                    localRangeProjection = cl::NDRange(1, 1, 2); // 381.26 RELAXED
                    localRangeProjection = cl::NDRange(1, 1, 1); // 697.49 RELAXED
        */
    }
    return projectorLocalNDRange;
}

cl::NDRange BaseReconstructor::guessBackprojectorLocalNDRange()
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
    /*
                // ZYX
                localRangeBackprojection = cl::NDRange(1, 8, 8);
                // Natural XYZ backprojection ordering
                localRangeBackprojection = cl::NDRange(); // 5.55 RELAXED
                localRangeBackprojection = cl::NDRange(256, 1, 1); // 5.55 RELAXED
                localRangeBackprojection = cl::NDRange(1, 256, 1); // 5.51 RELAXED
                localRangeBackprojection = cl::NDRange(1, 1, 256); // 5.55 RELAXED
                localRangeBackprojection = cl::NDRange(128, 2, 1); // 5.16 RELAXED
                localRangeBackprojection = cl::NDRange(128, 1, 2); // 5.18 RELAXED
                localRangeBackprojection = cl::NDRange(2, 128, 1); // 5.40 RELAXED
                localRangeBackprojection = cl::NDRange(1, 128, 2); // 7.18 RELAXED
                localRangeBackprojection = cl::NDRange(1, 2, 128); // 12.88 RELAXED
                localRangeBackprojection = cl::NDRange(2, 1, 128); // 13.99 RELAXED
                localRangeBackprojection = cl::NDRange(64, 4, 1); // 4.62 RELAXED
                localRangeBackprojection = cl::NDRange(64, 1, 4); // 4.78 RELAXED
                localRangeBackprojection = cl::NDRange(1, 64, 4); // 6.83 RELAXED
                localRangeBackprojection = cl::NDRange(4, 64, 1); // 4.29 RELAXED
                localRangeBackprojection = cl::NDRange(4, 1, 64); // 7.73 RELAXED
                localRangeBackprojection = cl::NDRange(1, 4, 64); // 8.71 RELAXED
                localRangeBackprojection = cl::NDRange(64, 2, 2); // 4.69 RELAXED
                localRangeBackprojection = cl::NDRange(2, 64, 2); // 5.02 RELAXED
                localRangeBackprojection = cl::NDRange(2, 2, 64); // 7.54 RELAXED
                localRangeBackprojection = cl::NDRange(8, 8, 4); // 4.30 RELAXED
                localRangeBackprojection = cl::NDRange(16, 16, 1); // 4.28 RELAXED
                localRangeBackprojection = cl::NDRange(32, 4, 2); // 4.46 RELAXED
                localRangeBackprojection = cl::NDRange(32, 2, 4); // 4.50 RELAXED
                localRangeBackprojection = cl::NDRange(4, 32, 2); // 4.26 RELAXED
                localRangeBackprojection = cl::NDRange(2, 32, 4); // 5.03 RELAXED
                localRangeBackprojection = cl::NDRange(2, 4, 32); // 6.28 RELAXED
                localRangeBackprojection = cl::NDRange(4, 2, 32); // 5.83 RELAXED
                localRangeBackprojection = cl::NDRange(32, 8, 1); // 4.39 RELAXED
                localRangeBackprojection = cl::NDRange(32, 1, 8); // 4.81 RELAXED
                localRangeBackprojection = cl::NDRange(8, 32, 1); // 4.17 RELAXED
                localRangeBackprojection = cl::NDRange(1, 32, 8); // 6.78 RELAXED
                localRangeBackprojection = cl::NDRange(1, 8, 32); // 7.46 RELAXED
                localRangeBackprojection = cl::NDRange(8, 1, 32); // 6.15 RELAXED
                localRangeBackprojection = cl::NDRange(16, 16, 1); // 4.28 RELAXED
                localRangeBackprojection = cl::NDRange(16, 1, 16); // 5.28 RELAXED
                localRangeBackprojection = cl::NDRange(1, 16, 16); // 6.92 RELAXED
                localRangeBackprojection = cl::NDRange(16, 8, 2); // 4.31 RELAXED
                localRangeBackprojection = cl::NDRange(16, 2, 8); // 4.65 RELAXED
                localRangeBackprojection = cl::NDRange(8, 16, 2); // 4.13 RELAXED
                localRangeBackprojection = cl::NDRange(2, 16, 8); // 5.08 RELAXED
                localRangeBackprojection = cl::NDRange(8, 2, 16); // 4.93 RELAXED
                localRangeBackprojection = cl::NDRange(2, 8, 16); // 5.49 RELAXED
                localRangeBackprojection = cl::NDRange(16, 4, 4); // 4.39 RELAXED
                localRangeBackprojection = cl::NDRange(4, 16, 4); // 4.29 RELAXED
                localRangeBackprojection = cl::NDRange(4, 4, 16); // 4.92 RELAXED
                localRangeBackprojection = cl::NDRange(8, 8, 4); // 4.2 RELAXED
                localRangeBackprojection = cl::NDRange(8, 4, 8); // 4.39 RELAXED
                localRangeBackprojection = cl::NDRange(8, 8, 4); // 4.23 RELAXED
                localRangeBackprojection = cl::NDRange(128, 1, 1); // 5.16 RELAXED
                localRangeBackprojection = cl::NDRange(1, 128, 1); // 7.12 RELAXED
                localRangeBackprojection = cl::NDRange(1, 1, 128); // 15.73 RELAXED
                localRangeBackprojection = cl::NDRange(64, 2, 1); // 4.72 RELAXED
                localRangeBackprojection = cl::NDRange(64, 1, 2); // 4.69 RELAXED
                localRangeBackprojection = cl::NDRange(1, 64, 2); // 5.97 RELAXED
                localRangeBackprojection = cl::NDRange(2, 64, 1); // 4.48 RELAXED
                localRangeBackprojection = cl::NDRange(2, 1, 64); // 8.29 RELAXED
                localRangeBackprojection = cl::NDRange(1, 2, 64); // 8.08 RELAXED
                localRangeBackprojection = cl::NDRange(32, 4, 1); // 4.46 RELAXED
                localRangeBackprojection = cl::NDRange(32, 1, 4); // 4.60 RELAXED
                localRangeBackprojection = cl::NDRange(1, 32, 4); // 5.78 RELAXED
                localRangeBackprojection = cl::NDRange(4, 32, 1); // 4.11 RELAXED
                localRangeBackprojection = cl::NDRange(4, 1, 32); // 5.80 RELAXED
                localRangeBackprojection = cl::NDRange(1, 4, 32); // 6.38 RELAXED
                localRangeBackprojection = cl::NDRange(32, 2, 2); // 4.49 RELAXED
                localRangeBackprojection = cl::NDRange(2, 32, 2); // 4.42 RELAXED
                localRangeBackprojection = cl::NDRange(2, 2, 32); // 5.64 RELAXED
                localRangeBackprojection = cl::NDRange(16, 8, 1); // 4.26 RELAXED
                localRangeBackprojection = cl::NDRange(16, 1, 8); // 4.72 RELAXED
                localRangeBackprojection = cl::NDRange(8, 16, 1); // 4.08 RELAXED
                localRangeBackprojection = cl::NDRange(1, 16, 8); // 5.80 RELAXED
                localRangeBackprojection = cl::NDRange(1, 8, 16); // 5.93 RELAXED
                localRangeBackprojection = cl::NDRange(8, 1, 16); // 4.85 RELAXED
                localRangeBackprojection = cl::NDRange(16, 4, 2); // 4.31 RELAXED
                localRangeBackprojection = cl::NDRange(16, 2, 4); // 4.39 RELAXED
                localRangeBackprojection = cl::NDRange(2, 16, 4); // 4.37 RELAXED
                localRangeBackprojection = cl::NDRange(4, 16, 2); // 4.08 RELAXED
                localRangeBackprojection = cl::NDRange(2, 4, 16); // 4.79 RELAXED
                localRangeBackprojection = cl::NDRange(4, 2, 16); // 4.58 RELAXED
                localRangeBackprojection = cl::NDRange(8, 8, 2); // 4.08 RELAXED
                localRangeBackprojection = cl::NDRange(8, 2, 8); // 4.35 RELAXED
                localRangeBackprojection = cl::NDRange(2, 8, 8); // 4.55 RELAXED
                localRangeBackprojection = cl::NDRange(8, 4, 4); // 4.20 RELAXED
                localRangeBackprojection = cl::NDRange(4, 8, 4); // 4.16 RELAXED
                localRangeBackprojection = cl::NDRange(4, 4, 8); // 4.27 RELAXED
                localRangeBackprojection = cl::NDRange(64, 1, 1); // 4.77 RELAXED
                localRangeBackprojection = cl::NDRange(1, 64, 1); // 5.93 RELAXED
                localRangeBackprojection = cl::NDRange(1, 1, 64); // 9.23 RELAXED
                localRangeBackprojection = cl::NDRange(32, 2, 1); // 4.49 RELAXED
                localRangeBackprojection = cl::NDRange(32, 1, 2); // 4.59 RELAXED
                localRangeBackprojection = cl::NDRange(1, 32, 2); // 5.20 RELAXED
                localRangeBackprojection = cl::NDRange(2, 32, 1); // 4.34 RELAXED
                localRangeBackprojection = cl::NDRange(1, 2, 32); // 6.28 RELAXED
                localRangeBackprojection = cl::NDRange(2, 1, 32); // 6.35 RELAXED
                localRangeBackprojection = cl::NDRange(16, 4, 1); // 4.31 RELAXED
                localRangeBackprojection = cl::NDRange(16, 1, 4); // 4.65 RELAXED
                localRangeBackprojection = cl::NDRange(4, 16, 1); // 4.05 RELAXED
                localRangeBackprojection = cl::NDRange(1, 16, 4); // 5.13 RELAXED
                localRangeBackprojection = cl::NDRange(4, 1, 16); // 4.97 RELAXED
                localRangeBackprojection = cl::NDRange(1, 4, 16); // 5.39 RELAXED
                localRangeBackprojection = cl::NDRange(16, 2, 2); // 4.42 RELAXED
                localRangeBackprojection = cl::NDRange(2, 16, 2); // 4.30 RELAXED
                localRangeBackprojection = cl::NDRange(2, 2, 16); // 4.82 RELAXED
                localRangeBackprojection = cl::NDRange(8, 8, 1); // 4.06 RELAXED
                localRangeBackprojection = cl::NDRange(8, 1, 8); // 4.59 RELAXED
                localRangeBackprojection = cl::NDRange(1, 8, 8); // 5.13 RELAXED
                localRangeBackprojection = cl::NDRange(8, 4, 2); // 4.17 RELAXED
                localRangeBackprojection = cl::NDRange(8, 2, 4); // 4.34 RELAXED
                localRangeBackprojection = cl::NDRange(4, 8, 2); // 4.09 RELAXED
                localRangeBackprojection = cl::NDRange(2, 8, 4); // 4.37 RELAXED
                localRangeBackprojection = cl::NDRange(4, 2, 8); // 4.36 RELAXED
                localRangeBackprojection = cl::NDRange(2, 4, 8); // 4.42 RELAXED
                localRangeBackprojection = cl::NDRange(4, 4, 4); // 4.18 RELAXED
                localRangeBackprojection = cl::NDRange(32, 1, 1); // 8.62 RELAXED
                localRangeBackprojection = cl::NDRange(1, 32, 1); // 8.30 RELAXED
                localRangeBackprojection = cl::NDRange(1, 1, 32); // 11.38 RELAXED
                localRangeBackprojection = cl::NDRange(16, 2, 1); // 8.33 RELAXED
                localRangeBackprojection = cl::NDRange(16, 1, 2); // 8.49 RELAXED
                localRangeBackprojection = cl::NDRange(2, 16, 1); // 7.87 RELAXED
                localRangeBackprojection = cl::NDRange(1, 16, 2); // 8.21 RELAXED
                localRangeBackprojection = cl::NDRange(2, 1, 16); // 8.94 RELAXED
                localRangeBackprojection = cl::NDRange(1, 2, 16); // 8.77 RELAXED
                localRangeBackprojection = cl::NDRange(8, 4, 1); // 7.88 RELAXED
                localRangeBackprojection = cl::NDRange(8, 1, 4); // 8.21 RELAXED
                localRangeBackprojection = cl::NDRange(4, 8, 1); // 7.75 RELAXED
                localRangeBackprojection = cl::NDRange(1, 8, 4); // 8.16 RELAXED
                localRangeBackprojection = cl::NDRange(4, 1, 8); // 8.31 RELAXED
                localRangeBackprojection = cl::NDRange(1, 4, 8); // 8.29 RELAXED
                localRangeBackprojection = cl::NDRange(8, 2, 2); // 8.03 RELAXED
                localRangeBackprojection = cl::NDRange(2, 8, 2); // 7.9 RELAXED
                localRangeBackprojection = cl::NDRange(2, 2, 8); // 8.14 RELAXED
                localRangeBackprojection = cl::NDRange(4, 4, 2); // 7.82 RELAXED
                localRangeBackprojection = cl::NDRange(4, 2, 4); // 7.90 RELAXED
                localRangeBackprojection = cl::NDRange(2, 4, 4); // 7.90 RELAXED
                localRangeBackprojection = cl::NDRange(16, 1, 1); // 16.07 RELAXED
                localRangeBackprojection = cl::NDRange(1, 16, 1); // 15.14 RELAXED
                localRangeBackprojection = cl::NDRange(1, 1, 16); // 16.06 RELAXED
                localRangeBackprojection = cl::NDRange(8, 2, 1); // 15.27 RELAXED
                localRangeBackprojection = cl::NDRange(8, 1, 2); // 15.35 RELAXED
                localRangeBackprojection = cl::NDRange(2, 8, 1); // 14.96 RELAXED
                localRangeBackprojection = cl::NDRange(1, 8, 2); // 14.96 RELAXED
                localRangeBackprojection = cl::NDRange(2, 1, 8); // 15.04 RELAXED
                localRangeBackprojection = cl::NDRange(1, 2, 8); // 14.84 RELAXED
                localRangeBackprojection = cl::NDRange(4, 4, 1); // 14.89 RELAXED
                localRangeBackprojection = cl::NDRange(4, 1, 4); // 15 RELAXED
                localRangeBackprojection = cl::NDRange(1, 4, 4); // 14.83 RELAXED
                localRangeBackprojection = cl::NDRange(4, 2, 2); // 14.89 RELAXED
                localRangeBackprojection = cl::NDRange(2, 4, 2); // 14.83 RELAXED
                localRangeBackprojection = cl::NDRange(2, 2, 4); // 14.80 RELAXED
                localRangeBackprojection = cl::NDRange(8, 1, 1); // 29.62 RELAXED
                localRangeBackprojection = cl::NDRange(1, 8, 1); // 28.72 RELAXED
                localRangeBackprojection = cl::NDRange(1, 1, 8); // 27.29 RELAXED
                localRangeBackprojection = cl::NDRange(4, 2, 1); // 28.79 RELAXED
                localRangeBackprojection = cl::NDRange(4, 1, 2); // 28.49 RELAXED
                localRangeBackprojection = cl::NDRange(2, 4, 1); // 28.60 RELAXED
                localRangeBackprojection = cl::NDRange(1, 4, 2); // 28.13 RELAXED
                localRangeBackprojection = cl::NDRange(2, 1, 4); // 27.58 RELAXED
                localRangeBackprojection = cl::NDRange(1, 2, 4); // 27.35 RELAXED
                localRangeBackprojection = cl::NDRange(2, 2, 2); // 28.18 RELAXED
                localRangeBackprojection = cl::NDRange(4, 1, 1); // 55.16 RELAXED
                localRangeBackprojection = cl::NDRange(1, 4, 1); // 54.45 RELAXED
                localRangeBackprojection = cl::NDRange(1, 1, 4); // 50.40 RELAXED
                localRangeBackprojection = cl::NDRange(2, 2, 1); // 54.57 RELAXED
                localRangeBackprojection = cl::NDRange(2, 1, 2); // 52.74 RELAXED
                localRangeBackprojection = cl::NDRange(1, 2, 2); // 52.48 RELAXED
                localRangeBackprojection = cl::NDRange(2, 1, 1); // 102.21 RELAXED
                localRangeBackprojection = cl::NDRange(1, 2, 1); // 101.91 RELAXED
                localRangeBackprojection = cl::NDRange(1, 1, 2); // 96.86 RELAXED
                localRangeBackprojection = cl::NDRange(1, 1, 1); // 187.64 RELAXED
    */
}

} // namespace KCT
