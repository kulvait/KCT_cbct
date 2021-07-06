#include "BaseReconstructor.hpp"

namespace CTL {

void BaseReconstructor::initializeCVPProjector(bool useExactScaling,
                                               bool useBarrierCalls,
                                               uint32_t LOCALARRAYSIZE)
{
    if(!isOpenCLInitialized())
    {
        useCVPProjector = true;
        exactProjectionScaling = useExactScaling;
        this->CVPBarrierImplementation = useBarrierCalls;
        useSidonProjector = false;
        pixelGranularity = { 1, 1 };
        useTTProjector = false;
        CLINCLUDEutils();
        CLINCLUDEinclude();
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
        CLINCLUDEbackprojector_minmax();
        CLINCLUDErescaleProjections();
    } else
    {
        std::string err = "Could not initialize projector when OpenCL was already initialized.";
        LOGE << err;
        throw std::runtime_error(err.c_str());
    }
}

void BaseReconstructor::initializeSidonProjector(uint32_t probesPerEdgeX, uint32_t probesPerEdgeY)
{
    if(!isOpenCLInitialized())
    {
        useSidonProjector = true;
        pixelGranularity = { probesPerEdgeX, probesPerEdgeY };
        useCVPProjector = false;
        exactProjectionScaling = false;
        useTTProjector = false;
        CLINCLUDEutils();
        CLINCLUDEinclude();
        CLINCLUDEprojector_sidon();
        CLINCLUDEbackprojector_sidon();
    } else
    {
        std::string err = "Could not initialize projector when OpenCL was already initialized.";
        LOGE << err;
        throw std::runtime_error(err.c_str());
    }
}

void BaseReconstructor::initializeTTProjector()
{
    if(!isOpenCLInitialized())
    {
        useTTProjector = true;
        useCVPProjector = false;
        exactProjectionScaling = false;
        useSidonProjector = false;
        pixelGranularity = { 1, 1 };
        CLINCLUDEutils();
        CLINCLUDEinclude();
        CLINCLUDEprojector();
        CLINCLUDEbackprojector();
        CLINCLUDEprojector_tt();
        CLINCLUDEbackprojector_tt();
    } else
    {
        std::string err = "Could not initialize projector when OpenCL was already initialized.";
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
        LOGE << err;
        throw std::runtime_error(err);
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
                                k, VN[0], VN[1], VN[2], center[0], center[1], center[2], vectorDotProduct(VN, center));
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
    initializeAlgorithmsBuffers();
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

int BaseReconstructor::B_equals_A_plus_B_offsets(cl::Buffer& from,
                                                 unsigned int from_offset,
                                                 cl::Buffer& to,
                                                 unsigned int to_offset,
                                                 unsigned int size)
{
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(size));
    (*FLOATvector_B_equals_A_plus_B_offsets)(eargs, from, from_offset, to, to_offset).wait();
    return 0;
}

int BaseReconstructor::invertFloatVector(cl::Buffer& X, unsigned int size)
{
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(size));
    (*FLOATvector_invert_except_zero)(eargs, X).wait();
    return 0;
}

int BaseReconstructor::vectorA_multiple_B_equals_C(cl::Buffer& A,
                                                   cl::Buffer& B,
                                                   cl::Buffer& C,
                                                   uint64_t size)
{
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(size));
    (*FLOATvector_C_equals_A_times_B)(eargs, A, B, C).wait();
    return 0;
}

int BaseReconstructor::multiplyVectorsIntoFirstVector(cl::Buffer& A, cl::Buffer& B, uint64_t size)
{
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(size));
    (*FLOATvector_A_equals_A_times_B)(eargs, A, B).wait();
    return 0;
}

int BaseReconstructor::backproject(cl::Buffer& B, cl::Buffer& X)
{
    Q[0]->enqueueFillBuffer<cl_float>(X, FLOATZERO, 0, XDIM * sizeof(float));
    unsigned int frameSize = pdimx * pdimy;
    copyFloatVector(B, *tmp_b_buf, BDIM);
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
    for(std::size_t i = 0; i != pdimz; i++)
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
            if(exactProjectionScaling)
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

int BaseReconstructor::backproject_minmax(cl::Buffer& B, cl::Buffer& X)
{
    Q[0]->enqueueFillBuffer<cl_float>(X, std::numeric_limits<float>::infinity(), 0,
                                      XDIM * sizeof(float));
    unsigned int frameSize = pdimx * pdimy;
    copyFloatVector(B, *tmp_b_buf, BDIM);
    // cl::EnqueueArgs eargs(*Q[0], cl::NDRange(vdimz, vdimy, vdimx));
    cl::NDRange voxelRange(vdimz, vdimy, vdimx);
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(vdimz, vdimy, vdimx), backprojectorLocalNDRange);
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
    for(std::size_t i = 0; i != pdimz; i++)
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
            if(exactProjectionScaling)
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

int BaseReconstructor::project(cl::Buffer& X, cl::Buffer& B)
{
    Q[0]->enqueueFillBuffer<cl_float>(B, FLOATZERO, 0, BDIM * sizeof(float));
    unsigned int frameSize = pdimx * pdimy;
    // cl::EnqueueArgs eargs(*Q[0], cl::NDRange(vdimz, vdimy, vdimx));
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(vdimz, vdimy, vdimx), projectorLocalNDRange);
    cl::NDRange barrierGlobalRange = cl::NDRange(vdimx, vdimy, vdimz);
    std::shared_ptr<cl::NDRange> barrierLocalRange
        = std::make_shared<cl::NDRange>(projectorLocalNDRangeBarrier);
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
    for(std::size_t i = 0; i != pdimz; i++)
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
            if(CVPBarrierImplementation)
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
            if(exactProjectionScaling)
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

int BaseReconstructor::copyFloatVector(cl::Buffer& from, cl::Buffer& to, unsigned int size)
{
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(size));
    (*FLOATvector_copy)(eargs, from, to).wait();
    return 0;
}

int BaseReconstructor::scaleFloatVector(cl::Buffer& v, float f, unsigned int size)
{
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(size));
    (*FLOATvector_scale)(eargs, v, f).wait();
    return 0;
}

int BaseReconstructor::addIntoFirstVectorSecondVectorScaled(cl::Buffer& a,
                                                            cl::Buffer& b,
                                                            float f,
                                                            unsigned int size)
{
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(size));
    (*FLOATvector_A_equals_A_plus_cB)(eargs, a, b, f).wait();
    return 0;
}

int BaseReconstructor::addIntoFirstVectorScaledSecondVector(cl::Buffer& a,
                                                            cl::Buffer& b,
                                                            float f,
                                                            unsigned int size)
{
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(size));
    (*FLOATvector_A_equals_Ac_plus_B)(eargs, a, b, f).wait();
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
    uint16_t buf[3];
    buf[0] = vdimy;
    buf[1] = vdimx;
    buf[2] = vdimz;
    io::createEmptyFile(path, 0, true); // Try if this is faster
    io::appendBytes(path, (uint8_t*)buf, 6);
    Q[0]->enqueueReadBuffer(X, CL_TRUE, 0, sizeof(float) * XDIM, x);
    io::appendBytes(path, (uint8_t*)x, XDIM * sizeof(float));
}

void BaseReconstructor::writeProjections(cl::Buffer& B, std::string path)
{
    uint16_t buf[3];
    buf[0] = pdimy;
    buf[1] = pdimx;
    buf[2] = pdimz;
    io::createEmptyFile(path, 0, true); // Try if this is faster
    io::appendBytes(path, (uint8_t*)buf, 6);
    Q[0]->enqueueReadBuffer(B, CL_TRUE, 0, sizeof(float) * BDIM, b);
    io::appendBytes(path, (uint8_t*)b, BDIM * sizeof(float));
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
                                               std::string progressPrefixPath)
{
    this->verbose = verbose;
    this->reportKthIteration = reportKthIteration;
    this->progressPrefixPath = progressPrefixPath;
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

} // namespace CTL
