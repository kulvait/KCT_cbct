#include "CuttingVoxelProjector.hpp"

namespace KCT {
cl::NDRange CuttingVoxelProjector::guessProjectionLocalNDRange(bool barrierCalls)
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

void CuttingVoxelProjector::initializeCVPProjector(bool useExactScaling,
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
        this->pixelGranularity = { 1, 1 };
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

void CuttingVoxelProjector::initializeSidonProjector(uint32_t probesPerEdgeX,
                                                     uint32_t probesPerEdgeY)
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

void CuttingVoxelProjector::initializeTTProjector()
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

void CuttingVoxelProjector::initializeAllAlgorithms()
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
    } else
    {
        std::string err = "Could not initialize projector when OpenCL was already initialized.";
        LOGE << err;
        throw std::runtime_error(err.c_str());
    }
}
int CuttingVoxelProjector::problemSetup(double voxelSizeX,
                                        double voxelSizeY,
                                        double voxelSizeZ,
                                        double volumeCenterX,
                                        double volumeCenterY,
                                        double volumeCenterZ)
{
    voxelSizes = cl_double3({ voxelSizeX, voxelSizeY, voxelSizeZ });
    volumeCenter = cl_double3({ volumeCenterX, volumeCenterY, volumeCenterZ });
    return 0;
}

int CuttingVoxelProjector::initializeOrUpdateVolumeBuffer(float* volumeArray)
{
    return initializeOrUpdateVolumeBuffer(vdimx, vdimy, vdimz, volumeArray);
}

int CuttingVoxelProjector::initializeOrUpdateVolumeBuffer(uint32_t vdimx,
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

int CuttingVoxelProjector::initializeOrUpdateProjectionBuffer(uint32_t projectionSizeZ,
                                                              float* projectionArray)
{
    return initializeOrUpdateProjectionBuffer(pdimx, pdimy, projectionSizeZ, projectionArray);
}
int CuttingVoxelProjector::initializeOrUpdateProjectionBuffer(float* projectionArray)
{
    return initializeOrUpdateProjectionBuffer(pdimx, pdimy, pdimz, projectionArray);
}
int CuttingVoxelProjector::initializeOrUpdateProjectionBuffer(uint32_t pdimx,
                                                              uint32_t pdimy,
                                                              uint32_t pdimz,
                                                              float* projectionArray)
{
    cl_int err = CL_SUCCESS;
    std::string msg;
    this->pdimx = pdimx;
    this->pdimy = pdimy;
    this->pdimz = pdimz;
    this->totalPixelNum = pdimx * pdimy * pdimz;
    pdims = cl_int2({ int(pdimx), int(pdimy) });
    pdims_uint = cl_uint2({ uint32_t(pdimx), uint32_t(pdimy) });
    if(projectionBuffer != nullptr)
    {
        if(this->totalProjectionBufferSize == sizeof(float) * this->totalPixelNum)
        {
            if(projectionArray != nullptr)
            {
                err = Q[0]->enqueueWriteBuffer(*projectionBuffer, CL_TRUE, 0,
                                               totalProjectionBufferSize, (void*)projectionArray);
            }
        } else
        {
            this->totalProjectionBufferSize = sizeof(float) * this->totalPixelNum;
            if(projectionArray != nullptr)
            {
                projectionBuffer = std::make_shared<cl::Buffer>(
                    *context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, totalProjectionBufferSize,
                    (void*)projectionArray, &err);
            } else
            {
                projectionBuffer = std::make_shared<cl::Buffer>(
                    *context, CL_MEM_READ_WRITE, totalProjectionBufferSize, nullptr, &err);
            }
        }
    } else
    {
        this->totalProjectionBufferSize = sizeof(float) * this->totalPixelNum;
        if(projectionArray != nullptr)
        {
            projectionBuffer = std::make_shared<cl::Buffer>(
                *context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, totalProjectionBufferSize,
                (void*)projectionArray, &err);
        } else
        {
            projectionBuffer = std::make_shared<cl::Buffer>(
                *context, CL_MEM_READ_WRITE, totalProjectionBufferSize, nullptr, &err);
        }
    }
    if(err != CL_SUCCESS)
    {
        msg = io::xprintf("Unsucessful initialization of Projection with error code %d!", err);
        LOGE << msg;
        throw std::runtime_error(msg);
    }
    return 0;
}

int CuttingVoxelProjector::fillVolumeBufferByConstant(float constant)
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

int CuttingVoxelProjector::fillProjectionBufferByConstant(float constant)
{
    std::string msg;
    if(projectionBuffer == nullptr)
    {
        msg = io::xprintf(
            "Projection buffer is not yet initialized, call initializeProjectionBuffer first!");
        LOGE << msg;
        throw std::runtime_error(msg);
    }
    return Q[0]->enqueueFillBuffer<cl_float>(*projectionBuffer, constant, 0,
                                             totalProjectionBufferSize);
}

double CuttingVoxelProjector::normSquare(float* v, uint32_t pdimx, uint32_t pdimy)
{
    size_t vecsize = sizeof(float) * pdimx * pdimy;
    if(tmpBuffer == nullptr || vecsize != tmpBuffer_size)
    {
        tmpBuffer_size = vecsize;
        tmpBuffer = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                 tmpBuffer_size, (void*)v);
    } else
    {
        Q[0]->enqueueWriteBuffer(*tmpBuffer, CL_TRUE, 0, tmpBuffer_size, (void*)v);
    }
    cl::Buffer onedouble(*context, CL_MEM_READ_WRITE, sizeof(double), nullptr);
    double sum;
    uint32_t framesize = pdimx * pdimy;
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(1));
    (*vector_NormSquarePartial)(eargs, *tmpBuffer, onedouble, framesize).wait();
    Q[0]->enqueueReadBuffer(onedouble, CL_TRUE, 0, sizeof(double), &sum);
    return sum;
}

double CuttingVoxelProjector::normSquareDifference(float* v, uint32_t pdimx, uint32_t pdimy)
{
    size_t vecsize = sizeof(float) * pdimx * pdimy;
    if(projectionBuffer == nullptr)
    {
        std::string msg = "Comparing to empty buffer is not possible.";
        LOGE << msg;
        throw new std::runtime_error(msg);
    }
    if(vecsize != totalProjectionBufferSize)
    {
        std::string msg = "Ca not compare buffers of incompatible dimensions.";
        throw new std::runtime_error(msg);
    }

    if(tmpBuffer == nullptr || vecsize != tmpBuffer_size)
    {
        tmpBuffer_size = vecsize;
        tmpBuffer = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                 tmpBuffer_size, (void*)v);
    } else
    {
        Q[0]->enqueueWriteBuffer(*tmpBuffer, CL_TRUE, 0, tmpBuffer_size, (void*)v);
    }
    uint32_t framesize = pdimx * pdimy;
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(framesize));
    float factor = -1.0;
    (*FLOATvector_A_equals_Ac_plus_B)(eargs, *tmpBuffer, *projectionBuffer, factor).wait();
    cl::Buffer onedouble(*context, CL_MEM_READ_WRITE, sizeof(double), nullptr);
    double sum;
    cl::EnqueueArgs ear(*Q[0], cl::NDRange(1));
    (*vector_NormSquarePartial)(ear, *tmpBuffer, onedouble, framesize).wait();
    Q[0]->enqueueReadBuffer(onedouble, CL_TRUE, 0, sizeof(double), &sum);
    return sum;
}

int CuttingVoxelProjector::project(float* projection, std::shared_ptr<matrix::CameraI> pm)
{
    if(useSidonProjector)
    {
        return projectSidon(projection, pm);
    } else if(useTTProjector)
    {
        return projectTA3(projection, pm);
    } else
    {
        if(useCVPExactProjectionsScaling)
        {
            return projectExact(projection, pm);
        } else
        {
            return projectCos(projection, pm);
        }
    }
}

int CuttingVoxelProjector::projectCos(float* projection, std::shared_ptr<matrix::CameraI> P)
{
    initializeOrUpdateProjectionBuffer(pdimx, pdimy, 1);
    fillProjectionBufferByConstant(0.0f);
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(vdimz, vdimy, vdimx));
    cl::EnqueueArgs eargs2(*Q[0], cl::NDRange(pdimx, pdimy));
    cl_double16 CM;
    cl_double16 ICM;
    cl_double3 SOURCEPOSITION, NORMALTODETECTOR;
    cl_double2 NORMALPROJECTION;
    cl_double2 VIRTUALPIXELSIZES;
    std::array<double, 2> focalLength;
    float scalingFactor;
    unsigned int offset;
    focalLength = P->focalLength();
    // Kernel parameters
    scalingFactor = focalLength[0] * focalLength[1];
    P->projectionMatrixAsVector12((double*)&CM);
    P->inverseProjectionMatrixAsVector16((double*)&ICM);
    P->normalToDetector((double*)&NORMALTODETECTOR);
    P->principalRayProjection((double*)&NORMALPROJECTION);
    P->sourcePosition((double*)&SOURCEPOSITION);
    VIRTUALPIXELSIZES = { 1.0 / focalLength[0], 1.0 / focalLength[1] };
    offset = 0;
    (*FLOATcutting_voxel_project)(eargs, *volumeBuffer, *projectionBuffer, offset, CM,
                                  SOURCEPOSITION, NORMALTODETECTOR, vdims, voxelSizes, volumeCenter,
                                  pdims, FLOATONE);
    (*FLOATrescale_projections_cos)(eargs2, *projectionBuffer, offset, ICM, SOURCEPOSITION,
                                    NORMALTODETECTOR, pdims_uint, scalingFactor);
    cl_int err = Q[0]->enqueueReadBuffer(*projectionBuffer, CL_TRUE, 0,
                                         sizeof(float) * pdimx * pdimy, projection);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful writte buffer to the projection variable, code %d!", err);
        return -1;
    }
    return 0;
}

int CuttingVoxelProjector::projectorWithoutScaling(float* projection,
                                                   std::shared_ptr<matrix::CameraI> P)
{
    initializeOrUpdateProjectionBuffer(pdimx, pdimy, 1);
    fillProjectionBufferByConstant(0.0f);
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(vdimz, vdimy, vdimx));
    cl::EnqueueArgs eargs2(*Q[0], cl::NDRange(pdimx, pdimy));
    cl_double16 CM;
    cl_double16 ICM;
    cl_double3 SOURCEPOSITION, NORMALTODETECTOR;
    cl_double2 NORMALPROJECTION;
    cl_double2 VIRTUALPIXELSIZES;
    std::array<double, 2> focalLength;
    unsigned int offset;
    focalLength = P->focalLength();
    // Kernel parameters
    P->projectionMatrixAsVector12((double*)&CM);
    P->inverseProjectionMatrixAsVector16((double*)&ICM);
    P->normalToDetector((double*)&NORMALTODETECTOR);
    P->principalRayProjection((double*)&NORMALPROJECTION);
    P->sourcePosition((double*)&SOURCEPOSITION);
    VIRTUALPIXELSIZES = { 1.0 / focalLength[0], 1.0 / focalLength[1] };
    offset = 0;
    (*FLOATcutting_voxel_project)(eargs, *volumeBuffer, *projectionBuffer, offset, CM,
                                  SOURCEPOSITION, NORMALTODETECTOR, vdims, voxelSizes, volumeCenter,
                                  pdims, FLOATONE);
    cl_int err = Q[0]->enqueueReadBuffer(*projectionBuffer, CL_TRUE, 0,
                                         sizeof(float) * pdimx * pdimy, projection);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful writte buffer to the projection variable, code %d!", err);
        return -1;
    }
    return 0;
}

int CuttingVoxelProjector::projectExact(float* projection, std::shared_ptr<matrix::CameraI> P)

{
    initializeOrUpdateProjectionBuffer(pdimx, pdimy, 1);
    fillProjectionBufferByConstant(0.0f);
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(vdimz, vdimy, vdimx));
    cl::EnqueueArgs eargs2(*Q[0], cl::NDRange(pdimx, pdimy));
    cl::NDRange barrierGlobalRange = cl::NDRange(vdimx, vdimy, vdimz);
    std::shared_ptr<cl::NDRange> barrierLocalRange
        = std::make_shared<cl::NDRange>(projectorLocalNDRangeBarrier);
    cl_double16 CM;
    cl_double16 ICM;
    cl_double3 SOURCEPOSITION, NORMALTODETECTOR;
    cl_double2 NORMALPROJECTION;
    cl_double2 VIRTUALPIXELSIZES;
    double VIRTUALDETECTORDISTANCE = 1.0;
    std::array<double, 2> focalLength;
    unsigned int offset;
    focalLength = P->focalLength();
    // Kernel parameters
    P->projectionMatrixAsVector12((double*)&CM);
    P->inverseProjectionMatrixAsVector16((double*)&ICM);
    P->normalToDetector((double*)&NORMALTODETECTOR);
    P->principalRayProjection((double*)&NORMALPROJECTION);
    P->sourcePosition((double*)&SOURCEPOSITION);
    VIRTUALPIXELSIZES = { 1.0 / focalLength[0], 1.0 / focalLength[1] };
    offset = 0;
    if(useBarrierImplementation)
    {
        algFLOATcutting_voxel_project_barrier(*volumeBuffer, *projectionBuffer, offset, CM,
                                              SOURCEPOSITION, NORMALTODETECTOR, vdims, voxelSizes,
                                              volumeCenter, pdims, FLOATONE, this->LOCALARRAYSIZE,
                                              barrierGlobalRange, barrierLocalRange, false);

    } else
    {
        (*FLOATcutting_voxel_project)(eargs, *volumeBuffer, *projectionBuffer, offset, CM,
                                      SOURCEPOSITION, NORMALTODETECTOR, vdims, voxelSizes,
                                      volumeCenter, pdims, FLOATONE);
    }
    (*FLOATrescale_projections_exact)(eargs2, *projectionBuffer, offset, pdims_uint,
                                      NORMALPROJECTION, VIRTUALPIXELSIZES, VIRTUALDETECTORDISTANCE);
    cl_int err = Q[0]->enqueueReadBuffer(*projectionBuffer, CL_TRUE, 0,
                                         sizeof(float) * pdimx * pdimy, projection);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful writte buffer to the projection variable, code %d!", err);
        return -1;
    }
    return 0;
}

int CuttingVoxelProjector::projectSidon(float* projection, std::shared_ptr<matrix::CameraI> P)
{
    initializeOrUpdateProjectionBuffer(pdimx, pdimy, 1);
    fillProjectionBufferByConstant(0.0f);
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(vdimz, vdimy, vdimx));
    cl::EnqueueArgs eargs2(*Q[0], cl::NDRange(pdimx, pdimy));
    cl_double16 CM;
    cl_double16 ICM;
    cl_double3 SOURCEPOSITION, NORMALTODETECTOR;
    cl_double2 NORMALPROJECTION;
    cl_double2 VIRTUALPIXELSIZES;
    std::array<double, 2> focalLength;
    unsigned int offset;
    focalLength = P->focalLength();
    // Kernel parameters
    P->projectionMatrixAsVector12((double*)&CM);
    P->inverseProjectionMatrixAsVector16((double*)&ICM);
    P->normalToDetector((double*)&NORMALTODETECTOR);
    P->principalRayProjection((double*)&NORMALPROJECTION);
    P->sourcePosition((double*)&SOURCEPOSITION);
    VIRTUALPIXELSIZES = { 1.0 / focalLength[0], 1.0 / focalLength[1] };
    offset = 0;
    (*FLOATsidon_project)(eargs2, *volumeBuffer, *projectionBuffer, offset, ICM, SOURCEPOSITION,
                          NORMALTODETECTOR, vdims, voxelSizes, volumeCenter, pdims, FLOATONE,
                          pixelGranularity);
    cl_int err = Q[0]->enqueueReadBuffer(*projectionBuffer, CL_TRUE, 0,
                                         sizeof(float) * pdimx * pdimy, projection);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful writte buffer to the projection variable, code %d!", err);
        return -1;
    }
    return 0;
}

int CuttingVoxelProjector::projectTA3(float* projection, std::shared_ptr<matrix::CameraI> P)
{
    initializeOrUpdateProjectionBuffer(pdimx, pdimy, 1);
    fillProjectionBufferByConstant(0.0f);
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(vdimz, vdimy, vdimx));
    cl::EnqueueArgs eargs2(*Q[0], cl::NDRange(pdimx, pdimy));
    cl_double16 CM;
    cl_double16 ICM;
    cl_double3 SOURCEPOSITION, NORMALTODETECTOR;
    cl_double2 NORMALPROJECTION;
    cl_double2 VIRTUALPIXELSIZES;
    std::array<double, 2> focalLength;
    unsigned int offset;
    focalLength = P->focalLength();
    // Kernel parameters
    P->projectionMatrixAsVector12((double*)&CM);
    P->inverseProjectionMatrixAsVector16((double*)&ICM);
    P->normalToDetector((double*)&NORMALTODETECTOR);
    P->principalRayProjection((double*)&NORMALPROJECTION);
    P->sourcePosition((double*)&SOURCEPOSITION);
    VIRTUALPIXELSIZES = { 1.0 / focalLength[0], 1.0 / focalLength[1] };
    offset = 0;
    (*FLOATta3_project)(eargs, *volumeBuffer, *projectionBuffer, offset, CM, SOURCEPOSITION,
                        NORMALTODETECTOR, vdims, voxelSizes, volumeCenter, pdims, FLOATONE);
    cl_int err = Q[0]->enqueueReadBuffer(*projectionBuffer, CL_TRUE, 0,
                                         sizeof(float) * pdimx * pdimy, projection);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful writte buffer to the projection variable, code %d!", err);
        return -1;
    }
    return 0;
}

std::vector<cl_double16>
CuttingVoxelProjector::invertProjectionMatrices(std::vector<matrix::ProjectionMatrix> CM)
{
    std::vector<cl_double16> inverseProjectionMatrices;
    for(std::size_t i = 0; i != CM.size(); i++)
    {
        matrix::ProjectionMatrix matrix = CM[i];

        double* P = matrix.getPtr();
        std::array<double, 3> sourcePosition = matrix.sourcePosition();
        KCT::matrix::SquareMatrix CME(4,
                                      { P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9],
                                        P[10], P[11], sourcePosition[0], sourcePosition[1],
                                        sourcePosition[2], 1.0 });
        matrix::LUDoolittleForm lu = matrix::LUDoolittleForm::LUDecomposeDoolittle(CME, 0.001);
        matrix::SquareMatrix invertedCameraMatrix = lu.inverseMatrix();
        double* icm = invertedCameraMatrix.getPtr();
        //    cl::Buffer buffer_P(*context, CL_MEM_COPY_HOST_PTR, sizeof(double) * 12, (void*)P);
        cl_double16 ICM({ icm[0], icm[1], icm[2], icm[3], icm[4], icm[5], icm[6], icm[7], icm[8],
                          icm[9], icm[10], icm[11], icm[12], icm[13], icm[14], icm[15] });
        inverseProjectionMatrices.push_back(ICM);
    }
    return inverseProjectionMatrices;
}

std::vector<float>
CuttingVoxelProjector::computeScalingFactors(std::vector<matrix::ProjectionMatrix> PM)
{
    std::vector<float> scalingFactors;
    double xoveryspacing = pdimx / pdimy;
    double yoverxspacing = pdimy / pdimx;
    for(std::size_t i = 0; i != PM.size(); i++)
    {
        double x1, x2, y1, y2;
        matrix::ProjectionMatrix pm = PM[i];
        std::array<double, 3> sourcePosition = pm.sourcePosition();
        std::array<double, 3> normalToDetector = pm.normalToDetector();
        std::array<double, 3> tangentToDetector = pm.tangentToDetectorYDirection();
        pm.project(sourcePosition[0] - normalToDetector[0], sourcePosition[1] - normalToDetector[1],
                   sourcePosition[2] - normalToDetector[2], &x1, &y1);
        pm.project(sourcePosition[0] - normalToDetector[0] + tangentToDetector[0],
                   sourcePosition[1] - normalToDetector[1] + tangentToDetector[1],
                   sourcePosition[2] - normalToDetector[2] + tangentToDetector[2], &x2, &y2);
        double scalingFactor
            = (x1 - x2) * (x1 - x2) * xoveryspacing + (y1 - y2) * (y1 - y2) * yoverxspacing;
        scalingFactors.push_back(scalingFactor);
    }
    return scalingFactors;
}

int CuttingVoxelProjector::backproject(float* volume,
                                       std::vector<std::shared_ptr<matrix::CameraI>>& cameraVector,
                                       uint64_t baseOffset)
{
    using namespace KCT::matrix;
    std::string msg;
    if(cameraVector.size() > pdimz)
    {
        msg = "Camera vector too large for given projection vector";
        LOGE << msg;
        throw std::runtime_error(msg);
    } else if(cameraVector.size() < pdimz)
    {
        LOGI << "Projection from reduced set of angles";
    }
    initializeOrUpdateVolumeBuffer();
    fillVolumeBufferByConstant(0.0f);
    if(tmpBuffer == nullptr || totalProjectionBufferSize > tmpBuffer_size)
    {
        tmpBuffer = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                 totalProjectionBufferSize, nullptr);
        tmpBuffer_size = totalProjectionBufferSize;
    }
    algFLOATvector_copy(*projectionBuffer, *tmpBuffer, totalPixelNum);
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(vdimx, vdimy, vdimz));
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
    for(std::size_t i = 0; i != cameraVector.size(); i++)
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
        offset = baseOffset + i * frameSize;
        if(useSidonProjector)
        {
            (*FLOATsidon_backproject)(eargs2, *volumeBuffer, *tmpBuffer, offset, ICM,
                                      SOURCEPOSITION, NORMALTODETECTOR, vdims, voxelSizes,
                                      volumeCenter, pdims, FLOATONE, pixelGranularity);
        } else if(useTTProjector)
        {
            (*FLOATta3_backproject)(eargs, *volumeBuffer, *tmpBuffer, offset, CM, SOURCEPOSITION,
                                    NORMALTODETECTOR, vdims, voxelSizes, volumeCenter, pdims,
                                    FLOATONE);
        } else
        {
            if(useCVPExactProjectionsScaling)
            {
                (*FLOATrescale_projections_exact)(eargs2, *tmpBuffer, offset, pdims_uint,
                                                  NORMALPROJECTION, VIRTUALPIXELSIZES,
                                                  VIRTUALDETECTORDISTANCE);
            } else
            {
                (*FLOATrescale_projections_cos)(eargs2, *tmpBuffer, offset, ICM, SOURCEPOSITION,
                                                NORMALTODETECTOR, pdims_uint, scalingFactor);
            }
            (*FLOATcutting_voxel_backproject)(eargs, *volumeBuffer, *tmpBuffer, offset, CM,
                                              SOURCEPOSITION, NORMALTODETECTOR, vdims, voxelSizes,
                                              volumeCenter, pdims, FLOATONE);
        }
    }
    cl_int err = Q[0]->enqueueReadBuffer(*volumeBuffer, CL_TRUE, 0, totalVolumeBufferSize, volume);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful writte buffer to the projection variable, code %d!", err);
        return -1;
    }
    return 0;
}

int CuttingVoxelProjector::backproject_minmax(
    float* volume, std::vector<std::shared_ptr<matrix::CameraI>>& cameraVector, uint64_t baseOffset)
{
    using namespace KCT::matrix;
    std::string msg;
    if(cameraVector.size() > pdimz)
    {
        msg = "Camera vector too large for given projection vector";
        LOGE << msg;
        throw std::runtime_error(msg);
    } else if(cameraVector.size() < pdimz)
    {
        LOGI << "Projection from reduced set of angles";
    }
    initializeOrUpdateVolumeBuffer();
    fillVolumeBufferByConstant(std::numeric_limits<float>::infinity());
    if(tmpBuffer == nullptr || totalProjectionBufferSize > tmpBuffer_size)
    {
        tmpBuffer = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE,
                                                 totalProjectionBufferSize, nullptr);
        tmpBuffer_size = totalProjectionBufferSize;
    }
    algFLOATvector_copy(*projectionBuffer, *tmpBuffer, totalPixelNum);
    cl::NDRange voxelRange(vdimx, vdimy, vdimz);
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(vdimx, vdimy, vdimz));
    cl::EnqueueArgs eargs2(*Q[0], cl::NDRange(pdimx, pdimy));
    cl_double16 CM;
    cl_double16 ICM;
    cl_double3 SOURCEPOSITION, NORMALTODETECTOR;
    cl_double2 NORMALPROJECTION;
    cl_double2 VIRTUALPIXELSIZES;
    double VIRTUALDETECTORDISTANCE = 1.0;
    std::shared_ptr<CameraI> P;
    std::array<double, 2> focalLength;
    unsigned int offset;
    for(std::size_t i = 0; i != cameraVector.size(); i++)
    {
        P = cameraVector[i];
        focalLength = P->focalLength();
        // Kernel parameters
        // scalingFactor = focalLength[0] * focalLength[1];
        P->projectionMatrixAsVector12((double*)&CM);
        P->inverseProjectionMatrixAsVector16((double*)&ICM);
        P->normalToDetector((double*)&NORMALTODETECTOR);
        P->principalRayProjection((double*)&NORMALPROJECTION);
        P->sourcePosition((double*)&SOURCEPOSITION);
        VIRTUALPIXELSIZES = { 1.0 / focalLength[0], 1.0 / focalLength[1] };
        offset = baseOffset + i * frameSize;
        (*FLOATrescale_projections_exact)(eargs2, *tmpBuffer, offset, pdims_uint, NORMALPROJECTION,
                                          VIRTUALPIXELSIZES, VIRTUALDETECTORDISTANCE);
        /*
                (*FLOATcutting_voxel_minmaxbackproject)(eargs, *volumeBuffer, *tmpBuffer, offset,
           CM, SOURCEPOSITION, NORMALTODETECTOR, vdims, voxelSizes, volumeCenter, pdims, FLOATONE);
         */

        algFLOATcutting_voxel_minmaxbackproject(*volumeBuffer, *tmpBuffer, offset, CM,
                                                SOURCEPOSITION, NORMALTODETECTOR, vdims, voxelSizes,
                                                volumeCenter, pdims, FLOATONE, voxelRange);
    }
    algFLOATvector_zero_infinite_values(*volumeBuffer, totalVoxelNum);
    cl_int err = Q[0]->enqueueReadBuffer(*volumeBuffer, CL_TRUE, 0, totalVolumeBufferSize, volume);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful writte buffer to the projection variable, code %d!", err);
        return -1;
    }
    return 0;
}

} // namespace KCT
