#include "Perfusion/BasePerfusionReconstructor.hpp"

namespace KCT {

void BasePerfusionReconstructor::setReportingParameters(bool verbose,
                                                        uint32_t reportKthIteration,
                                                        std::string progressPrefixPath)
{
    this->verbose = verbose;
    this->reportKthIteration = reportKthIteration;
    this->progressPrefixPath = progressPrefixPath;
}

void BasePerfusionReconstructor::initializeCVPProjector(bool useExactScaling)
{
    if(!isOpenCLInitialized())
    {
        useCVPProjector = true;
        exactProjectionScaling = useExactScaling;
        useSidonProjector = false;
        pixelGranularity = { 1, 1 };
        useTTProjector = false;
        CLINCLUDEutils();
        CLINCLUDEinclude();
        CLINCLUDEprojector();
        CLINCLUDEbackprojector();
        CLINCLUDErescaleProjections();
    } else
    {
        std::string err = "Could not initialize projector when OpenCL was already initialized.";
        LOGE << err;
        throw std::runtime_error(err.c_str());
    }
}

void BasePerfusionReconstructor::initializeSidonProjector(uint32_t probesPerEdgeX,
                                                          uint32_t probesPerEdgeY)
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

void BasePerfusionReconstructor::initializeTTProjector()
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

int BasePerfusionReconstructor::problemSetup(std::vector<float*> projections,
                                             std::vector<float*> basisVectorValues,
                                             std::vector<float*> volumes,
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
    std::array<double, 3> centerGlobal = { volumeOffsetX, volumeOffsetY, volumeOffsetZ };
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
            LOGW << io::xprintf(
                "Apparently the volume is specified such that its center do not "
                "belong to the half space orthogonal to the principal ray in %d-th projection.",
                k);
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
    initializeAlgorithmsBuffers(); // Partial algorithms
    return initializeVectors(projections, basisVectorValues, volumes, volumeContainsX0);
}

int BasePerfusionReconstructor::initializeVectors(std::vector<float*> projections,
                                                  std::vector<float*> basisVectorValues,
                                                  std::vector<float*> volumes,
                                                  bool volumeContainsX0)
{
    this->b = projections;
    this->basisFunctionsValues = basisVectorValues;
    this->x = volumes;
    XVNUM = volumes.size();
    BVNUM = projections.size();
    cl_int err;

    // Initialize buffers x_buf, v_buf and v_buf by zeros
    std::shared_ptr<cl::Buffer> bf;
    for(std::size_t i = 0; i != x.size(); i++)
    {
        if(volumeContainsX0)
        {
            bf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                              sizeof(float) * XDIM, (void*)volumes[i], &err);
        } else
        {
            bf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * XDIM,
                                              nullptr, &err);
        }
        x_buf.push_back(bf);
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
    }
    tmp_b_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * BDIM,
                                             nullptr, &err);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
        return -1;
    }
    tmp_x_buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * XDIM,
                                             nullptr, &err);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
        return -1;
    }
    return 0;
}

int BasePerfusionReconstructor::allocateXBuffers(uint32_t xBufferCount)
{
    cl_int err;
    std::shared_ptr<cl::Buffer> buf;
    while(this->x_buffers.size() < xBufferCount)
    {
        std::vector<std::shared_ptr<cl::Buffer>> XV;
        for(std::uint32_t basisFunctionID = 0; basisFunctionID != XVNUM; basisFunctionID++)
        {
            buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * XDIM,
                                               nullptr, &err);
            if(err != CL_SUCCESS)
            {
                LOGE << io::xprintf("Unsucessful initialization of X buffer with error code %d!",
                                    err);
                return -1;
            }
            XV.emplace_back(buf);
        }
        x_buffers.emplace_back(XV);
    }
    return 0;
}

int BasePerfusionReconstructor::allocateBBuffers(uint32_t bBufferCount)
{
    cl_int err;
    std::shared_ptr<cl::Buffer> buf;
    while(this->b_buffers.size() < bBufferCount)
    {
        std::vector<std::shared_ptr<cl::Buffer>> XB;
        for(std::uint32_t sweepID = 0; sweepID != BVNUM; sweepID++)
        {
            buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * BDIM,
                                               nullptr, &err);
            if(err != CL_SUCCESS)
            {
                LOGE << io::xprintf("Unsucessful initialization of B buffer with error code %d!",
                                    err);
                return -1;
            }
            XB.emplace_back(buf);
        }
        b_buffers.emplace_back(XB);
    }
    return 0;
}

int BasePerfusionReconstructor::allocateTmpXBuffers(uint32_t xBufferCount)
{
    cl_int err;
    std::shared_ptr<cl::Buffer> buf;
    while(this->tmp_x_buffers.size() < xBufferCount)
    {
        std::vector<std::shared_ptr<cl::Buffer>> XV;
        for(std::uint32_t basisFunctionID = 0; basisFunctionID != XVNUM; basisFunctionID++)
        {
            buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * XDIM,
                                               nullptr, &err);
            if(err != CL_SUCCESS)
            {
                LOGE << io::xprintf("Unsucessful initialization of X buffer with error code %d!",
                                    err);
                return -1;
            }
            XV.emplace_back(buf);
        }
        tmp_x_buffers.emplace_back(XV);
    }
    return 0;
}

int BasePerfusionReconstructor::allocateTmpBBuffers(uint32_t bBufferCount)
{
    cl_int err;
    std::shared_ptr<cl::Buffer> buf;
    while(this->tmp_b_buffers.size() < bBufferCount)
    {
        std::vector<std::shared_ptr<cl::Buffer>> XB;
        for(std::uint32_t sweepID = 0; sweepID != BVNUM; sweepID++)
        {
            buf = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(float) * BDIM,
                                               nullptr, &err);
            if(err != CL_SUCCESS)
            {
                LOGE << io::xprintf("Unsucessful initialization of B buffer with error code %d!",
                                    err);
                return -1;
            }
            XB.emplace_back(buf);
        }
        tmp_b_buffers.emplace_back(XB);
    }
    return 0;
}

std::vector<std::shared_ptr<cl::Buffer>> BasePerfusionReconstructor::getXBuffers(uint32_t i)
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

std::vector<std::shared_ptr<cl::Buffer>> BasePerfusionReconstructor::getBBuffers(uint32_t i)
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

std::vector<std::shared_ptr<cl::Buffer>> BasePerfusionReconstructor::getTmpXBuffers(uint32_t i)
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

std::vector<std::shared_ptr<cl::Buffer>> BasePerfusionReconstructor::getTmpBBuffers(uint32_t i)
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
 * Computes square of the Euclidean norm of the buffer with pdimx * pdimy * pdimz float elements
 *
 * @param B input buffer
 *
 * @return square Euclidean norm
 */
double
BasePerfusionReconstructor::normBBuffer_barrier_double(std::vector<std::shared_ptr<cl::Buffer>>& B)
{ // Use workGroupSize that is private constant default to 256
    double sum = 0.0;
    for(std::size_t i = 0; i != B.size(); i++)
    {
        sum += AlgorithmsBarrierBuffers::normBBuffer_barrier_double(*B[i]);
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
double
BasePerfusionReconstructor::normXBuffer_barrier_double(std::vector<std::shared_ptr<cl::Buffer>>& X)
{
    double sum = 0.0;
    for(std::size_t i = 0; i != X.size(); i++)
    {
        sum += AlgorithmsBarrierBuffers::normXBuffer_barrier_double(*X[i]);
    }
    return sum;
}

double BasePerfusionReconstructor::scalarProductXBuffer_barrier_double(
    std::vector<std::shared_ptr<cl::Buffer>>& A, std::vector<std::shared_ptr<cl::Buffer>>& B)
{
    if(A.size() != B.size())
    {
        std::string err = "Vectors A and B has a different sizes!";
        LOGE << err;
        throw std::runtime_error(err);
    }
    double sum = 0.0;
    for(std::size_t i = 0; i != A.size(); i++)
    {
        sum += AlgorithmsBarrierBuffers::scalarProductXBuffer_barrier_double(*A[i], *B[i]);
    }
    return sum;
}

double BasePerfusionReconstructor::scalarProductBBuffer_barrier_double(
    std::vector<std::shared_ptr<cl::Buffer>>& A, std::vector<std::shared_ptr<cl::Buffer>>& B)
{
    if(A.size() != B.size())
    {
        std::string err = "Vectors A and B has a different sizes!";
        LOGE << err;
        throw std::runtime_error(err);
    }
    double sum = 0.0;
    for(std::size_t i = 0; i != A.size(); i++)
    {
        sum += AlgorithmsBarrierBuffers::scalarProductBBuffer_barrier_double(*A[i], *B[i]);
    }
    return sum;
}

int BasePerfusionReconstructor::backproject_partial(cl::Buffer& B, cl::Buffer& X, uint32_t angleID)
{
    Q[0]->enqueueFillBuffer<cl_float>(X, FLOATZERO, 0, XDIM * sizeof(float));
    unsigned int frameSize = pdimx * pdimy;
    unsigned int offset = angleID * frameSize;
    algFLOATvector_copy_offset(B, *tmp_b_buf, offset, frameSize);
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(vdimz, vdimy, vdimx));
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
    P = cameraVector[angleID];
    focalLength = P->focalLength();
    // Kernel parameters
    scalingFactor = focalLength[0] * focalLength[1];
    P->projectionMatrixAsVector12((double*)&CM);
    P->inverseProjectionMatrixAsVector16((double*)&ICM);
    P->normalToDetector((double*)&NORMALTODETECTOR);
    P->principalRayProjection((double*)&NORMALPROJECTION);
    P->sourcePosition((double*)&SOURCEPOSITION);
    VIRTUALPIXELSIZES = { 1.0 / focalLength[0], 1.0 / focalLength[1] };
    if(useSidonProjector)
    {
        (*FLOATsidon_backproject)(eargs2, X, *tmp_b_buf, offset, ICM, SOURCEPOSITION,
                                  NORMALTODETECTOR, vdims, voxelSizes, volumeCenter, pdims,
                                  FLOATONE, pixelGranularity);
    } else if(useTTProjector)
    {
        (*FLOATta3_backproject)(eargs, X, *tmp_b_buf, offset, CM, SOURCEPOSITION, NORMALTODETECTOR,
                                vdims, voxelSizes, volumeCenter, pdims, FLOATONE);
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
                                          NORMALTODETECTOR, vdims, voxelSizes, volumeCenter, pdims,
                                          FLOATONE);
    }
    return 0;
}

int BasePerfusionReconstructor::backproject(std::vector<std::shared_ptr<cl::Buffer>>& B,
                                            std::vector<std::shared_ptr<cl::Buffer>>& X,
                                            bool blocking)
{
    std::shared_ptr<Watches> w;
    if(blocking)
    {
        Q[0]->finish();
        w = std::make_shared<Watches>();
        LOGD << "START Backrojection";
    }
    zeroXBuffers(X);
    for(std::size_t j = 0; j != B.size(); j++)
    {
        for(std::size_t i = 0; i != pdimz; i++)
        {
            backproject_partial(*B[j], *tmp_x_buf, i);
            for(std::size_t k = 0; k != X.size(); k++)
            {
                float v = basisFunctionsValues[k][j * pdimz + i];
                algFLOATvector_A_equals_A_plus_cB(*X[k], *tmp_x_buf, v, XDIM);
            }
        }
    }
    if(blocking)
    {
        Q[0]->finish();
        LOGD << w->textWithTimeFromLastReset("Backprojection duration");
    }
    return 0;
}

int BasePerfusionReconstructor::project(cl::Buffer& X, cl::Buffer& B)
{
    Q[0]->enqueueFillBuffer<cl_float>(B, FLOATZERO, 0, BDIM * sizeof(float));
    unsigned int frameSize = pdimx * pdimy;
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(vdimz, vdimy, vdimx));
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
            (*FLOATcutting_voxel_project)(eargs, X, B, offset, CM, SOURCEPOSITION, NORMALTODETECTOR,
                                          vdims, voxelSizes, volumeCenter, pdims, FLOATONE);
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

int BasePerfusionReconstructor::project(std::vector<std::shared_ptr<cl::Buffer>>& X,
                                        std::vector<std::shared_ptr<cl::Buffer>>& B,
                                        bool blocking)
{
    std::shared_ptr<Watches> w;
    if(blocking)
    {
        Q[0]->finish();
        w = std::make_shared<Watches>();
        LOGD << "START Projection";
    }
    zeroBBuffers(B);
    unsigned int frameSize = pdimx * pdimy;
    for(uint32_t basisIND = 0; basisIND != X.size(); ++basisIND)
    {
        Q[0]->enqueueFillBuffer<cl_float>(*tmp_b_buf, FLOATZERO, 0, BDIM * sizeof(float));
        project(*X[basisIND], *tmp_b_buf);
        for(uint32_t sweepID = 0; sweepID != B.size(); sweepID++)
        {
            for(std::size_t angleID = 0; angleID != pdimz; angleID++)
            {
                unsigned int offset = frameSize * angleID;
                float scaleBy = basisFunctionsValues[basisIND][sweepID * pdimz + angleID];
                algFLOATvector_A_equals_A_plus_cB_offset(*B[sweepID], *tmp_b_buf, scaleBy, offset,
                                                         frameSize);
            }
        }
    }
    if(blocking)
    {
        Q[0]->finish();
        LOGD << w->textWithTimeFromLastReset("Projection duration");
    }
    return 0;
}

void BasePerfusionReconstructor::zeroXBuffers(std::vector<std::shared_ptr<cl::Buffer>>& X)
{
    for(std::size_t basisIND = 0; basisIND != X.size(); ++basisIND)
    {
        Q[0]->enqueueFillBuffer<cl_float>(*X[basisIND], FLOATZERO, 0, XDIM * sizeof(float));
    }
}

void BasePerfusionReconstructor::zeroBBuffers(std::vector<std::shared_ptr<cl::Buffer>>& B)
{
    for(std::size_t sweepID = 0; sweepID != B.size(); sweepID++)
    {
        Q[0]->enqueueFillBuffer<cl_float>(*B[sweepID], FLOATZERO, 0, BDIM * sizeof(float));
    }
}

int BasePerfusionReconstructor::addIntoFirstVectorSecondVectorScaled(
    std::vector<std::shared_ptr<cl::Buffer>>& A,
    std::vector<std::shared_ptr<cl::Buffer>>& B,
    float c,
    unsigned int size)
{
    if(A.size() != B.size())
    {
        std::string err = "Vectors A and B has a different sizes!";
        LOGE << err;
        throw std::runtime_error(err);
    }
    for(std::size_t i = 0; i != A.size(); i++)
    {
        algFLOATvector_A_equals_A_plus_cB(*A[i], *B[i], c, size);
    }
    return 0;
}
int BasePerfusionReconstructor::algFLOATvector_A_equals_cB(
    std::vector<std::shared_ptr<cl::Buffer>>& A,
    std::vector<std::shared_ptr<cl::Buffer>>& B,
    float c,
    unsigned int size,
    bool blocking)
{
    if(A.size() != B.size())
    {
        std::string err = "Vectors A and B has a different sizes!";
        LOGE << err;
        throw std::runtime_error(err);
    }
    for(std::size_t i = 0; i != A.size(); i++)
    {
        Kniha::algFLOATvector_A_equals_cB(*A[i], *B[i], c, size, blocking);
    }
    return 0;
}

/**
 * @brief As a side effect uses x[0] as a tmp buffer and overwrites it
 *
 * @param X
 * @param path
 */
void BasePerfusionReconstructor::writeVolume(cl::Buffer& X, std::string path)
{
    uint16_t buf[3];
    buf[0] = vdimy;
    buf[1] = vdimx;
    buf[2] = vdimz;
    io::createEmptyFile(path, 0, true); // Try if this is faster
    io::appendBytes(path, (uint8_t*)buf, 6);
    Q[0]->enqueueReadBuffer(X, CL_TRUE, 0, sizeof(float) * XDIM, x[0]);
    io::appendBytes(path, (uint8_t*)x[0], XDIM * sizeof(float));
}

/**
 * @brief As a side effect uses b[0] as tmp buffer and overwrites it
 *
 * @param B
 * @param path
 */
void BasePerfusionReconstructor::writeProjections(cl::Buffer& B, std::string path)
{
    uint16_t buf[3];
    buf[0] = pdimy;
    buf[1] = pdimx;
    buf[2] = pdimz;
    io::createEmptyFile(path, 0, true); // Try if this is faster
    io::appendBytes(path, (uint8_t*)buf, 6);
    Q[0]->enqueueReadBuffer(B, CL_TRUE, 0, sizeof(float) * BDIM, b[0]);
    io::appendBytes(path, (uint8_t*)b[0], BDIM * sizeof(float));
}

void BasePerfusionReconstructor::writeVolume(std::vector<std::shared_ptr<cl::Buffer>>& X,
                                             std::string path)
{
    for(std::size_t i = 0; i != XVNUM; i++)
    {
        std::string newpath = io::xprintf("%s_elm%d", path.c_str(), i);
        writeVolume(*X[i], newpath);
    }
}

void BasePerfusionReconstructor::writeProjections(std::vector<std::shared_ptr<cl::Buffer>>& B,
                                                  std::string path)
{
    for(std::size_t i = 0; i != BVNUM; i++)
    {
        std::string newpath = io::xprintf("%s_proj%d", path.c_str(), i);
        writeProjections(*B[i], newpath);
    }
}

void BasePerfusionReconstructor::setTimestamp(bool finishCommandQueue)
{
    if(finishCommandQueue)
    {
        Q[0]->finish();
    }
    timestamp = std::chrono::steady_clock::now();
}

std::chrono::milliseconds
BasePerfusionReconstructor::millisecondsFromTimestamp(bool setNewTimestamp)
{
    std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - timestamp);
    if(setNewTimestamp)
    {
        setTimestamp(false);
    }
    return ms;
}

void BasePerfusionReconstructor::reportTime(std::string msg,
                                            bool finishCommandQueue,
                                            bool setNewTimestamp)
{
    if(finishCommandQueue)
    {
        Q[0]->finish();
    }
    auto duration = millisecondsFromTimestamp(setNewTimestamp);
    LOGD << io::xprintf("%s: %0.2fs", msg.c_str(), duration.count() / 1000.0);
}

int BasePerfusionReconstructor::copyFloatVector(std::vector<std::shared_ptr<cl::Buffer>>& VA,
                                                std::vector<std::shared_ptr<cl::Buffer>>& VB,
                                                unsigned int size)
{
    if(VA.size() != VB.size())
    {
        std::string err = "Vectors A and B has a different sizes!";
        LOGE << err;
        throw std::runtime_error(err);
    }
    for(std::size_t i = 0; i != VA.size(); i++)
    {
        algFLOATvector_copy(*VA[i], *VB[i], size);
    }
    return 0;
}

int BasePerfusionReconstructor::scaleFloatVector(std::vector<std::shared_ptr<cl::Buffer>>& A,
                                                 float f,
                                                 unsigned int size)
{
    for(uint32_t i = 0; i != A.size(); i++)
    {
        algFLOATvector_scale(*A[i], f, size);
    }
    return 0;
}

int BasePerfusionReconstructor::addIntoFirstVectorScaledSecondVector(
    std::vector<std::shared_ptr<cl::Buffer>>& a,
    std::vector<std::shared_ptr<cl::Buffer>>& b,
    float f,
    unsigned int size)
{
    for(std::size_t i = 0; i != a.size(); i++)
    {
        algFLOATvector_A_equals_Ac_plus_B(*a[i], *b[i], f, size);
    }
    return 0;
}

std::vector<matrix::ProjectionMatrix> BasePerfusionReconstructor::encodeProjectionMatrices(
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

double BasePerfusionReconstructor::adjointProductTest()
{
    std::vector<std::shared_ptr<cl::Buffer>> xa_buf; // X buffers
    allocateXBuffers(1);
    xa_buf = getXBuffers(0);
    allocateBBuffers(1);
    std::vector<std::shared_ptr<cl::Buffer>> ba_buf; // B buffers
    ba_buf = getBBuffers(0);
    project(x_buf, ba_buf);
    backproject(b_buf, xa_buf);
    double bdotAx = scalarProductBBuffer_barrier_double(b_buf, ba_buf);
    double ATbdotx = scalarProductXBuffer_barrier_double(x_buf, xa_buf);
    return (bdotAx / ATbdotx);
}

} // namespace KCT
