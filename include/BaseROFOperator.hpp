#pragma once

// External libraries
#include <CL/cl.hpp>
#include <chrono>
#include <memory>

// Internal libraries
#include "AlgorithmsBarrierBuffers.hpp"
#include "BasePBCTOperator.hpp" // Assuming this contains useful base operator methods
#include "GradientType.hpp"

namespace KCT {

class BaseROFOperator : public virtual Kniha, public AlgorithmsBarrierBuffers
{
public:
    BaseROFOperator(uint32_t vdimx, uint32_t vdimy, uint32_t vdimz, uint32_t workGroupSize = 256)
        : AlgorithmsBarrierBuffers(0, 0, 0, vdimx, vdimy, vdimz, workGroupSize)
        , workGroupSize(workGroupSize)
        , vdimx(vdimx)
        , vdimy(vdimy)
        , vdimz(vdimz)
    {
        XDIM = static_cast<uint64_t>(vdimx) * static_cast<uint64_t>(vdimy)
            * static_cast<uint64_t>(vdimz);
        vdims = cl_int3({ int(vdimx), int(vdimy), int(vdimz) });
        timestamp = std::chrono::steady_clock::now();
    }

    virtual int reconstruct(float mu,
                            float tau,
                            float sigma,
                            float theta,
                            uint32_t maxPDHGIterations,
                            float errConditionPDHG)
        = 0;

    int initializeVolume(float* volume);

    void initializeVolumeConvolution();
    void initializeProximal();
    void initializeGradient();

    void setTimestamp(bool finishCommandQueue);

    int initializeOpenCL(uint32_t platformID,
                         uint32_t* deviceIds,
                         uint32_t deviceIdsLength,
                         std::string xpath,
                         bool debug,
                         bool relaxed);

    int problemSetup(double voxelSpacingX, double voxelSpacingY, double voxelSpacingZ);

    int volume_gradient2D(cl::Buffer& F, cl::Buffer& GX, cl::Buffer& GY);

    int volume_gradient2D_adjoint(cl::Buffer& GX, cl::Buffer& GY, cl::Buffer& D);

    std::chrono::milliseconds millisecondsFromTimestamp(bool setNewTimestamp);

    std::string printTime(std::string msg, bool finishCommandQueue, bool setNewTimestamp);

    void reportTime(std::string msg, bool finishCommandQueue, bool setNewTimestamp);
    void setVerbose(bool verbose, std::string intermediatePrefix);
    void setReportingParameters(bool verbose,
                                uint32_t reportKthIteration,
                                std::string intermediatePrefix);

    void setGradientType(GradientType type);

protected:
    const uint32_t workGroupSize;
    const uint32_t vdimx, vdimy, vdimz;
    float* x = nullptr;
    double voxelSpacingX, voxelSpacingY, voxelSpacingZ;
    cl_double3 voxelSizes;
    cl_float3 voxelSizesF;
    GradientType useGradientType;

    std::shared_ptr<cl::Buffer> x_buf = nullptr;
    std::shared_ptr<cl::Buffer> tmp_x_buf = nullptr;
    std::vector<std::shared_ptr<cl::Buffer>> x_buffers, tmp_x_buffers;

    void writeVolume(cl::Buffer& X, const std::string& path);

    uint64_t XDIM;
    cl_int3 vdims;
    std::chrono::time_point<std::chrono::steady_clock> timestamp;

    bool verbose = false;
    std::string intermediatePrefix = "";
    uint32_t reportKthIteration = 0;

    int allocateXBuffers(uint32_t xBufferCount);
    std::shared_ptr<cl::Buffer> getXBuffer(uint32_t i);

private:
    bool gradientInitialized = false;
};

} // namespace KCT
