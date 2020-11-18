#pragma once

// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <CL/cl.hpp>
#include <chrono>
#include <ctime>
#include <iostream>

// Internal libraries
#include "BaseReconstructor.hpp"
#include "DEN/DenProjectionMatrixReader.hpp"
#include "MATRIX/LightProjectionMatrix.hpp"
#include "MATRIX/ProjectionMatrix.hpp"
#include "OPENCL/OpenCLManager.hpp"
#include "rawop.h"
#include "stringFormatter.h"

namespace CTL {

class CGLSReconstructor : public BaseReconstructor
{
public:
    /**
     * Initialize Cutting Voxel Projector
     *
     * @param volume Pointer to volume file
     * @param vdimx Volume x dimension
     * @param vdimy Volume y dimension
     * @param vdimz Volume z dimension
     * @param xpath Path of cl kernel files
     * @param debug Should debugging be used by suppliing source and -g as options
     */
    CGLSReconstructor(uint32_t pdimx,
                      uint32_t pdimy,
                      uint32_t pdimz,
                      double pixelSpacingX,
                      double pixelSpacingY,
                      uint32_t vdimx,
                      uint32_t vdimy,
                      uint32_t vdimz,
                      double voxelSpacingX,
                      double voxelSpacingY,
                      double voxelSpacingZ,
                      uint32_t workGroupSize = 256)
        : BaseReconstructor(pdimx,
                            pdimy,
                            pdimz,
                            pixelSpacingX,
                            pixelSpacingY,
                            vdimx,
                            vdimy,
                            vdimz,
                            voxelSpacingX,
                            voxelSpacingY,
                            voxelSpacingZ,
                            workGroupSize)
    {
    }

    void setReportingParameters(bool reportProgress,
                                std::string progressPrefixPath,
                                uint32_t reportKthIteration)
    {
        this->reportProgress = reportProgress;
        if(reportProgress)
        {
            this->progressPrefixPath = progressPrefixPath;
            this->reportKthIteration = reportKthIteration;
        }
    }

    virtual int reconstruct(std::shared_ptr<io::DenProjectionMatrixReader> matrices,
                            uint32_t maxIterations = 100,
                            float errCondition = 0.01);

    int reconstruct_experimental(std::shared_ptr<io::DenProjectionMatrixReader> matrices,
                                 uint32_t maxIterations = 100,
                                 float errCondition = 0.01);

    int reconstructDiagonalPreconditioner(std::shared_ptr<io::DenProjectionMatrixReader> matrices,
                                          std::shared_ptr<cl::Buffer> invertedpreconditioner_xbuf,
                                          uint32_t maxIterations = 100,
                                          float errCondition = 0.01);

    int reconstructDiagonalPreconditioner(std::shared_ptr<io::DenProjectionMatrixReader> matrices,
                                          float* invertedpreconditioner,
                                          uint32_t maxIterations = 100,
                                          float errCondition = 0.01);

    int reconstructJacobi(std::shared_ptr<io::DenProjectionMatrixReader> matrices,
                          uint32_t maxIterations = 100,
                          float errCondition = 0.01);

    void precomputeJacobiPreconditioner(std::shared_ptr<cl::Buffer> X,
                                        std::shared_ptr<io::DenProjectionMatrixReader> matrices);

private:
    bool reportProgress = false;
    std::string progressPrefixPath = "";
    uint32_t reportKthIteration = 0;
};

} // namespace CTL
