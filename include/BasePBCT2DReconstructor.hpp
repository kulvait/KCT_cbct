#pragma once

// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <CL/cl.hpp>
#include <chrono>
#include <ctime>
#include <iostream>

// Internal libraries
#include "AlgorithmsBarrierBuffers.hpp"
#include "BasePBCT2DOperator.hpp"
#include "BufferedFrame2D.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenProjectionMatrixReader.hpp"
#include "Kniha.hpp"
#include "MATRIX/utils.hpp"
#include "OPENCL/OpenCLManager.hpp"
#include "rawop.h"
#include "stringFormatter.h"

using namespace KCT::matrix;
namespace KCT {

class BasePBCT2DReconstructor : public virtual BasePBCT2DOperator
{
public:
    BasePBCT2DReconstructor(uint32_t pdimx,
                            uint32_t pdimy,
                            uint32_t pdimz,
                            uint32_t vdimx,
                            uint32_t vdimy,
                            uint32_t vdimz,
                            uint32_t workGroupSize = 256)
        : BasePBCT2DOperator(pdimx,
                             pdimy,
                             pdimz,
                             vdimx,
                             vdimy,
                             vdimz,
                             workGroupSize)
    {
    }

    virtual ~BasePBCT2DReconstructor() = default;

    virtual int reconstruct(uint32_t maxItterations, float minDiscrepancyError) = 0;

    void setReportingParameters(bool verbose,
                                uint32_t reportKthIteration = 0,
                                std::string intermediatePrefix = "");
    int initializeVectors(float* projection, float* volume, bool volumeContainsX0);
    double adjointProductTest();

protected:
    // OpenCL buffers
    std::shared_ptr<cl::Buffer> b_buf = nullptr;
    std::shared_ptr<cl::Buffer> x_buf = nullptr;
    // tmp_b_buf for rescaling, tmp_x_buf for LSQR
    std::shared_ptr<cl::Buffer> tmp_x_buf = nullptr, tmp_b_buf = nullptr;
    float* x = nullptr; // Volume data
    float* b = nullptr; // Projection data

    uint32_t reportKthIteration = 0;
    // Auxiliary functions
    void writeVolume(cl::Buffer& X, std::string path);
    void writeProjections(cl::Buffer& B, std::string path);
};

} // namespace KCT
