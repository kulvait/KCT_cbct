#pragma once

// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <CL/cl.hpp>
#include <chrono>
#include <ctime>
#include <iostream>

// Internal libraries
#include "DEN/DenProjectionMatrixReader.hpp"
#include "Kniha.hpp"
#include "MATRIX/LightProjectionMatrix.hpp"
#include "MATRIX/ProjectionMatrix.hpp"
#include "MATRIX/utils.hpp"
#include "OPENCL/OpenCLManager.hpp"
#include "PROG/KCTException.hpp"
#include "rawop.h"
#include "stringFormatter.h"

using namespace KCT::matrix;
namespace KCT {

class AlgorithmsBarrierBuffers : public virtual Kniha
{
public:
    AlgorithmsBarrierBuffers() { CLINCLUDEutils(); }

    AlgorithmsBarrierBuffers(uint32_t pdimx,
                             uint32_t pdimy,
                             uint32_t pdimz,
                             uint32_t vdimx,
                             uint32_t vdimy,
                             uint32_t vdimz,
                             uint32_t workGroupSize = 256)
    {
        initReductionParameters(pdimx, pdimy, pdimz, vdimx, vdimy, vdimz, workGroupSize);
        CLINCLUDEutils();
    }

    void initReductionParameters(uint32_t pdimx,
                                 uint32_t pdimy,
                                 uint32_t pdimz,
                                 uint32_t vdimx,
                                 uint32_t vdimy,
                                 uint32_t vdimz,
                                 uint32_t workGroupSize = 256);

    int updateReductionParameters(uint32_t pdimx,
                                  uint32_t pdimy,
                                  uint32_t pdimz,
                                  uint32_t vdimx,
                                  uint32_t vdimy,
                                  uint32_t vdimz,
                                  uint32_t workGroupSize = 256);
    // Initialization of buffers
    int initReductionBuffers();
    int updateReductionBuffers();

protected:
    uint32_t pdimx, pdimy, pdimz, vdimx, vdimy, vdimz;
    uint32_t workGroupSize = 256;
    uint64_t XDIM, BDIM, XDIM_ALIGNED, BDIM_ALIGNED, XDIM_REDUCED1, BDIM_REDUCED1,
        XDIM_REDUCED1_ALIGNED, BDIM_REDUCED1_ALIGNED, XDIM_REDUCED2, BDIM_REDUCED2,
        XDIM_REDUCED2_ALIGNED, BDIM_REDUCED2_ALIGNED;

    // Functions to manipulate with buffers
    float normBBuffer_barrier(cl::Buffer& B);
    float normXBuffer_barrier(cl::Buffer& X);
    float normBBuffer_frame(cl::Buffer& B);
    float normXBuffer_frame(cl::Buffer& X);
    float sumBBuffer_barrier_float(cl::Buffer& B);
    float sumXBuffer_barrier_float(cl::Buffer& X);
    float maxBBuffer_barrier_float(cl::Buffer& B);
    float maxXBuffer_barrier_float(cl::Buffer& X);
    double normBBuffer_barrier_double(cl::Buffer& B);
    double normXBuffer_barrier_double(cl::Buffer& X);
    double normBBuffer_frame_double(cl::Buffer& B);
    double normXBuffer_frame_double(cl::Buffer& X);
    double scalarProductBBuffer_barrier_double(cl::Buffer& A, cl::Buffer& B);
    double scalarProductXBuffer_barrier_double(cl::Buffer& A, cl::Buffer& B);

    /**
     * Copy float* array of size elements into the CL::buffer. The buffer must have appropriate
     * size.
     *
     * @param c_array Block of C memory
     * @param cl_buffer Block of OpenCL memory
     * @param size number of elements in c_array
     *
     * @return 0 on success
     */
    int arrayIntoBuffer(float* c_array, cl::Buffer cl_buffer, uint64_t size);
    /**
     * Copy CL:buffer into the float* array of size elements. The buffer must have appropriate size.
     *
     * @param cl_buffer Block of OpenCL memory
     * @param c_array Block of C memory
     * @param size number of elements in c_array
     *
     * @return 0 on success
     */
    int bufferIntoArray(cl::Buffer cl_buffer, float* c_array, uint64_t size);
    std::shared_ptr<cl::Buffer> tmp_b_red1 = nullptr, tmp_b_red2 = nullptr;
    std::shared_ptr<cl::Buffer> tmp_x_red1 = nullptr, tmp_x_red2 = nullptr;

private:
    bool reductionParametersSet = false;
    bool algorithmsBuffersInitialized = false;
    uint32_t tmp_red1_bytesize, tmp_red2_bytesize;
};

} // namespace KCT
