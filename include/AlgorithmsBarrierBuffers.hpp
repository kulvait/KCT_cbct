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
#include "ReductionParameters.hpp"
#include "rawop.h"
#include "stringFormatter.h"

using namespace KCT::matrix;
namespace KCT {

class AlgorithmsBarrierBuffers : public virtual Kniha
{
public:
    AlgorithmsBarrierBuffers()
    {
        rp = nullptr;
        CLINCLUDEutils();
    }

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
    std::shared_ptr<ReductionParameters> rp;

    // Functions to manipulate with buffers
    float normBBuffer_barrier(cl::Buffer& B,
                              std::shared_ptr<ReductionParameters> rp = nullptr,
                              uint32_t QID = 0);
    float normXBuffer_barrier(cl::Buffer& X,
                              std::shared_ptr<ReductionParameters> rp = nullptr,
                              uint32_t QID = 0);
    float normBBuffer_frame(cl::Buffer& B,
                            std::shared_ptr<ReductionParameters> rp = nullptr,
                            uint32_t QID = 0);
    float normXBuffer_frame(cl::Buffer& X,
                            std::shared_ptr<ReductionParameters> rp = nullptr,
                            uint32_t QID = 0);
    float sumBBuffer_barrier_float(cl::Buffer& B,
                                   std::shared_ptr<ReductionParameters> rp = nullptr,
                                   uint32_t QID = 0);
    float sumXBuffer_barrier_float(cl::Buffer& X,
                                   std::shared_ptr<ReductionParameters> rp = nullptr,
                                   uint32_t QID = 0);
    float maxBBuffer_barrier_float(cl::Buffer& B,
                                   std::shared_ptr<ReductionParameters> rp = nullptr,
                                   uint32_t QID = 0);
    float maxXBuffer_barrier_float(cl::Buffer& X,
                                   std::shared_ptr<ReductionParameters> rp = nullptr,
                                   uint32_t QID = 0);
    float isotropicTVNormXBuffer_barrier_float(cl::Buffer& GX,
                                               cl::Buffer& GY,
                                               std::shared_ptr<ReductionParameters> rp = nullptr,
                                               uint32_t QID = 0);
    double normBBuffer_barrier_double(cl::Buffer& B,
                                      std::shared_ptr<ReductionParameters> rp = nullptr,
                                      uint32_t QID = 0);
    double normXBuffer_barrier_double(cl::Buffer& X,
                                      std::shared_ptr<ReductionParameters> rp = nullptr,
                                      uint32_t QID = 0);
    double normBBuffer_frame_double(cl::Buffer& B,
                                    std::shared_ptr<ReductionParameters> rp = nullptr,
                                    uint32_t QID = 0);
    double normXBuffer_frame_double(cl::Buffer& X,
                                    std::shared_ptr<ReductionParameters> rp = nullptr,
                                    uint32_t QID = 0);
    double scalarProductBBuffer_barrier_double(cl::Buffer& A,
                                               cl::Buffer& B,
                                               std::shared_ptr<ReductionParameters> rp = nullptr,
                                               uint32_t QID = 0);
    double scalarProductXBuffer_barrier_double(cl::Buffer& A,
                                               cl::Buffer& B,
                                               std::shared_ptr<ReductionParameters> rp = nullptr,
                                               uint32_t QID = 0);

    double isotropicTVNormXBuffer_barrier_double(cl::Buffer& GX,
                                                 cl::Buffer& GY,
                                                 std::shared_ptr<ReductionParameters> rp = nullptr,
                                                 uint32_t QID = 0);
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
    int arrayIntoBuffer(float* c_array, cl::Buffer cl_buffer, uint64_t size, uint32_t QID = 0);
    /**
     * Copy CL:buffer into the float* array of size elements. The buffer must have appropriate size.
     *
     * @param cl_buffer Block of OpenCL memory
     * @param c_array Block of C memory
     * @param size number of elements in c_array
     *
     * @return 0 on success
     */
    int bufferIntoArray(cl::Buffer cl_buffer, float* c_array, uint64_t size, uint32_t QID = 0);
    std::vector<std::shared_ptr<cl::Buffer>> tmp_red1;
    std::vector<std::shared_ptr<cl::Buffer>> tmp_red2;

private:
    bool reductionParametersSet = false;
    bool algorithmsBuffersInitialized = false;
    uint64_t tmp_red1_bytesize = 0, tmp_red2_bytesize = 0;
};

} // namespace KCT
