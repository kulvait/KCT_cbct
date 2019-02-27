#pragma once

// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <CL/cl.hpp>
#include <iostream>

// Internal libraries
#include "DEN/DenProjectionMatrixReader.hpp"
#include "MATRIX/ProjectionMatrix.hpp"
#include "OPENCL/OpenCLManager.hpp"
#include "rawop.h"
#include "stringFormatter.h"

namespace CTL {

class CGLSReconstructor
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
                      uint32_t vdimx,
                      uint32_t vdimy,
                      uint32_t vdimz,
                      std::string xpath,
                      bool debug)
        : pdimx(pdimx)
        , pdimy(pdimy)
        , pdimz(pdimz)
        , vdimx(vdimx)
        , vdimy(vdimy)
        , vdimz(vdimz)
        , xpath(xpath)
        , debug(debug)
    {
    }

    /** Initializes OpenCL.
     *
     * Initialization is done via C++ layer that works also with OpenCL 1.1.
     *
     *
     * @return
     * @see [OpenCL C++
     * manual](https://www.khronos.org/registry/OpenCL/specs/opencl-cplusplus-1.1.pdf)
     * @see [OpenCL C++
     * tutorial](http://simpleopencl.blogspot.com/2013/06/tutorial-simple-start-with-opencl-and-c.html)
     */
    int initializeOpenCL(uint32_t platformId = 0);

    int initializeVectors(float* projection, float* volume);

    int reconstruct(std::shared_ptr<io::DenProjectionMatrixReader> matrices);

private:
    float* x = nullptr; // Volume data
    float* b = nullptr; // Projection data
    uint32_t pdimx, pdimy, pdimz, vdimx, vdimy, vdimz;
    std::string xpath; // Path where the program executes
    bool debug;

    std::shared_ptr<cl::Device> device = nullptr;
    std::shared_ptr<cl::Context> context = nullptr;
    std::shared_ptr<cl::Image3D> volumeImage = nullptr;
    std::shared_ptr<cl::CommandQueue> Q = nullptr;
    std::shared_ptr<cl::Buffer> b_buf = nullptr, c_buf = nullptr, d_buf = nullptr;
    std::shared_ptr<cl::Buffer> x_buf = nullptr, y_buf = nullptr, z_buf = nullptr;

    std::shared_ptr<cl::make_kernel<cl::Buffer&,
                                    cl::Buffer&,
                                    cl_double16&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int4&,
                                    cl_double3&,
                                    cl_int2&,
                                    float&>>
        FLOATcutting_voxel_project;
};

} // namespace CTL
