#pragma once

// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <CL/cl.hpp>
#include <iostream>

// Internal libraries
#include "MATRIX/ProjectionMatrix.hpp"
#include "OPENCL/OpenCLManager.hpp"
#include "rawop.h"
#include "stringFormatter.h"

namespace CTL {

class CuttingVoxelProjector
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
     * @param centerVoxelProjector Use center voxel projector istead of cutting voxels.
     */
    CuttingVoxelProjector(float* volume,
                          uint32_t vdimx,
                          uint32_t vdimy,
                          uint32_t vdimz,
                          std::string xpath,
                          bool debug,
                          bool centerVoxelProjector)
        : volume(volume)
        , vdimx(vdimx)
        , vdimy(vdimy)
        , vdimz(vdimz)
        , xpath(xpath)
        , debug(debug)
        , centerVoxelProjector(centerVoxelProjector)
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

    int initializeVolumeImage();

    int project(float* projection,
                uint32_t pdimx,
                uint32_t pdimy,
                matrix::ProjectionMatrix P,
                float scalingFactor);

private:
    float* volume = nullptr;
    uint32_t vdimx, vdimy, vdimz;
    std::string xpath; // Path where the program executes
    bool debug;
    bool centerVoxelProjector = false;

    std::shared_ptr<cl::Device> device = nullptr;
    std::shared_ptr<cl::Context> context = nullptr;
    std::shared_ptr<cl::Image3D> volumeImage = nullptr;
    std::shared_ptr<cl::CommandQueue> Q = nullptr;
    std::shared_ptr<cl::Buffer> volumeBuffer = nullptr;

    std::shared_ptr<cl::make_kernel<cl::Buffer&,
                                    cl::Buffer&,
                                    unsigned int&,
                                    cl_double16&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int4&,
                                    cl_double3&,
                                    cl_int2&,
                                    float&>>
        projector;
};

} // namespace CTL
