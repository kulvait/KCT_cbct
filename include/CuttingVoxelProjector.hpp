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
    CuttingVoxelProjector(float* volume, uint32_t vdimx, uint32_t vdimy, uint32_t vdimz)
    {
        this->volume = volume;
        this->vdimx = vdimx;
        this->vdimy = vdimy;
        this->vdimz = vdimz;
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
    int initializeOpenCL();

    int initializeVolumeImage();

    int project(float* projection,
                uint32_t pdimx,
                uint32_t pdimy,
                matrix::ProjectionMatrix P,
                float scalingFactor);

private:
    float* volume = nullptr;
    uint32_t vdimx, vdimy, vdimz;

    std::shared_ptr<cl::Device> device = nullptr;
    std::shared_ptr<cl::Context> context = nullptr;
    std::shared_ptr<cl::Image3D> volumeImage = nullptr;
    std::shared_ptr<cl::CommandQueue> Q = nullptr;
};

} // namespace CTL
