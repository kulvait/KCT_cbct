#pragma once

// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <CL/cl.hpp>
#include <iostream>

// Internal libraries
#include "MATRIX/LUDoolittleForm.hpp"
#include "MATRIX/ProjectionMatrix.hpp"
#include "MATRIX/SquareMatrix.hpp"
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
                          double vxs,
                          double vys,
                          double vzs,
                          std::string xpath,
                          bool debug,
                          bool centerVoxelProjector)
        : volume(volume)
        , vdimx(vdimx)
        , vdimy(vdimy)
        , vdimz(vdimz)
        , vxs(vxs)
        , vys(vys)
        , vzs(vzs)
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
    int updateVolumeImage();

    int project(float* projection,
                uint32_t pdimx,
                uint32_t pdimy,
                matrix::ProjectionMatrix P,
                float scalingFactor);

    int projectSiddon(float* projection,
                      uint32_t pdimx,
                      uint32_t pdimy,
                      matrix::ProjectionMatrix matrix,
                      float scalingFactor);

    double normSquare(float* projection, uint32_t pdimx, uint32_t pdimy);
    double normSquareDifference(float* projection, uint32_t pdimx, uint32_t pdimy);

private:
    float* volume = nullptr;
    uint32_t vdimx, vdimy, vdimz;
    double vxs, vys, vzs;
    std::string xpath; // Path where the program executes
    bool debug;
    bool centerVoxelProjector = false;

    std::shared_ptr<cl::Device> device = nullptr;
    std::shared_ptr<cl::Context> context = nullptr;
    std::shared_ptr<cl::Image3D> volumeImage = nullptr;
    std::shared_ptr<cl::CommandQueue> Q = nullptr;
    std::shared_ptr<cl::Buffer> volumeBuffer = nullptr;
    std::shared_ptr<cl::Buffer> projectionBuffer = nullptr;
    size_t projectionBuffer_size = 0;
    std::shared_ptr<cl::Buffer> tmpBuffer = nullptr;
    size_t tmpBuffer_size = 0;
    std::shared_ptr<cl::make_kernel<cl::Buffer&,
                                    cl::Buffer&,
                                    unsigned int&,
                                    cl_double16&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int3&,
                                    cl_double3&,
                                    cl_int2&,
                                    float&>>
        projector;
    std::shared_ptr<cl::make_kernel<cl::Buffer&,
                                    cl::Buffer&,
                                    unsigned int&,
                                    cl_double16&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int3&,
                                    cl_double3&,
                                    cl_int2&,
                                    float&,
                                    cl_uint2&>>
        projector_siddon;
    std::shared_ptr<cl::make_kernel<cl::Buffer&,
                                    unsigned int&,
                                    cl_double16&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int2&,
                                    float&>>
        scalingProjections;
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, float&>>
        FLOAT_addIntoFirstVectorSecondVectorScaled;
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&>> NormSquare;
};
const cl_float FLOATZERO = 0.0;
const cl_double DOUBLEZERO = 0.0;

} // namespace CTL
