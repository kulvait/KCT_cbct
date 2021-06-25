#pragma once

// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <CL/cl.hpp>
#include <chrono>
#include <ctime>
#include <functional>
#include <iostream>
#include <numeric>

// Internal libraries
#include "DEN/DenProjectionMatrixReader.hpp"
#include "MATRIX/ProjectionMatrix.hpp"
#include "OPENCL/OpenCLManager.hpp"
#include "rawop.h"
#include "stringFormatter.h"

namespace CTL {

class Kniha
{
public:
    Kniha() {}

    /** Initializes OpenCL.
     * @brief Initialize OpenCL engine. Before calling this function initialize required objects by
     * calling CLINCLUDE functions. These sources will be included as non null functors.
     *
     * @param xpath
     * @param platformId
     *
     * @return
     * @see [OpenCL C++
     * manual](https://www.khronos.org/registry/OpenCL/specs/opencl-cplusplus-1.1.pdf)
     * @see [OpenCL C++
     * tutorial](http://simpleopencl.blogspot.com/2013/06/tutorial-simple-start-with-opencl-and-c.html)
     */
    int initializeOpenCL(uint32_t platformId,
                         uint32_t* deviceIds,
                         uint32_t deviceIdsLength,
                         std::string xpath,
                         bool debug,
                         bool relaxed);

    bool isOpenCLInitialized() { return openCLInitialized; }

    void CLINCLUDEbackprojector();
    void CLINCLUDEbackprojector_minmax();
    void CLINCLUDEbackprojector_sidon();
    void CLINCLUDEbackprojector_tt();
    void CLINCLUDEcenterVoxelProjector();
    void CLINCLUDEinclude();
    void CLINCLUDEjacobiPreconditionedBackprojector();
    void CLINCLUDEjacobiPreconditionedProjector();
    void CLINCLUDEprecomputeJacobiPreconditioner();
    void CLINCLUDEprojector();
    void CLINCLUDEprojector_cvp_barrier();
    void CLINCLUDEprojector_old();
    void CLINCLUDEprojector_sidon();
    void CLINCLUDEprojector_tt();
    void CLINCLUDErescaleProjections();
    void CLINCLUDEutils();

protected:
    const cl_float FLOATZERO = 0.0;
    const cl_double DOUBLEZERO = 0.0;
    float FLOATONE = 1.0f;

    // Functions to manipulate with buffers
    int multiplyVectorsIntoFirstVector(cl::Buffer& A, cl::Buffer& B, uint64_t size);
    int vectorA_multiple_B_equals_C(cl::Buffer& A, cl::Buffer& B, cl::Buffer& C, uint64_t size);
    int copyFloatVector(cl::Buffer& from, cl::Buffer& to, unsigned int size);
    int scaleFloatVector(cl::Buffer& v, float f, unsigned int size);
    int copyFloatVectorOffset(cl::Buffer& from,
                              unsigned int from_offset,
                              cl::Buffer& to,
                              unsigned int to_offset,
                              unsigned int size);
    int
    addIntoFirstVectorSecondVectorScaled(cl::Buffer& a, cl::Buffer& b, float f, unsigned int size);
    int
    addIntoFirstVectorScaledSecondVector(cl::Buffer& a, cl::Buffer& b, float f, unsigned int size);
    int invertFloatVector(cl::Buffer& X, unsigned int size);

    // OpenCL objects
    std::shared_ptr<cl::Platform> platform = nullptr;
    std::vector<cl::Device> devices;
    std::shared_ptr<cl::Context> context = nullptr;
    std::vector<std::shared_ptr<cl::CommandQueue>> Q;

    // OpenCL functors
    // backprojector.cl
    std::shared_ptr<cl::make_kernel<cl::Buffer&,
                                    cl::Buffer&,
                                    unsigned int&,
                                    cl_double16&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int3&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int2&,
                                    float&>>
        FLOATcutting_voxel_backproject;

    // backprojector_minmax.cl last is the dummy parameter not to segfault on Intel when debuging
    std::shared_ptr<cl::make_kernel<cl::Buffer&,
                                    cl::Buffer&,
                                    unsigned int&,
                                    cl_double16&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int3&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int2&,
                                    float&,
                                    cl_int2&>>
        FLOATcutting_voxel_minmaxbackproject;
    int algFLOATcutting_voxel_minmaxbackproject(cl::Buffer& volume,
                                                cl::Buffer& projection,
                                                unsigned int& projectionOffset,
                                                cl_double16& CM,
                                                cl_double3& sourcePosition,
                                                cl_double3& normalToDetector,
                                                cl_int3& vdims,
                                                cl_double3& voxelSizes,
                                                cl_double3& volumeCenter,
                                                cl_int2& pdims,
                                                float globalScalingMultiplier,
                                                cl::NDRange& globalRange,
                                                std::shared_ptr<cl::NDRange> localRange = nullptr,
                                                bool blocking = false);

    // backprojector_sidon.cl
    std::shared_ptr<cl::make_kernel<cl::Buffer&,
                                    cl::Buffer&,
                                    unsigned int&,
                                    cl_double16&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int3&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int2&,
                                    float&,
                                    cl_uint2&>>
        FLOATsidon_backproject;

    // backprojector_tt.cl
    std::shared_ptr<cl::make_kernel<cl::Buffer&,
                                    cl::Buffer&,
                                    unsigned int&,
                                    cl_double16&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int3&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int2&,
                                    float&>>
        FLOATta3_backproject;

    // centerVoxelProjector.cl
    std::shared_ptr<cl::make_kernel<cl::Buffer&,
                                    cl::Buffer&,
                                    unsigned int&,
                                    cl_double16&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int3&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int2&,
                                    float&>>
        FLOATcenter_voxel_project;

    // jacobiPreconditionedBackprojector.cl
    std::shared_ptr<cl::make_kernel<cl::Buffer&,
                                    cl::Buffer&,
                                    cl::Buffer&,
                                    unsigned int&,
                                    cl_double16&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int3&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int2&,
                                    float&>>
        FLOATjacobiPreconditionedCutting_voxel_backproject;

    // jacobiPreconditionedProjector.cl
    std::shared_ptr<cl::make_kernel<cl::Buffer&,
                                    cl::Buffer&,
                                    cl::Buffer&,
                                    unsigned int&,
                                    cl_double16&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int3&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int2&,
                                    float&>>
        FLOATjacobiPreconditionedCutting_voxel_project;

    // precomputeJacobiPreconditioner.cl
    std::shared_ptr<cl::make_kernel<cl::Buffer&,
                                    cl_double16&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int3&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int2&,
                                    float&>>
        FLOATcutting_voxel_jacobiPreconditionerVector;

    // projector.cl
    std::shared_ptr<cl::make_kernel<cl::Buffer&,
                                    cl::Buffer&,
                                    unsigned int&,
                                    cl_double16&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int3&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int2&,
                                    float&>>
        FLOATcutting_voxel_project;

    // projector_cvp_barrier.cl
    std::shared_ptr<cl::make_kernel<cl::Buffer&,
                                    cl::Buffer&,
                                    unsigned int&,
                                    cl_double16&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int3&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int2&,
                                    float&>>
        FLOATcutting_voxel_project_barrier;
    int algFLOATcutting_voxel_project_barrier(cl::Buffer& volume,
                                             cl::Buffer& projection,
                                             unsigned int& projectionOffset,
                                             cl_double16& CM,
                                             cl_double3& sourcePosition,
                                             cl_double3& normalToDetector,
                                             cl_int3& vdims,
                                             cl_double3& voxelSizes,
                                             cl_double3& volumeCenter,
                                             cl_int2& pdims,
                                             float globalScalingMultiplier,
                                             cl::NDRange& globalRange,
                                             std::shared_ptr<cl::NDRange> localRange = nullptr,
                                             bool blocking = false);

    // projector_old.cl
    std::shared_ptr<cl::make_kernel<cl::Buffer&,
                                    cl::Buffer&,
                                    unsigned int&,
                                    cl_double16&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int3&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int2&,
                                    float&>>
        OLD_FLOATcutting_voxel_project;

    // projector_sidon.cl
    std::shared_ptr<cl::make_kernel<cl::Buffer&,
                                    cl::Buffer&,
                                    unsigned int&,
                                    cl_double16&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int3&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int2&,
                                    float&,
                                    cl_uint2&>>
        FLOATsidon_project;

    // projector_tt.cl
    std::shared_ptr<cl::make_kernel<cl::Buffer&,
                                    cl::Buffer&,
                                    unsigned int&,
                                    cl_double16&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int3&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int2&,
                                    float&>>
        FLOATta3_project;

    // rescaleProjections.cl
    std::shared_ptr<cl::make_kernel<cl::Buffer&,
                                    unsigned int&,
                                    cl_double16&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_uint2&,
                                    float&>>
        FLOATrescale_projections_cos;

    std::shared_ptr<
        cl::make_kernel<cl::Buffer&, unsigned int&, cl_uint2&, cl_double2&, cl_double2&, double&>>
        FLOATrescale_projections_exact;
    // utils.cl

    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&>>
        FLOATvector_NormSquarePartial;
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&>>
        FLOATvector_SumPartial;
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&>>
        FLOATvector_MaxPartial;
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::LocalSpaceArg&, unsigned int&>>
        FLOATvector_NormSquarePartial_barrier;
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::LocalSpaceArg&, unsigned int&>>
        FLOATvector_SumPartial_barrier;
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::LocalSpaceArg&, unsigned int&>>
        FLOATvector_MaxPartial_barrier;
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&>>
        vector_NormSquarePartial;
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&>> vector_SumPartial;

    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::LocalSpaceArg&, unsigned int&>>
        vector_NormSquarePartial_barrier;
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::LocalSpaceArg&, unsigned int&>>
        vector_SumPartial_barrier;
    std::shared_ptr<
        cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::LocalSpaceArg&, unsigned int&>>
        vector_ScalarProductPartial_barrier;
    // FLOATvector_zero
    std::shared_ptr<cl::make_kernel<cl::Buffer&>> FLOATvector_zero;
    int algFLOATvector_zero(cl::Buffer& A, uint64_t size, bool blocking = false);
    // FLOATvector_zero_infinite_values
    std::shared_ptr<cl::make_kernel<cl::Buffer&>> FLOATvector_zero_infinite_values;
    int algFLOATvector_zero_infinite_values(cl::Buffer& A, uint64_t size, bool blocking = false);
    // FLOATvector_scale
    std::shared_ptr<cl::make_kernel<cl::Buffer&, float&>> FLOATvector_scale;
    int algFLOATvector_scale(cl::Buffer& A, float c, uint64_t size, bool blocking = false);
    std::shared_ptr<cl::make_kernel<cl::Buffer&>> FLOATvector_sqrt;
    // FLOATvector_invert
    std::shared_ptr<cl::make_kernel<cl::Buffer&>> FLOATvector_invert;
    int algFLOATvector_invert(cl::Buffer& X, uint64_t size, bool blocking = false);
    // FLOATvector_invert_except_zero
    std::shared_ptr<cl::make_kernel<cl::Buffer&>> FLOATvector_invert_except_zero;
    int algFLOATvector_invert_except_zero(cl::Buffer& A, uint64_t size, bool blocking = false);
    // FLOATvector_substitute_greater_than
    std::shared_ptr<cl::make_kernel<cl::Buffer&, float&, float&>>
        FLOATvector_substitute_greater_than;
    int algFLOATvector_substitute_greater_than(
        cl::Buffer& A, float maxValue, float substitution, uint64_t size, bool blocking = false);
    // FLOATvector_substitute_lower_than
    std::shared_ptr<cl::make_kernel<cl::Buffer&, float&, float&>> FLOATvector_substitute_lower_than;
    int algFLOATvector_substitute_lower_than(
        cl::Buffer& A, float minValue, float substitution, uint64_t size, bool blocking = false);
    // FLOATvector_copy
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&>> FLOATvector_copy;
    int algFLOATvector_copy(cl::Buffer& A, cl::Buffer& B, uint64_t size, bool blocking = false);
    // FLOATvector_copy_offset
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&>>
        FLOATvector_copy_offset;
    int algFLOATvector_copy_offset(
        cl::Buffer& A, cl::Buffer& B, unsigned int offset, uint64_t size, bool blocking = false);
    // FLOATvector_copy_offsets
    std::shared_ptr<cl::make_kernel<cl::Buffer&, unsigned int&, cl::Buffer&, unsigned int&>>
        FLOATvector_copy_offsets;
    int algFLOATvector_copy_offsets(cl::Buffer& A,
                                    unsigned int oA,
                                    cl::Buffer& B,
                                    unsigned int oB,
                                    uint64_t size,
                                    bool blocking = false);
    // FLOATvector_A_equals_cB
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, float&>> FLOATvector_A_equals_cB;
    int algFLOATvector_A_equals_cB(
        cl::Buffer& A, cl::Buffer& B, float c, uint64_t size, bool blocking = false);
    // FLOATvector_A_equals_A_plus_cB
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, float&>>
        FLOATvector_A_equals_A_plus_cB;
    int algFLOATvector_A_equals_A_plus_cB(
        cl::Buffer& A, cl::Buffer& B, float c, uint64_t size, bool blocking = false);
    // FLOATvector_A_equals_Ac_plus_B
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, float&>>
        FLOATvector_A_equals_Ac_plus_B;
    int algFLOATvector_A_equals_Ac_plus_B(
        cl::Buffer& A, cl::Buffer& B, float c, uint64_t size, bool blocking = false);
    // FLOATvector_A_equals_A_times_B
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&>> FLOATvector_A_equals_A_times_B;

    int algFLOATvector_A_equals_A_times_B(cl::Buffer& A,
                                          cl::Buffer& B,
                                          uint64_t size,
                                          bool blocking = false);
    // FLOATvector_C_equals_A_times_B
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&>>
        FLOATvector_C_equals_A_times_B;
    int algFLOATvector_C_equals_A_times_B(
        cl::Buffer& A, cl::Buffer& B, cl::Buffer& C, uint64_t size, bool blocking = false);
    // FLOATvector_A_equals_A_plus_cB_offset
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, float&, unsigned int&>>
        FLOATvector_A_equals_A_plus_cB_offset;
    int algFLOATvector_A_equals_A_plus_cB_offset(cl::Buffer& A,
                                                 cl::Buffer& B,
                                                 float c,
                                                 unsigned int offset,
                                                 uint64_t size,
                                                 bool blocking = false);

    // FLOATvector_B_equals_A_plus_B_offsets
    std::shared_ptr<cl::make_kernel<cl::Buffer&, unsigned int&, cl::Buffer&, unsigned int&>>
        FLOATvector_B_equals_A_plus_B_offsets;
    int algFLOATvector_B_equals_A_plus_B_offsets(cl::Buffer& from,
                                                 unsigned int from_offset,
                                                 cl::Buffer& to,
                                                 unsigned int to_offset,
                                                 uint64_t size,
                                                 bool blocking = false);
    // FLOATvector_A_equals_A_plus_cB_offsets
    std::shared_ptr<cl::make_kernel<cl::Buffer&, unsigned int&, cl::Buffer&, unsigned int&, float&>>
        FLOATvector_A_equals_A_plus_cB_offsets;
    int algFLOATvector_A_equals_A_plus_cB_offsets(cl::Buffer& A,
                                                  unsigned int oA,
                                                  cl::Buffer& B,
                                                  unsigned int oB,
                                                  float c,
                                                  uint64_t size,
                                                  bool blocking = false);

private:
    bool openCLInitialized = false;
    void insertCLFile(std::string f);
    std::vector<std::string> CLFiles;
    std::vector<std::function<void(cl::Program)>> callbacks;
};

} // namespace CTL
