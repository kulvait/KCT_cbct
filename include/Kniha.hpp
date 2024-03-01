#pragma once

// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <CL/cl.hpp>
#include <chrono>
#include <ctime>
#include <experimental/filesystem>
#include <functional>
#include <iostream>
#include <numeric>

// Internal libraries
#include "DEN/DenProjectionMatrixReader.hpp"
#include "MATRIX/ProjectionMatrix.hpp"
#include "OPENCL/OpenCLManager.hpp"
#include "PROG/KCTException.hpp"
#include "rawop.h"
#include "stringFormatter.h"

namespace KCT {

class Kniha
{
public:
    Kniha() {}

    /** Initializes OpenCL.
     * @brief Initialize OpenCL engine. Before calling this function initialize required objects by
     * calling CLINCLUDE functions. These sources will be included as non null functors.
     *
     * @param xpath
     * @param platformID
     *
     * @return
     * @see [OpenCL C++
     * manual](https://www.khronos.org/registry/OpenCL/specs/opencl-cplusplus-1.1.pdf)
     * @see [OpenCL C++
     * tutorial](http://simpleopencl.blogspot.com/2013/06/tutorial-simple-start-with-opencl-and-c.html)
     */
    int initializeOpenCL(uint32_t platformID,
                                 uint32_t* deviceIds,
                                 uint32_t deviceIdsLength,
                                 std::string xpath,
                                 bool debug,
                                 bool relaxed);

    bool isOpenCLInitialized() { return openCLInitialized; }

    void addOptString(std::string option);
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
    void CLINCLUDEconvolution();
    void CLINCLUDEpbct_cvp();
    void CLINCLUDEpbct_cvp_barrier();
    void CLINCLUDEpbct2d_cvp();
    void CLINCLUDEpbct2d_cvp_barrier();

    static std::string infoString(cl_int cl_info_id);

    std::vector<std::shared_ptr<cl::CommandQueue>> getCommandQueues() { return Q; }

protected:
    const cl_float FLOATZERO = 0.0f;
    const cl_double DOUBLEZERO = 0.0;
    float FLOATONE = 1.0f;

    std::string err;

    std::vector<std::string> optstrings;

    // OpenCL objects
    std::shared_ptr<cl::Platform> platform = nullptr;
    std::vector<cl::Device> devices;
    std::shared_ptr<cl::Context> context = nullptr;
    std::vector<std::shared_ptr<cl::CommandQueue>> Q;
    // Info objects
    uint64_t localMemBytesize;
    uint32_t maxWorkGroupSize;

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
                                                cl::NDRange globalRange,
                                                cl::NDRange localRange = cl::NullRange,
                                                bool blocking = false,
                                                uint32_t QID = 0);

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
                                    cl::LocalSpaceArg&,
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
                                              unsigned int LOCALARRAYSIZE,
                                              cl::NDRange globalRange,
                                              cl::NDRange localRange = cl::NullRange,
                                              bool blocking = false,
                                              uint32_t QID = 0);

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
    // pbct2d_cvp.cl
    std::shared_ptr<cl::make_kernel<cl::Buffer&,
                                    cl::Buffer&,
                                    unsigned long&,
                                    cl_double3&,
                                    cl_int3&,
                                    cl_double3&,
                                    cl_double2&,
                                    cl_int2&,
                                    float&,
                                    int&,
                                    int&>>
        FLOAT_pbct2d_cutting_voxel_project;
    int algFLOAT_pbct2d_cutting_voxel_project(cl::Buffer& volume,
                                              cl::Buffer& projection,
                                              unsigned long projectionOffset,
                                              cl_double3& CM,
                                              cl_int3& vdims,
                                              cl_double3& voxelSizes,
                                              cl_double2& volumeCenter,
                                              cl_int2& pdims,
                                              float& globalScalingMultiplier,
                                              int& k_from,
                                              int& k_count,
                                              cl::NDRange globalRange,
                                              cl::NDRange localRange = cl::NullRange,
                                              bool blocking = false,
                                              uint32_t QID = 0);
    std::shared_ptr<cl::make_kernel<cl::Buffer&,
                                    cl::Buffer&,
                                    unsigned long&,
                                    cl_double3&,
                                    cl_int3&,
                                    cl_double3&,
                                    cl_double2&,
                                    cl_int2&,
                                    float&,
                                    int&,
                                    int&>>
        FLOAT_pbct2d_cutting_voxel_backproject;
    int algFLOAT_pbct2d_cutting_voxel_backproject(cl::Buffer& volume,
                                                  cl::Buffer& projection,
                                                  unsigned long projectionOffset,
                                                  cl_double3& CM,
                                                  cl_int3& vdims,
                                                  cl_double3& voxelSizes,
                                                  cl_double2& volumeCenter,
                                                  cl_int2& pdims,
                                                  float& globalScalingMultiplier,
                                                  int& k_from,
                                                  int& k_count,
                                                  cl::NDRange globalRange,
                                                  cl::NDRange localRange = cl::NullRange,
                                                  bool blocking = false,
                                                  uint32_t QID = 0);
    // pbct2d_cvp_barrier.cl
    std::shared_ptr<cl::make_kernel<cl::Buffer&,
                                    cl::Buffer&,
                                    cl::LocalSpaceArg&,
                                    unsigned long&,
                                    cl_double3&,
                                    cl_int3&,
                                    cl_double3&,
                                    cl_double2&,
                                    cl_int2&,
                                    float&,
                                    int&,
                                    int&>>
        FLOAT_pbct2d_cutting_voxel_project_barrier;
    int algFLOAT_pbct2d_cutting_voxel_project_barrier(cl::Buffer& volume,
                                                      cl::Buffer& projection,
                                                      unsigned long& projectionOffset,
                                                      cl_double3& CM,
                                                      cl_int3& vdims,
                                                      cl_double3& voxelSizes,
                                                      cl_double2& volumeCenter,
                                                      cl_int2& pdims,
                                                      float globalScalingMultiplier,
                                                      int& k_from,
                                                      int& k_count,
                                                      unsigned int LOCALARRAYSIZE,
                                                      cl::NDRange globalRange,
                                                      cl::NDRange localRange = cl::NullRange,
                                                      bool blocking = false,
                                                      uint32_t QID = 0);
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
    int algFLOATvector_NormSquarePartial(cl::Buffer& V,
                                         cl::Buffer& PARTIAL_OUT,
                                         unsigned int partialFrameSize,
                                         uint32_t partialFrameCount,
                                         bool blocking = false,
                                         uint32_t QID = 0);

    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&>>
        FLOATvector_SumPartial;
    int algFLOATvector_SumPartial(cl::Buffer& V,
                                  cl::Buffer& PARTIAL_OUT,
                                  unsigned int partialFrameSize,
                                  uint32_t partialFrameCount,
                                  bool blocking = false,
                                  uint32_t QID = 0);

    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&>>
        FLOATvector_MaxPartial;
    int algFLOATvector_MaxPartial(cl::Buffer& V,
                                  cl::Buffer& PARTIAL_OUT,
                                  unsigned int partialFrameSize,
                                  uint32_t partialFrameCount,
                                  bool blocking = false,
                                  uint32_t QID = 0);

    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::LocalSpaceArg&, unsigned long&>>
        FLOATvector_NormSquarePartial_barrier;
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::LocalSpaceArg&, unsigned long&>>
        FLOATvector_SumPartial_barrier;
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::LocalSpaceArg&, unsigned long&>>
        FLOATvector_MaxPartial_barrier;

    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&>>
        vector_NormSquarePartial;
    int algvector_NormSquarePartial(cl::Buffer& V,
                                    cl::Buffer& SQUARE_OUT,
                                    unsigned int partialFrameSize,
                                    uint32_t partialFrameCount,
                                    bool blocking = false,
                                    uint32_t QID = 0);

    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned int&>> vector_SumPartial;
    int algvector_SumPartial(cl::Buffer& V,
                             cl::Buffer& SUM_OUT,
                             unsigned int partialFrameSize,
                             uint32_t partialFrameCount,
                             bool blocking = false,
                             uint32_t QID = 0);

    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::LocalSpaceArg&, unsigned long&>>
        vector_NormSquarePartial_barrier;
    int algvector_NormSquarePartial_barrier(cl::Buffer& V,
                                            cl::Buffer& V_red,
                                            unsigned long& VDIM,
                                            unsigned long& VDIM_ALIGNED,
                                            uint32_t workGroupSize,
                                            bool blocking = false,
                                            uint32_t QID = 0);

    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::LocalSpaceArg&, unsigned long&>>
        vector_SumPartial_barrier;
    int algvector_SumPartial_barrier(cl::Buffer& V,
                                     cl::Buffer& V_red,
                                     unsigned long& VDIM,
                                     unsigned long& VDIM_ALIGNED,
                                     uint32_t workGroupSize,
                                     bool blocking = false,
                                     uint32_t QID = 0);
    std::shared_ptr<
        cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::LocalSpaceArg&, unsigned long&>>
        vector_ScalarProductPartial_barrier;
    // FLOATvector_zero
    std::shared_ptr<cl::make_kernel<cl::Buffer&>> FLOATvector_zero;
    int algFLOATvector_zero(cl::Buffer& A, uint64_t size, bool blocking = false, uint32_t QID = 0);
    // FLOATvector_zero_infinite_values
    std::shared_ptr<cl::make_kernel<cl::Buffer&>> FLOATvector_zero_infinite_values;
    int algFLOATvector_zero_infinite_values(cl::Buffer& A,
                                            uint64_t size,
                                            bool blocking = false,
                                            uint32_t QID = 0);
    // FLOATvector_scale
    std::shared_ptr<cl::make_kernel<cl::Buffer&, float&>> FLOATvector_scale;
    int algFLOATvector_scale(
        cl::Buffer& A, float c, uint64_t size, bool blocking = false, uint32_t QID = 0);
    std::shared_ptr<cl::make_kernel<cl::Buffer&>> FLOATvector_sqrt;
    // FLOATvector_invert
    std::shared_ptr<cl::make_kernel<cl::Buffer&>> FLOATvector_invert;
    int
    algFLOATvector_invert(cl::Buffer& X, uint64_t size, bool blocking = false, uint32_t QID = 0);
    // FLOATvector_invert_except_zero
    std::shared_ptr<cl::make_kernel<cl::Buffer&>> FLOATvector_invert_except_zero;
    int algFLOATvector_invert_except_zero(cl::Buffer& A,
                                          uint64_t size,
                                          bool blocking = false,
                                          uint32_t QID = 0);
    // FLOATvector_substitute_greater_than
    std::shared_ptr<cl::make_kernel<cl::Buffer&, float&, float&>>
        FLOATvector_substitute_greater_than;
    int algFLOATvector_substitute_greater_than(cl::Buffer& A,
                                               float maxValue,
                                               float substitution,
                                               uint64_t size,
                                               bool blocking = false,
                                               uint32_t QID = 0);
    // FLOATvector_substitute_lower_than
    std::shared_ptr<cl::make_kernel<cl::Buffer&, float&, float&>> FLOATvector_substitute_lower_than;
    int algFLOATvector_substitute_lower_than(cl::Buffer& A,
                                             float minValue,
                                             float substitution,
                                             uint64_t size,
                                             bool blocking = false,
                                             uint32_t QID = 0);
    // FLOATvector_copy
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&>> FLOATvector_copy;
    int algFLOATvector_copy(
        cl::Buffer& A, cl::Buffer& B, uint64_t size, bool blocking = false, uint32_t QID = 0);
    // FLOATvector_copy_offset
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, unsigned long&>>
        FLOATvector_copy_offset;
    int algFLOATvector_copy_offset(cl::Buffer& A,
                                   cl::Buffer& B,
                                   unsigned long offset,
                                   uint64_t size,
                                   bool blocking = false,
                                   uint32_t QID = 0);
    // FLOATvector_copy_offsets
    std::shared_ptr<cl::make_kernel<cl::Buffer&, unsigned long&, cl::Buffer&, unsigned long&>>
        FLOATvector_copy_offsets;
    int algFLOATvector_copy_offsets(cl::Buffer& A,
                                    uint64_t oA,
                                    cl::Buffer& B,
                                    uint64_t oB,
                                    uint64_t size,
                                    bool blocking = false,
                                    uint32_t QID = 0);
    // FLOATvector_A_equals_cB
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, float&>> FLOATvector_A_equals_cB;
    int algFLOATvector_A_equals_cB(cl::Buffer& A,
                                   cl::Buffer& B,
                                   float c,
                                   uint64_t size,
                                   bool blocking = false,
                                   uint32_t QID = 0);
    // FLOATvector_A_equals_A_plus_cB
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, float&>>
        FLOATvector_A_equals_A_plus_cB;
    int algFLOATvector_A_equals_A_plus_cB(cl::Buffer& A,
                                          cl::Buffer& B,
                                          float c,
                                          uint64_t size,
                                          bool blocking = false,
                                          uint32_t QID = 0);
    // FLOATvector_A_equals_Ac_plus_B
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, float&>>
        FLOATvector_A_equals_Ac_plus_B;
    int algFLOATvector_A_equals_Ac_plus_B(cl::Buffer& A,
                                          cl::Buffer& B,
                                          float c,
                                          uint64_t size,
                                          bool blocking = false,
                                          uint32_t QID = 0);
    // FLOATvector_A_equals_A_times_B
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&>> FLOATvector_A_equals_A_times_B;

    int algFLOATvector_A_equals_A_times_B(
        cl::Buffer& A, cl::Buffer& B, uint64_t size, bool blocking = false, uint32_t QID = 0);
    // FLOATvector_C_equals_A_times_B
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&>>
        FLOATvector_C_equals_A_times_B;
    int algFLOATvector_C_equals_A_times_B(cl::Buffer& A,
                                          cl::Buffer& B,
                                          cl::Buffer& C,
                                          uint64_t size,
                                          bool blocking = false,
                                          uint32_t QID = 0);
    // FLOATvector_A_equals_A_plus_cB_offset
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, float&, unsigned long&>>
        FLOATvector_A_equals_A_plus_cB_offset;
    int algFLOATvector_A_equals_A_plus_cB_offset(cl::Buffer& A,
                                                 cl::Buffer& B,
                                                 float c,
                                                 unsigned long offset,
                                                 uint64_t size,
                                                 bool blocking = false,
                                                 uint32_t QID = 0);

    // FLOATvector_B_equals_A_plus_B_offsets
    std::shared_ptr<cl::make_kernel<cl::Buffer&, unsigned long&, cl::Buffer&, unsigned long&>>
        FLOATvector_B_equals_A_plus_B_offsets;
    int algFLOATvector_B_equals_A_plus_B_offsets(cl::Buffer& from,
                                                 unsigned long from_offset,
                                                 cl::Buffer& to,
                                                 unsigned long to_offset,
                                                 uint64_t size,
                                                 bool blocking = false,
                                                 uint32_t QID = 0);
    // FLOATvector_A_equals_A_plus_cB_offsets
    std::shared_ptr<
        cl::make_kernel<cl::Buffer&, unsigned long&, cl::Buffer&, unsigned long&, float&>>
        FLOATvector_A_equals_A_plus_cB_offsets;
    int algFLOATvector_A_equals_A_plus_cB_offsets(cl::Buffer& A,
                                                  unsigned long oA,
                                                  cl::Buffer& B,
                                                  unsigned long oB,
                                                  float c,
                                                  uint64_t size,
                                                  bool blocking = false,
                                                  uint32_t QID = 0);
    // convolution.cl
    // FLOATvector_2Dconvolution3x3
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, cl_int3&, cl_float16&>>
        FLOATvector_2Dconvolution3x3ZeroBoundary;
    int algFLOATvector_2Dconvolution3x3ZeroBoundary(cl::Buffer& A,
                                                    cl::Buffer& B,
                                                    cl_int3& vdims,
                                                    cl_float16& convolutionKernel,
                                                    cl::NDRange globalRange,
                                                    cl::NDRange localRange = cl::NullRange,
                                                    bool blocking = false,
                                                    uint32_t QID = 0);

    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, cl_int3&, cl_float16&>>
        FLOATvector_2Dconvolution3x3ReflectionBoundary;
    int algFLOATvector_2Dconvolution3x3ReflectionBoundary(cl::Buffer& A,
                                                          cl::Buffer& B,
                                                          cl_int3& vdims,
                                                          cl_float16& convolutionKernel,
                                                          cl::NDRange globalRange,
                                                          cl::NDRange localRange = cl::NullRange,
                                                          bool blocking = false,
                                                          uint32_t QID = 0);
    std::shared_ptr<
        cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl_int3&, cl_float3&>>
        FLOATvector_3DconvolutionGradientSobelFeldmanReflectionBoundary;
    int algFLOATvector_3DconvolutionGradientSobelFeldmanReflectionBoundary(cl::Buffer& F,
                                                                           cl::Buffer& GX,
                                                                           cl::Buffer& GY,
                                                                           cl::Buffer& GZ,
                                                                           cl_int3& vdims,
                                                                           cl_float3& voxelSizes,
                                                                           cl::NDRange globalRange,
                                                                           cl::NDRange localRange
                                                                           = cl::NullRange,
                                                                           bool blocking = false,
                                                                           uint32_t QID = 0);
    std::shared_ptr<
        cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl_int3&, cl_float3&>>
        FLOATvector_3DconvolutionGradientSobelFeldmanZeroBoundary;
    int algFLOATvector_3DconvolutionGradientSobelFeldmanZeroBoundary(cl::Buffer& F,
                                                                     cl::Buffer& GX,
                                                                     cl::Buffer& GY,
                                                                     cl::Buffer& GZ,
                                                                     cl_int3& vdims,
                                                                     cl_float3& voxelSizes,
                                                                     cl::NDRange globalRange,
                                                                     cl::NDRange localRange
                                                                     = cl::NullRange,
                                                                     bool blocking = false,
                                                                     uint32_t QID = 0);
    std::shared_ptr<cl::make_kernel<cl::Buffer&,
                                    cl::Buffer&,
                                    cl::Buffer&,
                                    cl::Buffer&,
                                    cl_int3&,
                                    cl_float3&,
                                    int&>>
        FLOATvector_3DconvolutionGradientFarid5x5x5;
    int algFLOATvector_3DconvolutionGradientFarid5x5x5(cl::Buffer& F,
                                                       cl::Buffer& GX,
                                                       cl::Buffer& GY,
                                                       cl::Buffer& GZ,
                                                       cl_int3& vdims,
                                                       cl_float3& voxelSizes,
                                                       int reflectionBoundary,
                                                       cl::NDRange globalRange,
                                                       cl::NDRange localRange = cl::NullRange,
                                                       bool blocking = false,
                                                       uint32_t QID = 0);
    std::shared_ptr<
        cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl_int3&, cl_float3&, int&>>
        FLOATvector_2DconvolutionGradientFarid5x5;
    int algFLOATvector_2DconvolutionGradientFarid5x5(cl::Buffer& F,
                                                     cl::Buffer& GX,
                                                     cl::Buffer& GY,
                                                     cl_int3& vdims,
                                                     cl_float3& voxelSizes,
                                                     int reflectionBoundary,
                                                     cl::NDRange globalRange,
                                                     cl::NDRange localRange = cl::NullRange,
                                                     bool blocking = false,
                                                     uint32_t QID = 0);
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, cl_int3&, cl_float3&>>
        FLOATvector_3DconvolutionLaplaceZeroBoundary;
    int algFLOATvector_3DconvolutionLaplaceZeroBoundary(cl::Buffer& A,
                                                        cl::Buffer& B,
                                                        cl_int3& vdims,
                                                        cl_float3& voxelSizes,
                                                        cl::NDRange globalRange,
                                                        cl::NDRange localRange = cl::NullRange,
                                                        bool blocking = false,
                                                        uint32_t QID = 0);

    std::shared_ptr<
        cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl_int3&, cl_float3&>>
        FLOATvector_3DisotropicGradient;
    int algFLOATvector_3DisotropicGradient(cl::Buffer& F,
                                           cl::Buffer& GX,
                                           cl::Buffer& GY,
                                           cl::Buffer& GZ,
                                           cl_int3& vdims,
                                           cl_float3& voxelSizes,
                                           cl::NDRange globalRange,
                                           cl::NDRange localRange = cl::NullRange,
                                           bool blocking = false,
                                           uint32_t QID = 0);

    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl_int3&, cl_float3&>>
        FLOATvector_2DisotropicGradient;
    int algFLOATvector_2DisotropicGradient(cl::Buffer& F,
                                           cl::Buffer& GX,
                                           cl::Buffer& GY,
                                           cl_int3& vdims,
                                           cl_float3& voxelSizes,
                                           cl::NDRange globalRange,
                                           cl::NDRange localRange = cl::NullRange,
                                           bool blocking = false,
                                           uint32_t QID = 0);
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, cl_int3&, cl_float3&>>
        FLOATvector_isotropicBackDx;
    int algFLOATvector_isotropicBackDx(cl::Buffer& F,
                                       cl::Buffer& DX,
                                       cl_int3& vdims,
                                       cl_float3& voxelSizes,
                                       cl::NDRange globalRange,
                                       cl::NDRange localRange = cl::NullRange,
                                       bool blocking = false,
                                       uint32_t QID = 0);
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, cl_int3&, cl_float3&>>
        FLOATvector_isotropicBackDy;
    int algFLOATvector_isotropicBackDy(cl::Buffer& F,
                                       cl::Buffer& DY,
                                       cl_int3& vdims,
                                       cl_float3& voxelSizes,
                                       cl::NDRange globalRange,
                                       cl::NDRange localRange = cl::NullRange,
                                       bool blocking = false,
                                       uint32_t QID = 0);
    std::shared_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, cl_int3&, cl_float3&>>
        FLOATvector_isotropicBackDz;
    int algFLOATvector_isotropicBackDz(cl::Buffer& F,
                                       cl::Buffer& DZ,
                                       cl_int3& vdims,
                                       cl_float3& voxelSizes,
                                       cl::NDRange globalRange,
                                       cl::NDRange localRange = cl::NullRange,
                                       bool blocking = false,
                                       uint32_t QID = 0);

    // pbct_cvp.cl
    std::shared_ptr<cl::make_kernel<cl::Buffer&,
                                    cl::Buffer&,
                                    unsigned long&,
                                    cl_double8&,
                                    cl_int3&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int2&,
                                    float&>>
        FLOAT_pbct_cutting_voxel_project;
    int algFLOAT_pbct_cutting_voxel_project(cl::Buffer& volume,
                                            cl::Buffer& projection,
                                            unsigned long& projectionOffset,
                                            cl_double8& CM,
                                            cl_int3& vdims,
                                            cl_double3& voxelSizes,
                                            cl_double3& volumeCenter,
                                            cl_int2& pdims,
                                            float globalScalingMultiplier,
                                            cl::NDRange globalRange,
                                            cl::NDRange localRange = cl::NullRange,
                                            bool blocking = false,
                                            uint32_t QID = 0);

    std::shared_ptr<cl::make_kernel<cl::Buffer&,
                                    cl::Buffer&,
                                    unsigned long&,
                                    cl_double8&,
                                    cl_int3&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int2&,
                                    float&>>
        FLOAT_pbct_cutting_voxel_backproject;
    int algFLOAT_pbct_cutting_voxel_backproject(cl::Buffer& volume,
                                                cl::Buffer& projection,
                                                unsigned long& projectionOffset,
                                                cl_double8& CM,
                                                cl_int3& vdims,
                                                cl_double3& voxelSizes,
                                                cl_double3& volumeCenter,
                                                cl_int2& pdims,
                                                float globalScalingMultiplier,
                                                cl::NDRange globalRange,
                                                cl::NDRange localRange = cl::NullRange,
                                                bool blocking = false,
                                                uint32_t QID = 0);

    // pbct_cvp_barrier.cl
    std::shared_ptr<cl::make_kernel<cl::Buffer&,
                                    cl::Buffer&,
                                    cl::LocalSpaceArg&,
                                    unsigned long&,
                                    cl_double8&,
                                    cl_int3&,
                                    cl_double3&,
                                    cl_double3&,
                                    cl_int2&,
                                    float&>>
        FLOAT_pbct_cutting_voxel_project_barrier;
    int algFLOAT_pbct_cutting_voxel_project_barrier(cl::Buffer& volume,
                                                    cl::Buffer& projection,
                                                    unsigned long& projectionOffset,
                                                    cl_double8& CM,
                                                    cl_int3& vdims,
                                                    cl_double3& voxelSizes,
                                                    cl_double3& volumeCenter,
                                                    cl_int2& pdims,
                                                    float globalScalingMultiplier,
                                                    unsigned int LOCALARRAYSIZE,
                                                    cl::NDRange globalRange,
                                                    cl::NDRange localRange = cl::NullRange,
                                                    bool blocking = false,
                                                    uint32_t QID = 0);

private:
    /**
     * Will wait if blocking and print an error message if the status indicates error.
     *
     * @param exe
     * @param blocking
     */
    int handleKernelExecution(cl::Event exe, bool blocking, std::string& errout);
    cl::NDRange assignLocalRange(cl::NDRange localRange, cl::NDRange globalRange);
    bool openCLInitialized = false;
    void insertCLFile(std::string f);
    std::vector<std::string> CLFiles;
    std::vector<std::function<void(cl::Program)>> callbacks;
    std::shared_ptr<cl::Program> program;
};

} // namespace KCT
