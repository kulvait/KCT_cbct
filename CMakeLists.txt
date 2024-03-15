cmake_minimum_required(VERSION 3.9)
project(CBCT)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
#When you want a different install folder
#cmake -DCMAKE_INSTALL_PREFIX:PATH=/usr/local/bin ..
IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
set(CMAKE_INSTALL_PREFIX $ENV{HOME}/KCT_bin CACHE PATH "Path to install binaries of the CBCT" FORCE)
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
set(CMAKE_CXX_STANDARD 17)#KCT C++ 17
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -Wextra")
message("Current value of flags is ${CMAKE_CXX_FLAGS} and debug is ${CMAKE_CXX_FLAGS_DEBUG} and release is ${CMAKE_CXX_FLAGS_RELEASE}")
#Default CMAKE_CXX_FLAGS are empty, default for CMAKE_CXX_FLAGS_DEBUG is -g and CMAKE_CXX_FLAGS_RELEASE are not empty
set(CMAKE_CXX_FLAGS "-Wall -Wno-ignored-attributes -Wno-deprecated-copy")
#See https://stackoverflow.com/questions/13905200/is-it-wise-to-ignore-gcc-clangs-wmissing-braces-warning
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-braces")
#Turn -Werror or -Wextra to see problems
#Set -Wno-ignored-attributes due to the OpenCL Nvidia libraries
#Is it also possible to set it target specific using target_compile_options https://www.foonathan.net/2018/10/cmake-warnings/
set(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG")
message("New value of flags is ${CMAKE_CXX_FLAGS} and debug is ${CMAKE_CXX_FLAGS_DEBUG} and release is ${CMAKE_CXX_FLAGS_RELEASE}")
#add_definitions(-DDEBUG)
set(CMAKE_BUILD_TYPE Debug)
set(BUILD_SHARED_LIBS False)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

#Header directories for the project
set(GLOBAL_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
include_directories(${GLOBAL_INCLUDE_DIR})

#OpenCL
find_package(OpenCL REQUIRED)
IF(OpenCL_FOUND)
    add_definitions("-DCL_HPP_TARGET_OPENCL_VERSION=220")
    add_definitions("-DCL_TARGET_OPENCL_VERSION=220")
#possible values 100,110,120,200,210,220,300 for 120 and older uncomment the next line    
#add_definitions("-DCL_USE_DEPRECATED_OPENCL_1_2_APIS")
    message("Found OpenCL_INCLUDE_DIRS ${OpenCL_INCLUDE_DIRS}")
    message("Found path OpenCL_LIBRARY ${OpenCL_LIBRARY} and OpenCL_LIBRARIES ${OpenCL_LIBRARIES}")
    include_directories(${OpenCL_INCLUDE_DIRS})
ENDIF()

#For OpenCL on Intel processors the following might be necessary
SET(INTELLIBDIR "/opt/kulvait/intel/opencl_compilers_and_libraries_18.1.0.015/linux/compiler/lib/intel64_lin/")
IF(EXISTS ${INTELLIBDIR})
SET(INTELLIBRARIES "/opt/kulvait/intel/opencl_compilers_and_libraries_18.1.0.015/linux/compiler/lib/intel64_lin/libOpenCL.so")
SET(OpenCL_LIBRARY "${INTELLIBDIR}")
SET(OpenCL_LIBRARIES "${INTELLIBRARIES}")
SET(OpenCL_INCLUDE_DIRS "/home/kulvait/intel/system_studio_2020/opencl/SDK/include/")
ENDIF()
include_directories(OpenCL_INCLUDE_DIRS)
message("USE OpenCL_INCLUDE_DIRS ${OpenCL_INCLUDE_DIRS}")
message("USE path OpenCL_LIBRARY ${OpenCL_LIBRARY} and OpenCL_LIBRARIES ${OpenCL_LIBRARIES}")


#target_include_directories(kct-pb2d-projector PUBLIC ${OpenCL_INCLUDE_DIRS})
#target_compile_definitions(kct-pb2d-projector PUBLIC CL_USE_DEPRECATED_OPENCL_1_2_APIS)
#target_compile_definitions(kct-pb2d-projector PUBLIC CL_TARGET_OPENCL_VERSION=120)

SET(CUDA_INCLUDE_DIR "/opt/cuda/include")
IF(EXISTS ${CUDA_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIR})
ENDIF()
#set(AMDGPUPRO_INCLUDE_DIR "/opt/amdgpu-pro/include/")
#IF(EXISTS ${AMDGPUPRO_INCLUDE_DIR})
#include_directories(${AMDGPUPRO_INCLUDE_DIR})
#ENDIF()
set(ROCMINCLUDEDIR "/opt/rocm-4.3.0/opencl/include/")
IF(EXISTS ${ROCMINCLUDEDIR})
include_directories(${ROCMINCLUDEDIR})
ENDIF()


#Plog logger include
include_directories(${CMAKE_SOURCE_DIR}/submodules/plog/include)

#CLI11 comand line parser library include
include_directories(${CMAKE_SOURCE_DIR}/submodules/CLI11/include)

#Thread pool management lib ctpl from https://github.com/vit-vit/ctpl include
include_directories(${CMAKE_SOURCE_DIR}/submodules/ctpl)
find_package (Threads)#include pthreads

#Matplotlibcpp include
include_directories(${CMAKE_SOURCE_DIR}/submodules/matplotlib-cpp)

#Python for matplotlib
#find_package(PythonLibs 2.7)
#include_directories(${PYTHON_INCLUDE_DIRS})
find_package (Python3 COMPONENTS Interpreter Development NumPy)
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Python3_NumPy_INCLUDE_DIRS})

#Intel MKL
find_package(MKL)
include_directories(${MKL_INCLUDE_DIRS})
message("Including directory ${MKL_INCLUDE_DIRS}")

#Git versioning
INCLUDE(${CMAKE_SOURCE_DIR}/submodules/gitversion/cmake.cmake)
include_directories(${CMAKE_SOURCE_DIR}/submodules/gitversion)


#CTIOL library
FILE( GLOB CTIOL_SRC ${CMAKE_SOURCE_DIR}/submodules/CTIOL/src/*.cpp ${CMAKE_SOURCE_DIR}/submodules/CTIOL/src/PROG/*.cpp)
add_library(ctiol ${CTIOL_SRC})
include_directories(${CMAKE_SOURCE_DIR}/submodules/CTIOL/include)
target_link_libraries(ctiol stdc++fs)


#CTIOL_OPENCL
IF(OpenCL_FOUND)
FILE( GLOB LIBCTIOLOPENCLSRC ${CMAKE_SOURCE_DIR}/submodules/CTIOL/src/OPENCL/*.cpp )
add_library(CTIOL_OPENCL ${LIBCTIOLOPENCLSRC})
set_target_properties(
        CTIOL_OPENCL
        PROPERTIES 
        OUTPUT_NAME "ctiol_opencl.so"
        SUFFIX ""
)
target_link_libraries(CTIOL_OPENCL OpenCL::OpenCL)
target_link_libraries(CTIOL_OPENCL ctiol)
#target_compile_definitions(CTIOL_OPENCL PUBLIC CL_USE_DEPRECATED_OPENCL_1_2_APIS)
#target_compile_definitions(CTIOL_OPENCL PUBLIC CL_TARGET_OPENCL_VERSION=120)
ENDIF()

#CTMAL library
include_directories(${CMAKE_SOURCE_DIR}/submodules/CTMAL/include)
FILE( GLOB CTMAL_SRC ${CMAKE_SOURCE_DIR}/submodules/CTMAL/src/*.cpp )
add_library(ctmal ${CTMAL_SRC})
target_link_libraries(ctmal ctiol)
target_link_libraries(ctmal ${MKL_CORE_LIBRARY})
set_target_properties(
    ctmal
    PROPERTIES 
        OUTPUT_NAME "ctmal.so"
        SUFFIX ""
)

add_custom_target(formatWebkit
./formatWebkit
WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

#####################START PROGRAM DEFS##################################
#CL files
configure_file(${CMAKE_SOURCE_DIR}/opencl/backprojector.cl opencl/backprojector.cl COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/opencl/backprojector_minmax.cl opencl/backprojector_minmax.cl COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/opencl/backprojector_sidon.cl opencl/backprojector_sidon.cl COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/opencl/backprojector_tt.cl opencl/backprojector_tt.cl COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/opencl/centerVoxelProjector.cl opencl/centerVoxelProjector.cl COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/opencl/convolution.cl opencl/convolution.cl COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/opencl/include.cl opencl/include.cl COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/opencl/jacobiPreconditionedBackprojector.cl opencl/jacobiPreconditionedBackprojector.cl COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/opencl/jacobiPreconditionedProjector.cl opencl/jacobiPreconditionedProjector.cl COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/opencl/pbct_cvp.cl opencl/pbct_cvp.cl COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/opencl/pbct_cvp_barrier.cl opencl/pbct_cvp_barrier.cl COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/opencl/pbct2d_cvp.cl opencl/pbct2d_cvp.cl COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/opencl/pbct2d_cvp_barrier.cl opencl/pbct2d_cvp_barrier.cl COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/opencl/precomputeJacobiPreconditioner.cl opencl/precomputeJacobiPreconditioner.cl COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/opencl/projector.cl opencl/projector.cl COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/opencl/projector_cvp_barrier.cl opencl/projector_cvp_barrier.cl COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/opencl/projector_old.cl opencl/projector_old.cl COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/opencl/projector_sidon.cl opencl/projector_sidon.cl COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/opencl/projector_tt.cl opencl/projector_tt.cl COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/opencl/rescaleProjections.cl opencl/rescaleProjections.cl COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/opencl/utils.cl opencl/utils.cl COPYONLY)

install(FILES ${CMAKE_SOURCE_DIR}/opencl/backprojector.cl DESTINATION opencl)
install(FILES ${CMAKE_SOURCE_DIR}/opencl/backprojector_minmax.cl DESTINATION opencl)
install(FILES ${CMAKE_SOURCE_DIR}/opencl/backprojector_sidon.cl DESTINATION opencl)
install(FILES ${CMAKE_SOURCE_DIR}/opencl/backprojector_tt.cl DESTINATION opencl)
install(FILES ${CMAKE_SOURCE_DIR}/opencl/centerVoxelProjector.cl DESTINATION opencl)
install(FILES ${CMAKE_SOURCE_DIR}/opencl/convolution.cl DESTINATION opencl)
install(FILES ${CMAKE_SOURCE_DIR}/opencl/include.cl DESTINATION opencl)
install(FILES ${CMAKE_SOURCE_DIR}/opencl/jacobiPreconditionedBackprojector.cl DESTINATION opencl)
install(FILES ${CMAKE_SOURCE_DIR}/opencl/jacobiPreconditionedProjector.cl DESTINATION opencl)
install(FILES ${CMAKE_SOURCE_DIR}/opencl/pbct_cvp.cl DESTINATION opencl)
install(FILES ${CMAKE_SOURCE_DIR}/opencl/pbct_cvp_barrier.cl DESTINATION opencl)
install(FILES ${CMAKE_SOURCE_DIR}/opencl/pbct2d_cvp.cl DESTINATION opencl)
install(FILES ${CMAKE_SOURCE_DIR}/opencl/pbct2d_cvp_barrier.cl DESTINATION opencl)
install(FILES ${CMAKE_SOURCE_DIR}/opencl/precomputeJacobiPreconditioner.cl DESTINATION opencl)
install(FILES ${CMAKE_SOURCE_DIR}/opencl/projector.cl DESTINATION opencl)
install(FILES ${CMAKE_SOURCE_DIR}/opencl/projector_cvp_barrier.cl DESTINATION opencl)
install(FILES ${CMAKE_SOURCE_DIR}/opencl/projector_old.cl DESTINATION opencl)
install(FILES ${CMAKE_SOURCE_DIR}/opencl/projector_sidon.cl DESTINATION opencl)
install(FILES ${CMAKE_SOURCE_DIR}/opencl/projector_tt.cl DESTINATION opencl)
install(FILES ${CMAKE_SOURCE_DIR}/opencl/rescaleProjections.cl DESTINATION opencl)
install(FILES ${CMAKE_SOURCE_DIR}/opencl/utils.cl DESTINATION opencl)

SET(KCTPROJECTOR_SOURCES src/kct-cb-projector.cpp src/Kniha.cpp src/alg/CuttingVoxelProjector.cpp src/CArmArguments.cpp)
add_executable(kct-projector ${KCTPROJECTOR_SOURCES})
set_target_properties(
    kct-projector
    PROPERTIES 
        OUTPUT_NAME "kct-projector"
        SUFFIX ""
)
target_link_libraries(kct-projector ctiol)
target_link_libraries(kct-projector ctmal)
target_link_libraries(kct-projector OpenCL::OpenCL)
target_link_libraries(kct-projector CTIOL_OPENCL)
#target_include_directories(kct-projector PUBLIC ${OpenCL_INCLUDE_DIRS})
#target_compile_definitions(kct-projector PUBLIC CL_USE_DEPRECATED_OPENCL_1_2_APIS)
#target_compile_definitions(kct-projector PUBLIC CL_TARGET_OPENCL_VERSION=120)
target_git_version_init(kct-projector)
target_link_libraries (kct-projector ${CMAKE_THREAD_LIBS_INIT})
install (TARGETS kct-projector RUNTIME DESTINATION .)

SET(CLLINCONVOLUTION_SOURCES src/CLlin-convolution.cpp src/Kniha.cpp src/alg/VolumeConvolutionOperator.cpp src/CArmArguments.cpp)
add_executable(CLlin-convolution ${CLLINCONVOLUTION_SOURCES})
set_target_properties(
    CLlin-convolution
    PROPERTIES 
        OUTPUT_NAME "kct-convolution"
        SUFFIX ""
)
target_link_libraries(CLlin-convolution ctiol)
target_link_libraries(CLlin-convolution ctmal)
target_link_libraries(CLlin-convolution OpenCL::OpenCL)
target_link_libraries(CLlin-convolution CTIOL_OPENCL)
target_include_directories(CLlin-convolution PUBLIC ${OpenCL_INCLUDE_DIRS})
#target_compile_definitions(CLlin-convolution PUBLIC CL_USE_DEPRECATED_OPENCL_1_2_APIS)
#target_compile_definitions(CLlin-convolution PUBLIC CL_TARGET_OPENCL_VERSION=120)
target_git_version_init(CLlin-convolution)
target_link_libraries (CLlin-convolution ${CMAKE_THREAD_LIBS_INIT})
install (TARGETS CLlin-convolution RUNTIME DESTINATION .)


SET(CLLINBACKPROJECTOR_SOURCES src/CLlin-backprojector.cpp src/Kniha.cpp src/alg/CuttingVoxelProjector.cpp src/CArmArguments.cpp)
add_executable(CLlin-backprojector ${CLLINBACKPROJECTOR_SOURCES})
set_target_properties(
    CLlin-backprojector
    PROPERTIES 
        OUTPUT_NAME "kct-backprojector"
        SUFFIX ""
)
target_link_libraries(CLlin-backprojector ctiol)
target_link_libraries(CLlin-backprojector ctmal)
target_link_libraries(CLlin-backprojector OpenCL::OpenCL)
target_link_libraries(CLlin-backprojector CTIOL_OPENCL)
#target_include_directories(CLlin-backprojector PUBLIC ${OpenCL_INCLUDE_DIRS})
#target_compile_definitions(CLlin-backprojector PUBLIC CL_USE_DEPRECATED_OPENCL_1_2_APIS)
#target_compile_definitions(CLlin-backprojector PUBLIC CL_TARGET_OPENCL_VERSION=120)
target_git_version_init(CLlin-backprojector)
target_link_libraries (CLlin-backprojector ${CMAKE_THREAD_LIBS_INIT})
install (TARGETS CLlin-backprojector RUNTIME DESTINATION .)


SET(CLLINPROJECTORVOLUMES_SOURCES src/CLlin-projector-volumes.cpp src/CArmArguments.cpp  src/Kniha.cpp src/alg/CuttingVoxelProjector.cpp)
add_executable(CLlin-projector-volumes ${CLLINPROJECTORVOLUMES_SOURCES})
set_target_properties(
    CLlin-projector-volumes
    PROPERTIES 
        OUTPUT_NAME "kct-projector-volumes"
        SUFFIX ""
)
target_link_libraries(CLlin-projector-volumes ctiol)
target_link_libraries(CLlin-projector-volumes ctmal)
target_link_libraries(CLlin-projector-volumes OpenCL::OpenCL)
target_link_libraries(CLlin-projector-volumes CTIOL_OPENCL)
#target_include_directories(CLlin-projector-volumes PUBLIC ${OpenCL_INCLUDE_DIRS})
#target_compile_definitions(CLlin-projector-volumes PUBLIC CL_USE_DEPRECATED_OPENCL_1_2_APIS)
#target_compile_definitions(CLlin-projector-volumes PUBLIC CL_TARGET_OPENCL_VERSION=120)
target_git_version_init(CLlin-projector-volumes)
target_link_libraries (CLlin-projector-volumes ${CMAKE_THREAD_LIBS_INIT})
install (TARGETS CLlin-projector-volumes RUNTIME DESTINATION .)


SET(CLLINCGLS_SOURCES src/CLlin-krylov.cpp src/Kniha.cpp src/AlgorithmsBarrierBuffers.cpp src/CArmArguments.cpp src/BaseReconstructor.cpp src/GLSQRReconstructor.cpp src/CGLSReconstructor.cpp src/PSIRTReconstructor.cpp src/OSSARTReconstructor.cpp)
add_executable(CLlin-krylov ${CLLINCGLS_SOURCES})
set_target_properties(
    CLlin-krylov
    PROPERTIES 
        OUTPUT_NAME "kct-krylov"
        SUFFIX ""
)
target_link_libraries(CLlin-krylov ctiol)
target_link_libraries(CLlin-krylov ctmal)
target_link_libraries(CLlin-krylov OpenCL::OpenCL)
target_link_libraries(CLlin-krylov CTIOL_OPENCL)
#target_link_libraries (CLlin-krylov ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(CLlin-krylov PUBLIC ${OpenCL_INCLUDE_DIRS})
#target_compile_definitions(CLlin-krylov PUBLIC CL_USE_DEPRECATED_OPENCL_1_2_APIS)
#target_compile_definitions(CLlin-krylov PUBLIC CL_TARGET_OPENCL_VERSION=120)
target_git_version_init(CLlin-krylov)
install (TARGETS CLlin-krylov RUNTIME DESTINATION .)

SET(KCTPBKRYLOV_SOURCES src/kct-pb-krylov.cpp src/Kniha.cpp src/BasePBCTReconstructor.cpp src/BasePBCTOperator.cpp src/AlgorithmsBarrierBuffers.cpp src/CArmArguments.cpp src/CGLSPBCTReconstructor.cpp src/NDRange/PBCTLocalNDRangeFactory.cpp src/NDRange/NDRangeHelper.cpp)
add_executable(kct-pb-krylov ${KCTPBKRYLOV_SOURCES})
set_target_properties(
    kct-pb-krylov
    PROPERTIES 
        OUTPUT_NAME "kct-pb-krylov"
        SUFFIX ""
)
target_link_libraries(kct-pb-krylov ctiol)
target_link_libraries(kct-pb-krylov ctmal)
target_link_libraries(kct-pb-krylov OpenCL::OpenCL)
target_link_libraries(kct-pb-krylov CTIOL_OPENCL)
#target_link_libraries (kct-pb-krylov ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(kct-pb-krylov PUBLIC ${OpenCL_INCLUDE_DIRS})
#target_compile_definitions(kct-pb-krylov PUBLIC CL_USE_DEPRECATED_OPENCL_1_2_APIS)
#target_compile_definitions(kct-pb-krylov PUBLIC CL_TARGET_OPENCL_VERSION=120)
target_git_version_init(kct-pb-krylov)
install (TARGETS kct-pb-krylov RUNTIME DESTINATION .)

SET(KCTPB2DKRYLOV_SOURCES src/kct-pb2d-krylov.cpp src/Kniha.cpp src/BasePBCT2DReconstructor.cpp src/BasePBCT2DOperator.cpp src/AlgorithmsBarrierBuffers.cpp src/CArmArguments.cpp src/CGLSPBCT2DReconstructor.cpp src/NDRange/NDRangeHelper.cpp src/NDRange/PBCT2DLocalNDRangeFactory.cpp)
add_executable(kct-pb2d-krylov ${KCTPB2DKRYLOV_SOURCES})
set_target_properties(
    kct-pb2d-krylov
    PROPERTIES 
        OUTPUT_NAME "kct-pb2d-krylov"
        SUFFIX ""
)
target_link_libraries(kct-pb2d-krylov ctiol)
target_link_libraries(kct-pb2d-krylov ctmal)
target_link_libraries(kct-pb2d-krylov OpenCL::OpenCL)
target_link_libraries(kct-pb2d-krylov CTIOL_OPENCL)
#target_link_libraries (kct-pb2d-krylov ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(kct-pb2d-krylov PUBLIC ${OpenCL_INCLUDE_DIRS})
#target_compile_definitions(kct-pb2d-krylov PUBLIC CL_USE_DEPRECATED_OPENCL_1_2_APIS)
#target_compile_definitions(kct-pb2d-krylov PUBLIC CL_TARGET_OPENCL_VERSION=120)
target_git_version_init(kct-pb2d-krylov)
install (TARGETS kct-pb2d-krylov RUNTIME DESTINATION .)

SET(CLLINCGLS_SOURCES src/kct-pb-projector.cpp src/Kniha.cpp src/AlgorithmsBarrierBuffers.cpp src/CArmArguments.cpp src/alg/ParallelBeamProjector.cpp src/BasePBCTOperator.cpp src/alg/PartialParallelBeamProjector.cpp src/PartialPBCTOperator.cpp src/NDRange/PBCTLocalNDRangeFactory.cpp src/NDRange/NDRangeHelper.cpp)
add_executable(kct-pb-projector ${CLLINCGLS_SOURCES})
set_target_properties(
    kct-pb-projector
    PROPERTIES 
        OUTPUT_NAME "kct-pb-projector"
        SUFFIX ""
)
target_link_libraries(kct-pb-projector ctiol)
target_link_libraries(kct-pb-projector ctmal)
target_link_libraries(kct-pb-projector OpenCL::OpenCL)
target_link_libraries(kct-pb-projector CTIOL_OPENCL)
target_link_libraries (kct-pb-projector ${CMAKE_THREAD_LIBS_INIT})
#target_include_directories(kct-pb-projector PUBLIC ${OpenCL_INCLUDE_DIRS})
#target_compile_definitions(kct-pb-projector PUBLIC CL_USE_DEPRECATED_OPENCL_1_2_APIS)
#target_compile_definitions(kct-pb-projector PUBLIC CL_TARGET_OPENCL_VERSION=120)
target_git_version_init(kct-pb-projector)
install (TARGETS kct-pb-projector RUNTIME DESTINATION .)

SET(PB2DPROJECTOR_SOURCES src/kct-pb2d-projector.cpp src/Kniha.cpp src/AlgorithmsBarrierBuffers.cpp src/CArmArguments.cpp src/alg/PartialParallelBeam2DProjector.cpp src/PartialPBCT2DOperator.cpp src/NDRange/PBCT2DLocalNDRangeFactory.cpp src/NDRange/NDRangeHelper.cpp)
add_executable(kct-pb2d-projector ${PB2DPROJECTOR_SOURCES})
set_target_properties(
    kct-pb2d-projector
    PROPERTIES 
        OUTPUT_NAME "kct-pb2d-projector"
        SUFFIX ""
)
target_link_libraries(kct-pb2d-projector ctiol)
target_link_libraries(kct-pb2d-projector ctmal)
target_link_libraries(kct-pb2d-projector OpenCL::OpenCL)
target_link_libraries(kct-pb2d-projector CTIOL_OPENCL)
target_link_libraries (kct-pb2d-projector ${CMAKE_THREAD_LIBS_INIT})
#target_include_directories(kct-pb2d-projector PUBLIC ${OpenCL_INCLUDE_DIRS})
#target_compile_definitions(kct-pb2d-projector PUBLIC CL_USE_DEPRECATED_OPENCL_1_2_APIS)
#target_compile_definitions(kct-pb2d-projector PUBLIC CL_TARGET_OPENCL_VERSION=120)
target_git_version_init(kct-pb2d-projector)
install (TARGETS kct-pb2d-projector RUNTIME DESTINATION .)

SET(CLLINCGLS_SOURCES src/kct-pb-backprojector.cpp src/Kniha.cpp src/AlgorithmsBarrierBuffers.cpp src/CArmArguments.cpp src/BaseReconstructor.cpp src/GLSQRReconstructor.cpp src/CGLSReconstructor.cpp src/PSIRTReconstructor.cpp src/OSSARTReconstructor.cpp src/alg/ParallelBeamProjector.cpp src/BasePBCTOperator.cpp src/NDRange/PBCTLocalNDRangeFactory.cpp src/NDRange/NDRangeHelper.cpp)
add_executable(kct-pb-backprojector ${CLLINCGLS_SOURCES})
set_target_properties(
    kct-pb-backprojector
    PROPERTIES 
        OUTPUT_NAME "kct-pb-backprojector"
        SUFFIX ""
)
target_link_libraries(kct-pb-backprojector ctiol)
target_link_libraries(kct-pb-backprojector ctmal)
target_link_libraries(kct-pb-backprojector OpenCL::OpenCL)
target_link_libraries(kct-pb-backprojector CTIOL_OPENCL)
#target_link_libraries (kct-pb-backprojector ${CMAKE_THREAD_LIBS_INIT})
#target_include_directories(kct-pb-backprojector PUBLIC ${OpenCL_INCLUDE_DIRS})
#target_compile_definitions(kct-pb-backprojector PUBLIC CL_USE_DEPRECATED_OPENCL_1_2_APIS)
#target_compile_definitions(kct-pb-backprojector PUBLIC CL_TARGET_OPENCL_VERSION=120)
target_git_version_init(kct-pb-backprojector)
install (TARGETS kct-pb-backprojector RUNTIME DESTINATION .)

SET(PB2D_BACKPROJECTOR_SOURCES src/kct-pb2d-backprojector.cpp src/Kniha.cpp src/AlgorithmsBarrierBuffers.cpp src/CArmArguments.cpp src/BasePBCT2DReconstructor.cpp src/BasePBCT2DOperator.cpp src/CGLSPBCT2DReconstructor.cpp src/NDRange/NDRangeHelper.cpp src/NDRange/PBCT2DLocalNDRangeFactory.cpp)
add_executable(kct-pb2d-backprojector ${PB2D_BACKPROJECTOR_SOURCES})
set_target_properties(
    kct-pb2d-backprojector
    PROPERTIES 
        OUTPUT_NAME "kct-pb2d-backprojector"
        SUFFIX ""
)
target_link_libraries(kct-pb2d-backprojector ctiol)
target_link_libraries(kct-pb2d-backprojector ctmal)
target_link_libraries(kct-pb2d-backprojector OpenCL::OpenCL)
target_link_libraries(kct-pb2d-backprojector CTIOL_OPENCL)
target_git_version_init(kct-pb2d-backprojector)
install (TARGETS kct-pb2d-backprojector RUNTIME DESTINATION .)

SET(CLLINJACOBIGLSQR_SOURCES src/JacobiGLSQRReconstructor.cpp src/CLlin-jacobiglsqr.cpp)
add_executable(CLlin-jacobiglsqr ${CLLINJACOBIGLSQR_SOURCES})
set_target_properties(
    CLlin-jacobiglsqr
    PROPERTIES 
        OUTPUT_NAME "kct-jacobiglsqr"
        SUFFIX ""
)
target_link_libraries(CLlin-jacobiglsqr ctiol)
target_link_libraries(CLlin-jacobiglsqr ctmal)
target_link_libraries(CLlin-jacobiglsqr OpenCL::OpenCL)
target_link_libraries(CLlin-jacobiglsqr CTIOL_OPENCL)
#target_include_directories(CLlin-jacobiglsqr PUBLIC ${OpenCL_INCLUDE_DIRS})
#target_compile_definitions(CLlin-jacobiglsqr PUBLIC CL_USE_DEPRECATED_OPENCL_1_2_APIS)
#target_compile_definitions(CLlin-jacobiglsqr PUBLIC CL_TARGET_OPENCL_VERSION=120)
target_link_libraries (CLlin-jacobiglsqr ${CMAKE_THREAD_LIBS_INIT})
target_git_version_init(CLlin-jacobiglsqr)
install (TARGETS CLlin-jacobiglsqr RUNTIME DESTINATION .)

SET(CLLINPERFUSION_SOURCES src/CLlin-perfusion.cpp src/Kniha.cpp src/AlgorithmsBarrierBuffers.cpp src/Perfusion/BasePerfusionReconstructor.cpp src/Perfusion/CGLSPerfusionReconstructor.cpp src/Perfusion/GLSQRPerfusionReconstructor.cpp src/CArmArguments.cpp)
add_executable(CLlin-perfusion ${CLLINPERFUSION_SOURCES})
set_target_properties(
    CLlin-perfusion
    PROPERTIES 
        OUTPUT_NAME "kct-perfusion"
        SUFFIX ""
)
target_link_libraries(CLlin-perfusion ctiol)
target_link_libraries(CLlin-perfusion ctmal)
target_link_libraries(CLlin-perfusion OpenCL::OpenCL)
target_link_libraries(CLlin-perfusion CTIOL_OPENCL)
#target_include_directories(CLlin-perfusion PUBLIC ${OpenCL_INCLUDE_DIRS})
#target_compile_definitions(CLlin-perfusion PUBLIC CL_USE_DEPRECATED_OPENCL_1_2_APIS)
#target_compile_definitions(CLlin-perfusion PUBLIC CL_TARGET_OPENCL_VERSION=120)
target_git_version_init(CLlin-perfusion)
target_link_libraries (CLlin-perfusion ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(CLlin-perfusion ${Python3_LIBRARIES})
install (TARGETS CLlin-perfusion RUNTIME DESTINATION .)

SET(CLLINPERFUSIONPROJECTOR_SOURCES src/CLlin-perfusion-projector.cpp src/CArmArguments.cpp src/Kniha.cpp src/AlgorithmsBarrierBuffers.cpp src/Perfusion/BasePerfusionReconstructor.cpp src/Perfusion/PerfusionOperator.cpp)
add_executable(CLlin-perfusion-projector ${CLLINPERFUSIONPROJECTOR_SOURCES})
set_target_properties(
    CLlin-perfusion-projector
    PROPERTIES 
        OUTPUT_NAME "kct-perfusion-projector"
        SUFFIX ""
)
target_link_libraries(CLlin-perfusion-projector ctiol)
target_link_libraries(CLlin-perfusion-projector ctmal)
target_link_libraries(CLlin-perfusion-projector OpenCL::OpenCL)
target_link_libraries(CLlin-perfusion-projector CTIOL_OPENCL)
#target_include_directories(CLlin-perfusion-projector PUBLIC ${OpenCL_INCLUDE_DIRS})
#target_compile_definitions(CLlin-perfusion-projector PUBLIC CL_USE_DEPRECATED_OPENCL_1_2_APIS)
#target_compile_definitions(CLlin-perfusion-projector PUBLIC CL_TARGET_OPENCL_VERSION=120)
target_git_version_init(CLlin-perfusion-projector)
target_link_libraries (CLlin-perfusion-projector ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(CLlin-perfusion-projector ${Python3_LIBRARIES})
install (TARGETS CLlin-perfusion-projector RUNTIME DESTINATION .)

SET(CTMATRIX_SOURCES src/CTMatrix.cpp)
add_executable(ctmatrix ${CTMATRIX_SOURCES})
set_target_properties(
    ctmatrix
    PROPERTIES 
        OUTPUT_NAME "alg-ctmatrix"
        SUFFIX ""
)
target_link_libraries(ctmatrix ctiol)
target_link_libraries(ctmatrix ctmal)
target_link_libraries (ctmatrix ${CMAKE_THREAD_LIBS_INIT})
install (TARGETS ctmatrix RUNTIME DESTINATION .)

SET(CTMATRIXEXACT_SOURCES src/linalg-exactmatrix.cpp)
add_executable(exactmatrix ${CTMATRIXEXACT_SOURCES})
set_target_properties(
    exactmatrix
    PROPERTIES 
        OUTPUT_NAME "linalg-exactmatrix"
        SUFFIX ""
)
target_link_libraries(exactmatrix ctiol)
target_link_libraries(exactmatrix ctmal)
target_link_libraries (exactmatrix ${CMAKE_THREAD_LIBS_INIT})
install (TARGETS exactmatrix RUNTIME DESTINATION .)

SET(MRG_SOURCES src/mergeFiles.cpp)
add_executable(dacmerge ${MRG_SOURCES})
set_target_properties(
    dacmerge
    PROPERTIES 
        OUTPUT_NAME "alg-merge"
        SUFFIX ""
)
target_link_libraries(dacmerge ctiol)
target_link_libraries(dacmerge ctmal)
target_link_libraries (dacmerge ${CMAKE_THREAD_LIBS_INIT})
#target_link_libraries(dacprojector /usr/lib/liblapacke.so)
#target_link_libraries(dacprojector ${MKL_CORE_LIBRARY})
install (TARGETS dacmerge RUNTIME DESTINATION .)

#SET(SRTJJ_SOURCES src/sortFileJ.cpp)
#add_executable(sortdacj ${SRTJJ_SOURCES})
#set_target_properties(
#    sortdacj
#    PROPERTIES 
#        OUTPUT_NAME "alg-sortj"
#        SUFFIX ""
#)
#target_link_libraries(sortdacj ctiol)
#target_link_libraries(sortdacj ctmal)
#target_link_libraries (sortdacj ${CMAKE_THREAD_LIBS_INIT})
#target_link_libraries(dacprojector /usr/lib/liblapacke.so)
#target_link_libraries(dacprojector ${MKL_CORE_LIBRARY})
#install (TARGETS sortdacj RUNTIME DESTINATION .)

SET(SRT_SOURCES src/linalg-sort.cpp)
add_executable(linalgsort ${SRT_SOURCES})
set_target_properties(
    linalgsort
    PROPERTIES 
        OUTPUT_NAME "linalg-sort"
        SUFFIX ""
)
target_link_libraries(linalgsort ctiol)
target_link_libraries(linalgsort ctmal)
target_link_libraries (linalgsort ${CMAKE_THREAD_LIBS_INIT})
#target_link_libraries(dacprojector /usr/lib/liblapacke.so)
#target_link_libraries(dacprojector ${MKL_CORE_LIBRARY})
install (TARGETS linalgsort RUNTIME DESTINATION .)

SET(PROJECTF_SOURCES src/linalg-projector.cpp)
add_executable(linalgprojector ${PROJECTF_SOURCES})
set_target_properties(
    linalgprojector
    PROPERTIES 
        OUTPUT_NAME "linalg-projector"
        SUFFIX ""
)
target_link_libraries(linalgprojector ctiol)
target_link_libraries(linalgprojector ctmal)
target_link_libraries (linalgprojector ${CMAKE_THREAD_LIBS_INIT})
#target_link_libraries(dacprojector /usr/lib/liblapacke.so)
#target_link_libraries(dacprojector ${MKL_CORE_LIBRARY})
install (TARGETS linalgprojector RUNTIME DESTINATION .)

SET(PROJECTVOL_SOURCES src/projectVolume.cpp)
add_executable(volprojector ${PROJECTVOL_SOURCES})
set_target_properties(
    volprojector
    PROPERTIES 
        OUTPUT_NAME "alg-projectd"
        SUFFIX ""
)
target_link_libraries(volprojector ctiol)
target_link_libraries(volprojector ctmal)
target_link_libraries (volprojector ${CMAKE_THREAD_LIBS_INIT})
#target_link_libraries(dacprojector /usr/lib/liblapacke.so)
#target_link_libraries(dacprojector ${MKL_CORE_LIBRARY})
install (TARGETS volprojector RUNTIME DESTINATION .)


#Catch lib
include_directories(${CMAKE_SOURCE_DIR}/tests)
add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CMAKE_SOURCE_DIR}/submodules/Catch2/single_include/catch2)
FILE( GLOB TST_SRC ${CMAKE_SOURCE_DIR}/tests/*.cpp src/GLSQRReconstructor.cpp src/CGLSReconstructor.cpp src/Perfusion/BasePerfusionReconstructor.cpp src/Perfusion/GLSQRPerfusionReconstructor.cpp src/BaseReconstructor.cpp src/Kniha.cpp src/AlgorithmsBarrierBuffers.cpp)
add_executable(test_all ${TST_SRC})
target_link_libraries(test_all Catch)
target_link_libraries(test_all ctiol)
target_link_libraries(test_all ctmal)
target_link_libraries(test_all stdc++fs)
target_link_libraries(test_all OpenCL::OpenCL)
target_link_libraries(test_all CTIOL_OPENCL)
target_link_libraries(test_all ${Python3_LIBRARIES})
#target_compile_definitions(test_all PUBLIC CL_TARGET_OPENCL_VERSION=120)



#Documentation target, used workaround described in https://gitlab.kitware.com/cmake/cmake/-/issues/18708
set(DOXYGEN_EXECUTABLE "")
find_package(Doxygen QUIET)
if (DOXYGEN_FOUND)
    # set input and output files
    set(DOXYGEN_IN ${CMAKE_SOURCE_DIR}/doc/doxygen.conf.in)
    set(DOXYGEN_OUT ${CMAKE_SOURCE_DIR}/doc/doxygen.conf)

    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT})

    # note the option ALL which allows to build the docs together with the application
    add_custom_target( doxygen_doc
        ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Generating documentation with Doxygen"
        VERBATIM )

    add_custom_target( doxygen
        make
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/doc/latex
        COMMENT "Generating PDF manual"
        VERBATIM )

    add_dependencies(doxygen doxygen_doc)

else (DOXYGEN_FOUND)
  message("Doxygen need to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)
