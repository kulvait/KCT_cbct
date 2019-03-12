cmake_minimum_required(VERSION 3.1)
project(CTRegression)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_INSTALL_PREFIX $ENV{HOME}/bin)
set(CMAKE_CXX_STANDARD 14)#Supported values are 98, 11 and 14.
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -Wextra")
message("Current value of flags is ${CMAKE_CXX_FLAGS} and debug is ${CMAKE_CXX_FLAGS_DEBUG} and release is ${CMAKE_CXX_FLAGS_RELEASE}")
#Default CMAKE_CXX_FLAGS are empty, default for CMAKE_CXX_FLAGS_DEBUG is -g and CMAKE_CXX_FLAGS_RELEASE are not empty
set(CMAKE_CXX_FLAGS "-Wall")
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

#Plog logger
include_directories(${CMAKE_SOURCE_DIR}/submodules/plog/include)

#CLI11 comand line parser library
include_directories(${CMAKE_SOURCE_DIR}/submodules/CLI11/include)

#Matrix class ssh://git@gitlab.stimulate.ovgu.de:2200/robert-frysch/Matrix-Template.git
include_directories(${CMAKE_SOURCE_DIR}/submodules/Matrix-Template)

#Thread pool management lib ctpl from https://github.com/vit-vit/ctpl
include_directories(${CMAKE_SOURCE_DIR}/submodules/ctpl)
find_package (Threads)#include pthreads

#Intel MKL
find_package(MKL)
include_directories(${MKL_INCLUDE_DIRS})

#Matplotlibcpp
include_directories(${CMAKE_SOURCE_DIR}/submodules/matplotlib-cpp)

#Python for matplotlib
find_package(PythonLibs 2.7)
include_directories(${PYTHON_INCLUDE_DIRS})

#CT input output library
FILE( GLOB CTIOL_SRC ${CMAKE_SOURCE_DIR}/submodules/CTIOL/src/*.cpp ${CMAKE_SOURCE_DIR}/submodules/CTIOL/src/ARGPARSE/*.cpp)
add_library(ctiol ${CTIOL_SRC})
include_directories(${CMAKE_SOURCE_DIR}/submodules/CTIOL/include)

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
SET(CLLINPROJECTOR_SOURCES src/CLlin-projector.cpp src/CuttingVoxelProjector.cpp)
add_executable(CLlin-projector ${CLLINPROJECTOR_SOURCES})
set_target_properties(
    CLlin-projector
    PROPERTIES 
        OUTPUT_NAME "cllin-projector"
        SUFFIX ""
)
target_link_libraries(CLlin-projector ctiol)
target_link_libraries(CLlin-projector ctmal)
target_link_libraries(CLlin-projector OpenCL::OpenCL)
target_link_libraries(CLlin-projector CTIOL_OPENCL)
target_include_directories(CLlin-projector PUBLIC ${OpenCL_INCLUDE_DIRS})
target_compile_definitions(CLlin-projector PUBLIC CL_USE_DEPRECATED_OPENCL_1_2_APIS)
target_link_libraries (CLlin-projector ${CMAKE_THREAD_LIBS_INIT})
configure_file(${CMAKE_SOURCE_DIR}/opencl/projector.cl opencl/projector.cl COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/opencl/centerVoxelProjector.cl opencl/centerVoxelProjector.cl COPYONLY)
install (TARGETS CLlin-projector RUNTIME DESTINATION .)
install(FILES ${CMAKE_SOURCE_DIR}/opencl/projector.cl DESTINATION opencl)
install(FILES ${CMAKE_SOURCE_DIR}/opencl/centerVoxelProjector.cl DESTINATION opencl)


SET(CLLINCGLS_SOURCES src/CLlin-cgls.cpp src/CGLSReconstructor.cpp)
add_executable(CLlin-cgls ${CLLINCGLS_SOURCES})
set_target_properties(
    CLlin-cgls
    PROPERTIES 
        OUTPUT_NAME "cllin-cgls"
        SUFFIX ""
)
target_link_libraries(CLlin-cgls ctiol)
target_link_libraries(CLlin-cgls ctmal)
target_link_libraries(CLlin-cgls OpenCL::OpenCL)
target_link_libraries(CLlin-cgls CTIOL_OPENCL)
target_include_directories(CLlin-cgls PUBLIC ${OpenCL_INCLUDE_DIRS})
target_compile_definitions(CLlin-cgls PUBLIC CL_USE_DEPRECATED_OPENCL_1_2_APIS)
target_link_libraries (CLlin-cgls ${CMAKE_THREAD_LIBS_INIT})
configure_file(${CMAKE_SOURCE_DIR}/opencl/projector.cl opencl/projector.cl COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/opencl/utils.cl opencl/utils.cl COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/opencl/allsources.cl opencl/allsources.cl COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/opencl/centerVoxelProjector.cl opencl/centerVoxelProjector.cl COPYONLY)
install (TARGETS CLlin-cgls RUNTIME DESTINATION .)
install(FILES ${CMAKE_SOURCE_DIR}/opencl/projector.cl DESTINATION opencl)
install(FILES ${CMAKE_SOURCE_DIR}/opencl/utils.cl DESTINATION opencl)
install(FILES ${CMAKE_SOURCE_DIR}/opencl/allsources.cl DESTINATION opencl)

SET(CLLINPERFUSION_SOURCES src/CLlin-perfusion.cpp src/CGLSPerfusionReconstructor.cpp)
add_executable(CLlin-perfusion ${CLLINPERFUSION_SOURCES})
set_target_properties(
    CLlin-perfusion
    PROPERTIES 
        OUTPUT_NAME "cllin-perfusion"
        SUFFIX ""
)
target_link_libraries(CLlin-perfusion ctiol)
target_link_libraries(CLlin-perfusion ctmal)
target_link_libraries(CLlin-perfusion OpenCL::OpenCL)
target_link_libraries(CLlin-perfusion CTIOL_OPENCL)
target_include_directories(CLlin-perfusion PUBLIC ${OpenCL_INCLUDE_DIRS})
target_compile_definitions(CLlin-perfusion PUBLIC CL_USE_DEPRECATED_OPENCL_1_2_APIS)
target_link_libraries (CLlin-perfusion ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(CLlin-perfusion ${PYTHON_LIBRARIES})
configure_file(${CMAKE_SOURCE_DIR}/opencl/projector.cl opencl/projector.cl COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/opencl/utils.cl opencl/utils.cl COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/opencl/allsources.cl opencl/allsources.cl COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/opencl/centerVoxelProjector.cl opencl/centerVoxelProjector.cl COPYONLY)
install (TARGETS CLlin-perfusion RUNTIME DESTINATION .)
install(FILES ${CMAKE_SOURCE_DIR}/opencl/projector.cl DESTINATION opencl)
install(FILES ${CMAKE_SOURCE_DIR}/opencl/utils.cl DESTINATION opencl)
install(FILES ${CMAKE_SOURCE_DIR}/opencl/allsources.cl DESTINATION opencl)

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
#Documentation target
find_package(Doxygen)
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
