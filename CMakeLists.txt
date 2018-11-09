cmake_minimum_required(VERSION 3.1)
project(CTRegression)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_INSTALL_PREFIX $ENV{HOME}/bin)
set(CMAKE_CXX_STANDARD 14)#Supported values are 98, 11 and 14.
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_BUILD_TYPE Debug)
set(BUILD_SHARED_LIBS False)

#Header directories for the project
set(GLOBAL_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
include_directories(${GLOBAL_INCLUDE_DIR})

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
#find_package(MKL)
#include_directories(${MKL_INCLUDE_DIRS})

#CT input output library
FILE( GLOB CTIOL_SRC ${CMAKE_SOURCE_DIR}/submodules/CTIOL/src/*.cpp ${CMAKE_SOURCE_DIR}/submodules/CTIOL/src/ARGPARSE/*.cpp)
add_library(ctiol ${CTIOL_SRC})
include_directories(${CMAKE_SOURCE_DIR}/submodules/CTIOL/include)

#CTMAL
FILE( GLOB CTMAL_SRC ${CMAKE_SOURCE_DIR}/submodules/CTMAL/src/*.cpp )
add_library(ctmal ${CTMAL_SRC})
include_directories(${CMAKE_SOURCE_DIR}/submodules/CTMAL/include)

#file(GLOB GLOBAL_SOURCES "src/*.cpp")
SET(GLOBAL_SOURCES src/projector.cpp)

add_custom_target(formatWebkit
./formatWebkit
WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(dacprojector ${GLOBAL_SOURCES})
set_target_properties(
    dacprojector
    PROPERTIES 
        OUTPUT_NAME "dac"
        SUFFIX ""
)
target_link_libraries(dacprojector ctiol)
target_link_libraries(dacprojector ctmal)
target_link_libraries (dacprojector ${CMAKE_THREAD_LIBS_INIT})
#target_link_libraries(dacprojector /usr/lib/liblapacke.so)
#target_link_libraries(dacprojector ${MKL_CORE_LIBRARY})
install (TARGETS dacprojector RUNTIME DESTINATION .)



SET(PROJECTVOL_SOURCES src/projectVolume.cpp)
add_executable(volprojector ${PROJECTVOL_SOURCES})
set_target_properties(
    volprojector
    PROPERTIES 
        OUTPUT_NAME "volprojector"
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
