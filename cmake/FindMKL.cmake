################################################################################
#
# \file      cmake/FindMKL.cmake
# \author    J. Bakosi
# \copyright 2012-2015, Jozsef Bakosi, 2016, Los Alamos National Security, LLC.
# \brief     Find the Math Kernel Library from Intel
# \date      Thu 26 Jan 2017 02:05:50 PM MST
#
################################################################################

# Find the Math Kernel Library from Intel
#
#  MKL_FOUND - System has MKL
#  MKL_INCLUDE_DIRS - MKL include files directories
#  MKL_LIBRARIES - The MKL libraries
#  MKL_INTERFACE_LIBRARY - MKL interface library
#  MKL_SEQUENTIAL_LAYER_LIBRARY - MKL sequential layer library
#  MKL_CORE_LIBRARY - MKL core library
#
#  The environment variables MKLROOT and INTEL are used to find the library.
#  Everything else is ignored. If MKL is found "-DMKL_ILP64" is added to
#  CMAKE_C_FLAGS and CMAKE_CXX_FLAGS.
#
#  Example usage:
#
#  find_package(MKL)
#  if(MKL_FOUND)
#    target_link_libraries(TARGET ${MKL_LIBRARIES})
#  endif()

# If already in cache, be silent
if (MKL_INCLUDE_DIRS AND MKL_LIBRARIES AND MKL_INTERFACE_LIBRARY AND
    MKL_SEQUENTIAL_LAYER_LIBRARY AND MKL_CORE_LIBRARY)
  set (MKL_FIND_QUIETLY TRUE)
endif()

if(NOT BUILD_SHARED_LIBS)
  #set(COR_LIB "libmkl_core.a")
  set(COR_LIB "libmkl_rt.so")
else()
  set(COR_LIB "libmkl_rt.so")
endif()

find_path(MKL_INCLUDE_DIR NAMES mkl_lapacke.h HINTS /opt/intel/compilers_and_libraries_2019.4.243/linux/mkl/include /opt/intel/compilers_and_libraries_2019.0.117/linux/mkl/includei /opt/intel/mkl/include /usr/include/mkl $ENV{HOME}/opt/MKL/2022/mkl/2022.0.1/include)
set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
message("Found MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR}")
get_filename_component(MKL_BASE_DIR ${MKL_INCLUDE_DIR} DIRECTORY)

find_library(MKL_CORE_LIBRARY
             NAMES ${COR_LIB}
             PATHS ${MKL_BASE_DIR}/lib/intel64_lin/ /lib/x86_64-linux-gnu/ $ENV{HOME}/opt/MKL/2022/mkl/2022.0.1/lib/intel64/
             NO_DEFAULT_PATH)

set(MKL_LIBRARIES ${MKL_CORE_LIBRARY})

# Handle the QUIETLY and REQUIRED arguments and set MKL_FOUND to TRUE if
# all listed variables are TRUE.
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(MKL DEFAULT_MSG MKL_LIBRARIES MKL_INCLUDE_DIRS MKL_CORE_LIBRARY)

MARK_AS_ADVANCED(MKL_INCLUDE_DIRS MKL_LIBRARIES MKL_CORE_LIBRARY)
