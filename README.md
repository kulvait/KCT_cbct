# KCT CBCT

**Fast, accurate and reliable software for algebraic CT reconstruction.**

This software package provides a comprehensive suite of tools for modern CT and CBCT reconstruction, featuring highly optimized OpenCL implementations of advanced algorithms, including unpublished methods developed by the author. Initially focused on algebraic reconstruction using Krylov-based LSQR and CGLS methods, the package has since expanded to include other widely used techniques, such as OS-SIRT.

At the core of this software is the cutting voxel projector [CVP](https://doi.org/10.1016/j.jocs.2025.102573), a highly efficient projector that calculates voxel contributions to pixel projections based on volume integrals of voxel cuts. The CVP projector, implemented with OpenCL local memory optimizations, achieves remarkable computational speed, making it one of the fastest projector implementations available for algebraic reconstruction. The package also includes implementations of the TT projector and Siddon projector, providing users with a versatile range of reconstruction tools.

The software has been tested on multiple architectures, including AMD Radeon VII Vega 20 GPUs, NVIDIA GeForce RTX 2080 Ti GPUs, and Intel CPUs. Compatibility with [OpenCL 1.2](https://www.khronos.org/registry/OpenCL/specs/opencl-1.2.pdf) ensures portability, and the codebase leverages OpenCL 1.2 C++ wrappers for implementation. It could be compilled with OpenCL 2.2 even on NVIDIA GPUs. Specifically, testing has been conducted on NVIDIA's V100 and A100 GPUs, with CUDA 11.8 or CUDA 12.1 OpenCL runtime.

Supported platforms include Red Hat Enterprise Linux 9 (with GCC 11) and Debian 12 (with GCC 12), ensuring compatibility with modern Linux distributions.

Present and future development efforts focus on:

- **Preconditioning for Krylov Methods:** Developing advanced preconditioning techniques to enhance convergence rates.
- **Adaptive Voxel Grids:** Enabling computations on grids with variable voxel edge sizes to improve reconstruction quality in regions of interest.
- **Regularization Techniques:** Integrating regularization methods, such as total variation (TV) minimization, to reduce noise and artifacts in reconstructed images.

This software represents a powerful toolset for researchers and engineers in tomographic imaging, offering state-of-the-art performance across diverse hardware platforms while maintaining flexibility and scalability for ongoing advancements in the field.

## Algorithms

The following algorithms are implemented as part of this framework

### Implemented iterative CT reconstruction algorithms
CGLS algorithm with delayed residual computation as described in [Kulvait, Rose, 2021](https://doi.org/10.48550/arXiv.2110.13526)
LSQR algorithm was implemented according to [Reichel, Ye 2008](https://doi.org/10.1002/nla.611)
PSIRT algorithm was implemented according to [Gregor, Benson 2008](https://doi.org/10.1109/TMI.2008.923696)
OS-SART algorithm was implemented according to [Jiang, Wang, 2003](https://doi.org/10.1109/TIP.2003.815295)

### Implemented projectors
Cutting voxel projector was implemented according to [Kulvait et al., 2025](https://doi.org/10.1016/j.jocs.2025.102573)
TT projector was implemented according to [Long et al., 2010](https://doi.org/10.1109/TMI.2010.2050898)
Siddon projector was implemented according to [Siddon, 1985](https://doi.org/10.1118/1.595715)

## Repositories

The `KCT_cbct` package is hosted on both **GitHub** and **Bitbucket**. Clone it using the following links:

### GitHub public repository

```bash
git clone --recurse-submodules https://github.com/kulvait/KCT_cbct.git
```

### Bitbucket public repository

```bash
git clone --recurse-submodules https://bitbucket.org/kulvait/kct_cbct.git
```


## Submodules

The project relies on the following internal and external submodules, all of which are hosted on **GitHub** or **Bitbucket** and linked via the Git submodule mechanism. The project is tested against specific versions of these submodules stored in its Git repository.

To ensure compliance with the correct versions of submodules, the recommended approach is to clone the repository with its submodules using:
```bash
git clone --recurse-submodules
```
If the repository is already cloned, submodule initialization can be performed using
```bash
git submodule init
```
After fetching a new version, or if the submodules are in an inconsistent state, the following command ensures submodules are updated to the correct versions corresponding to particular checkout
```bash
git submodule update
```

### Internal Submodules

The submodules **CTIOL** and **CTMAL** are developed alongside the `KCT_cbct` package to ensure reusable functionality across multiple projects. These submodules are also utilized in the `KCT_dentk` package for projection and volume data manipulation.

**[CTIOL](https://bitbucket.org/kulvait/KCT_ctiol)**  
- **Purpose**: Handles input/output routines for asynchronous, thread-safe reading and writing of CT data.  
- Implements the **DEN** format for efficient data read/write operations.  
- **License**: GNU General Public License v3.0 (GPL-3.0).  

**[CTMAL](https://bitbucket.org/kulvait/KCT_ctmal)**  
- **Purpose**: Provides mathematical and algebraic algorithms to support CT data manipulation.  
- **License**: GNU General Public License v3.0 (GPL-3.0).  

### External Submodules

The following external submodules provide additional functionality to the `KCT_cbct` package. These dependencies include logging, command-line parsing, testing, and tools for multi-threading and visualization.

**[Plog](https://github.com/SergiusTheBest/plog)**  
- **Purpose**: Logger used for structured, thread-safe logging within the project.  
- **License**: Mozilla Public License v2.0.  

**[CLI11](https://github.com/CLIUtils/CLI11)**  
- **Purpose**: Command-line parser for handling program arguments efficiently and reliably.  
- **License**: 3-Clause BSD License.  

**[Catch2](https://github.com/catchorg/Catch2)**  
- **Purpose**: Lightweight C++ testing framework used for unit tests and validation.  
- **License**: Boost Software License 1.0.  

**[matplotlib-cpp](https://github.com/lava/matplotlib-cpp)**  
- **Purpose**: Generates plots and visualizations via bindings to Python’s Matplotlib library.  
- **License**: MIT License.  

**[gitversion](https://github.com/kulvait/gitversion)**  
- **Purpose**: Enables tools to print the exact version they were built from, based on Git commit metadata.  
  - For example, running `kct-krylov -h` may produce output like:  
    `"OpenCL implementation cone beam CT reconstruction operator. Git commit 697eecf"`.  
  - This is extremely useful for debugging as bug reports can include the precise version of the tools in use.  
- **License**: GNU General Public License v3.0 (GPL-3.0).  

**[ctpl](https://github.com/vit-vit/ctpl)**  
- **Purpose**: Implements thread-pool management for parallelism, mainly used in legacy tools like `DivideAndConquerFootprintExecutor` or `VolumeFootprintExecutor`.  
  - These legacy tools relied on CPU computation for projector evaluation, but recent tools now use `OpenCL` for parallel computing.
  - In newer tools linked against `${CMAKE_THREAD_LIBS_INIT}`, CPU threading is usually not performed via `ctpl` but using C++11/C++17 primitives, e.g. in parallel beam partial projector.  
- **License**: Apache-2.0 License.  

## Building the Project  

The project uses **CMake** for building and is designed to support a variety of dependencies, including OpenCL, Intel MKL, and optional features like CUDA and Python.  

### Features and Configuration  

- **C++ Standard**: The project requires **C++17**.  
- **Parallel Computing**: OpenCL is mandatory for GPU support, with optional CUDA integration providing additional functionality (e.g., through the OpenCL runtime).  
- **Mathematical Libraries**: Intel MKL is mandatory for optimized mathematical computations and primarily used by the **CTMAL** library for **LAPACK** primitives.  
- **Python Integration**: If Python support is enabled, development versions of Python and NumPy are required for linking.  
- **Threading**: Most tools are linked against `${CMAKE_THREAD_LIBS_INIT}`; thus, a development version of `pthread` or a similar threading library must be available on your system.  

### Dependencies  

1. **CMake**: Version **3.9** or higher.  
2. **OpenCL SDK**: Required for GPU acceleration.  
3. **Intel MKL**: Mandatory for mathematical operations (see troubleshooting section below).  
4. **CUDA Toolkit** (Optional): Provides OpenCL runtime and CUDA-specific optimizations if available.  
5. **Python** (Optional): For Python support, install development versions of both Python and NumPy.  

### Build Steps  

To build the project, follow these steps:  

```bash  
mkdir build && cd build  
cmake ..  
make  
```

### Installation
By default, the binaries are installed in `$HOME/KCT_bin`. To change the installation location, specify a custom prefix using the `CMAKE_INSTALL_PREFIX` variable
```bash
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..  
make install
```
#### Building with OpenCL

If OpenCL is detected on your system, the project automatically links against `OpenCL::OpenCL`. Ensure the appropriate libraries and include paths are specified in your environment

- **For CUDA GPUs:** Install the CUDA Toolkit, which also provides an OpenCL runtime.
- **For AMD GPUs:** Use ROCm or the AMD OpenCL SDK.
- **For Intel CPUs**: Install the Intel OpenCL runtime.

When having multiple OpenCL runtimes, additional configuration might be needed to run on proper architecture. 
Switch `-p PlatformID:DeviceID` is supported by KCT tools to run it on selected platforms and devices.

#### Python Integration
The **Python development libraries** such as `python3-dev` might be required.

#### Threading
Most tools link against `${CMAKE_THREAD_LIBS_INIT}` for multi-threading support. Ensure you have development libraries for threading installed on your system, such as `pthread` e.g., `libpthread-dev` on Linux systems.

#### Intel MKL
Intel MKL is a third-party library used primarily by the CTMAL library for LAPACK operations. While CT reconstruction itself does not directly rely on MKL, the CTMAL library must still be compiled, making Intel MKL a mandatory dependency.

**Downloading and Installing MKL**
Intel MKL is a third-party library and is not bundled with this project. It must be downloaded and installed separately from the official Intel website. On Linux systems, MKL is often installed in nonstandard locations, so additional configuration may be required for the project to locate the necessary files. MKL is part of the Intel oneAPI or oneMKL bundle, and users will need to ensure that the appropriate libraries and include files are accessible. Specifically, the project requires access to the `mkl_lapacke.h` header file and the MKL libraries, which are typically found in directories like `${MKL_BASE_DIR}/lib/intel64_lin/`.


The project provides a custom CMake module file (`cmake/FindMKL.cmake`) to help locate the MKL installation. During configuration you might need to adjust directories specified in `cmake/FindMKL.cmake` so that cmake finds MKL instalation.

**Troubleshooting MKL**

- **Header File Not Found:** If CMake cannot locate the `mkl_lapacke.h` file, ensure the `MKL_INCLUDE_DIR` is correctly set to the path containing the MKL headers. 
- **Library File Not Found:** If the MKL libraries are missing or in a nonstandard location, set the `MKL_LIBRARY_DIR` to point to the directory containing the library files. In this directory shall be `libmkl_rt.so` for dynamic linking or `libmkl_intel_lp64.a`, `libmkl_core.a` amd `libmkl_sequential.a` for static linking.
- **Compatibility:** KCT framework was tested against MKL versions 2019, 2022.0, and 2023.1.

You might wish to provide your directories directly to cmake using e.g.

```bash
cmake -DMKL_INCLUDE_DIR=/path/to/mkl/include -DMKL_LIBRARY_DIR=/path/to/mkl/lib/intel64_lin ..
make install
```

## Documentation

The project documentation is generated using Doxygen and is available both locally and online

### Online documentation

A [GitHub pages site](https://kulvait.github.io/KCT_doc/) includes examples, tutorials, and some use cases of this project. It also contains a few blog posts.

### Doxygen documentation
The Doxygen configuration file is included in the repository `doc`. Documentation can be generated locally

- The config file for doxygen was prepared runing `doxygen -g .`
- Doc files and this file can be written using [Markdown syntax](https://daringfireball.net/projects/markdown/syntax), 
- `JAVADOC_AUTOBRIEF` is set to yes to treat first line of the doc comment as a brief description, comments are of the format 
```
/**Brief description.
*
*Long description
*that might span multiple lines.
*/
```
.

## Bug Reports

When filing bug reports, please include the tool's Git version information as displayed by the -h or help flag. For example when running `kct-krylov -h` the first line of the output is
``
OpenCL implementation cone beam CT reconstruction operator.  Git commit 697eecf
``
Including this version ensures efficient debugging and resolution.

## Cite this repository

[![DOI](https://zenodo.org/badge/405354358.svg)](https://doi.org/10.5281/zenodo.14641395)

To cite this repository, you can use its Zenodo record and the following BibTeX entry.

```
@Misc{KCTCBCT2025,
  author    = {Kulvait, Vojtěch},
  title     = {Software for algebraic {CT} reconstruction {KCT CBCT}: Version 1.0},
  year      = {2025},
  copyright = {GNU General Public License v3.0},
  doi       = {10.5281/ZENODO.14641395},
  publisher = {Zenodo},
  note = {Github repository \url{https://github.com/kulvait/KCT_cbct}}
}
```

## Licensing

When there is no other licensing and/or copyright information in the source files of this project, the following apply for the source files in the directories include, src, opencl and for CMakeLists.txt file:

Copyright (C) 2018-2025 Vojtěch Kulvait

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 3 of the License.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.


This licensing applies to the direct source files in the directories include and src of this project and not for submodules.
