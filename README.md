# KCT CBCT

Fast, accurate and reliable software for algebraic CT reconstruction.

This set of software tools includes OpenCL implementation of modern CT and CBCT reconstruction algorithms including unpublished algorithms by the author. Initially, the focus was on CT reconstruction using Krylov LSQR and CGLS methods. Gradually, other widely used methods such as OS-SIRT are added. Initially, the software was based on the idea of a projector that directly computes the projections of individual voxels onto pixels using the volume integrals of the voxel cuts. The author intends to publish a paper on this cutting voxel projector (CVP) in late 2021. However, the package also includes implementations of the TT projector and the Siddon projector the DD and TR projectors will be implemented in the near future. The code for the CVP projector is optimized using OpenCL local memory and is probably one of the fastest projector implementations ever for algebraic reconstruction. 

The package has been tested and is compatible with the AMD Radeon VII Vega 20 GPU and NVIDIA GeForce RTX 2080 Ti GPU. Some routines have been optimized specifically for these GPU architectures. OpenCL code conforms to the [OpenCL 1.2 specification](https://www.khronos.org/registry/OpenCL/specs/opencl-1.2.pdf) and the implementation uses C++ wrappers from OpenCL 1.2. OpenCL 2.0 is not supported due to lack of support from NVidia. 

## Algorithms

Cutting voxel projector yet to be published.

LSQR algorithm was implemented according to https://doi.org/10.1002/nla.611

CGLS algorithm with delayed residual computation as described in the proceedings of Fully3D conference 2021
Software Implementation of the Krylov Methods Based Reconstruction for the 3D Cone Beam CT Operator
Poster and extendend absract can be found in the publications directory

## Repositories

The KCT package is hosted on Bitbucket and GitHub

### GitHub public repository

```
git clone https://github.com/kulvait/KCT_cbct.git
```

### Bitbucket public repository

```
git clone https://bitbucket.org/kulvait/kct_cbct.git
```


## Submodules

Submodules lives in the submodules directory. To clone project including submodules one have to use the following commands

```
git submodule init
git submodule update
```
or use the following command when cloning repository

```
git clone --recurse-submodules
```


### [CTIOL](https://bitbucket.org/kulvait/KCT_ctiol)

Input output routines for asynchronous thread safe reading/writing CT data. The DEN format read/write is implemented.

### [CTMAL](https://bitbucket.org/kulvait/KCT_ctmal)

Mathematic/Algebraic algorithms for supporting CT data manipulation.

### [Plog](https://github.com/SergiusTheBest/plog) logger

Logger Plog is used for logging. It is licensed under the Mozilla Public License Version 2.0.

### [CLI11](https://github.com/CLIUtils/CLI11)

Comand line parser CLI11. It is licensed under 3 Clause BSD License.

### [Catch2](https://github.com/catchorg/Catch2)

Testing framework. Licensed under Boost Software License 1.0.

### [CTPL](https://github.com/vit-vit/ctpl)

Threadpool library.


## Documentation

Documentation is generated using doxygen and lives in doc directory.
First the config file for doxygen was prepared runing doxygen -g.
Doc files and this file can be written using [Markdown syntax](https://daringfireball.net/projects/markdown/syntax), JAVADOC_AUTOBRIEF is set to yes to treat first line of the doc comment as a brief description, comments are of the format 
```
/**Brief description.
*
*Long description
*thay might span multiple lines.
*/
```
.

## Licensing

When there is no other licensing and/or copyright information in the source files of this project, the following apply for the source files in the directories include and src and for CMakeLists.txt file:

Copyright (C) 2018-2022 Vojtěch Kulvait

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
