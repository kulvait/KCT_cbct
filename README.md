# CBCT

Cutting voxel projector and LSQR implementation for cone beam CT operator.

LSQR algorithm was implemented according to https://doi.org/10.1002/nla.611


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


### [CTIOL](https://bitbucket.org/kulvait/ctiol)

Input output routines for asynchronous thread safe reading/writing CT data. The DEN format read/write is implemented.

### [CTMAL](https://bitbucket.org/kulvait/ctmal)

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

Copyright (C) 2018-2021 Vojtěch Kulvait

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
