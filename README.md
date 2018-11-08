# Divide and Conquer CT Projector

Using divide and conquer techniques to construct CT system matrix.


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

### [Plog](https://github.com/SergiusTheBest/plog) logger

Logger Plog is used for logging. It is licensed under the Mozilla Public License Version 2.0.

### [CLI11](https://github.com/CLIUtils/CLI11)

Comand line parser CLI11. It is licensed under 3 Clause BSD License.

### [Catch2](https://github.com/catchorg/Catch2)

Testing framework. Licensed under Boost Software License 1.0.

### [Matrix template](ssh://git@gitlab.stimulate.ovgu.de:2200/robert-frysch/Matrix-Template.git)

Manipulations with matrices by Robert Frysch.

### [CTPL](https://github.com/vit-vit/ctpl)

Threadpool library.

### [CTIOL](ssh://git@gitlab.stimulate.ovgu.de:2200/vojtech.kulvait/CTIOL.git)

Input output routines for asynchronous thread safe reading/writing CT data. The DEN format read/write is implemented.

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
