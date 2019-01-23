#pragma once

// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <iostream>
#include <CL/cl.hpp>

//Internal libraries
#include "stringFormatter.h"

namespace CTL {

class CuttingVoxelProjector
{
public:

/** Initializes OpenCL.
*
* Initialization is done via C++ layer that works also with OpenCL 1.1.
* 
*
* @return 
* @see [OpenCL C++ manual](https://www.khronos.org/registry/OpenCL/specs/opencl-cplusplus-1.1.pdf)
* @see [OpenCL C++ tutorial](http://simpleopencl.blogspot.com/2013/06/tutorial-simple-start-with-opencl-and-c.html)
*/
int initializeOpenCL();
	
};

} // namespace CTL
