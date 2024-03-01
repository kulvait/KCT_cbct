#pragma once

// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <CL/cl.hpp>
#include <chrono>
#include <ctime>
#include <iostream>

// Internal libraries
#include "rawop.h"
#include "stringFormatter.h"

namespace KCT {

class NDRangeHelper
{
public:
    static cl::NDRange flipNDRange(cl::NDRange& x);
    static std::string NDRangeToString(cl::NDRange& x, std::string name="");
};

} // namespace KCT
