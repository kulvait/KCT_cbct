#pragma once

// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <CL/cl.hpp>
#include <chrono>
#include <ctime>
#include <iostream>

// Internal libraries
#include "NDRange/NDRangeHelper.hpp"
#include "rawop.h"
#include "stringFormatter.h"

namespace KCT {

class PBCT2DLocalNDRangeFactory
{
public:
    PBCT2DLocalNDRangeFactory(uint32_t vdimx, uint32_t vdimy, uint32_t maxWorkGroupSize)
        : vdimx(vdimx)
        , vdimy(vdimy)
        , maxWorkGroupSize(maxWorkGroupSize)
    {
    }

    cl::NDRange getProjectorLocalNDRange(cl::NDRange lr = cl::NullRange,
                                         bool verbose = false) const;
    cl::NDRange getProjectorBarrierLocalNDRange(cl::NDRange lr = cl::NullRange,
                                                bool verbose = false) const;
    cl::NDRange getBackprojectorLocalNDRange(cl::NDRange lr = cl::NullRange,
                                             bool verbose = false) const;

private:
    uint32_t vdimx, vdimy;
    uint32_t maxWorkGroupSize;

    bool isLocalRangeAdmissible(cl::NDRange& localRange) const;
    bool isAdmissibleRange(uint32_t n0, uint32_t n1) const;
    void checkLocalRange(cl::NDRange& localRange, std::string name) const;
    cl::NDRange guessProjectorLocalNDRange(bool barrierCalls) const;
    cl::NDRange guessBackprojectorLocalNDRange() const;
};

} // namespace KCT
