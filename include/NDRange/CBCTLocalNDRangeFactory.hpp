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
#include "NDRange/NDRangeHelper.hpp"

namespace KCT {

class CBCTLocalNDRangeFactory
{
public:
    CBCTLocalNDRangeFactory(uint32_t vdimx,
                            uint32_t vdimy,
                            uint32_t vdimz,
                            uint32_t maxWorkGroupSize)
        : vdimx(vdimx)
        , vdimy(vdimy)
        , vdimz(vdimz)
        , maxWorkGroupSize(maxWorkGroupSize)
    {
    }

    cl::NDRange getProjectorLocalNDRange(cl::NDRange lr = cl::NullRange, bool verbose = false) const;
    cl::NDRange getProjectorBarrierLocalNDRange(cl::NDRange lr = cl::NullRange, bool verbose = false) const;
    cl::NDRange getBackprojectorLocalNDRange(cl::NDRange lr = cl::NullRange, bool verbose = false) const;

private:
    uint32_t vdimx, vdimy, vdimz;
    uint32_t maxWorkGroupSize;
    bool isAdmissibleRange(uint32_t n0, uint32_t n1, uint32_t n2) const;
    bool isLocalRangeAdmissible(cl::NDRange& localRange) const;
    void checkLocalRange(cl::NDRange& localRange, std::string name) const;
    cl::NDRange guessProjectorLocalNDRange(bool barrierCalls) const;
    cl::NDRange guessBackprojectorLocalNDRange() const;
};

} // namespace KCT
