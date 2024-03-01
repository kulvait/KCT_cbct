#include "NDRange/PBCT2DLocalNDRangeFactory.hpp"

namespace KCT {

cl::NDRange PBCT2DLocalNDRangeFactory::getProjectorLocalNDRange(cl::NDRange defaultRange,
                                                                bool verbose) const
{
    std::size_t dim = defaultRange.dimensions();
    cl::NDRange out;
    bool barrier = false;
    bool guess = false;
    if(dim != 0)
    {
        if(dim == 3)
        // Input parse is 3D vector for FIX
        {
            defaultRange = cl::NDRange(defaultRange[0], defaultRange[1]);
            dim = 2;
        }
        if(dim == 2 && defaultRange[0] == 0 && defaultRange[1] == 0)
        // Special value (0, 0) returns NullRange
        {
            out = cl::NullRange;
        } else if(dim == 2 && defaultRange[0] == 0 && defaultRange[1] == 1)
        // Special value (0, 1) guess range
        {
            guess = true;
        } else if(isLocalRangeAdmissible(defaultRange))
        {
            out = defaultRange;
        } else
        {
            LOGW << io::xprintf(
                "Provided projectorLocalNDRange=%s is invalid, setting default values!",
                NDRangeHelper::NDRangeToString(defaultRange).c_str());
            guess = true;
        }
    } else
    {
        guess = true;
    }
    if(guess)
    {
        out = guessProjectorLocalNDRange(barrier);
    }
    if(verbose)
    {
        LOGD << NDRangeHelper::NDRangeToString(out, "projectorLocalNDRange");
    }
    return out;
}

cl::NDRange PBCT2DLocalNDRangeFactory::getProjectorBarrierLocalNDRange(cl::NDRange defaultRange,
                                                                       bool verbose) const
{
    std::size_t dim = defaultRange.dimensions();
    cl::NDRange out;
    bool barrier = true;
    bool guess = false;
    if(dim != 0)
    {
        if(dim == 3)
        // Input parse is 3D vector for FIX
        {
            defaultRange = cl::NDRange(defaultRange[0], defaultRange[1]);
            dim = 2;
        }
        if(dim == 2 && defaultRange[0] == 0 && defaultRange[1] == 0)
        // Special value (0, 0) returns NullRange
        {
            out = cl::NullRange;
        } else if(dim == 2 && defaultRange[0] == 0 && defaultRange[1] == 1)
        // Special value (0, 1) guess range
        {
            guess = true;
        } else if(isLocalRangeAdmissible(defaultRange))
        {
            out = defaultRange;
        } else
        {
            LOGW << io::xprintf(
                "Provided projectorLocalNDRangeBarrier=%s is invalid, setting default values!",
                NDRangeHelper::NDRangeToString(defaultRange).c_str());
            guess = true;
        }
    } else
    {
        guess = true;
    }
    if(guess)
    {
        out = guessProjectorLocalNDRange(barrier);
    }
    if(verbose)
    {
        LOGD << NDRangeHelper::NDRangeToString(out, "projectorLocalNDRangeBarrier");
    }
    return out;
}

cl::NDRange PBCT2DLocalNDRangeFactory::getBackprojectorLocalNDRange(cl::NDRange defaultRange,
                                                                    bool verbose) const
{
    std::size_t dim = defaultRange.dimensions();
    cl::NDRange out;
    bool guess = false;
    if(dim != 0)
    {
        if(dim == 3)
        // Input parse is 3D vector for FIX
        {
            defaultRange = cl::NDRange(defaultRange[0], defaultRange[1]);
            dim = 2;
        }
        if(dim == 2 && defaultRange[0] == 0 && defaultRange[1] == 0)
        // Special value (0, 0) returns NullRange
        {
            out = cl::NullRange;
        } else if(dim == 2 && defaultRange[0] == 0 && defaultRange[1] == 1)
        // Special value (0, 1) guess range
        {
            guess = true;
        } else if(isLocalRangeAdmissible(defaultRange))
        {
            out = defaultRange;
        } else
        {
            LOGW << io::xprintf(
                "Provided backprojectorLocalNDRange=%s is invalid, setting default values!",
                NDRangeHelper::NDRangeToString(defaultRange).c_str());
            guess = true;
        }
    } else
    {
        guess = true;
    }
    if(guess)
    {
        out = guessBackprojectorLocalNDRange();
    }
    if(verbose)
    {
        LOGD << NDRangeHelper::NDRangeToString(out, "backprojectorLocalNDRange");
    }
    return out;
}

bool PBCT2DLocalNDRangeFactory::isLocalRangeAdmissible(cl::NDRange& localRange) const
{

    size_t dim = localRange.dimensions();
    if(dim == 0)
    {
        return false;
    } else if(dim == 2)
    {
        uint64_t totalSize = localRange[0] * localRange[1];
        if(totalSize == 0)
        {
            return false;
        }
        if(totalSize > maxWorkGroupSize)
        {
            return false;
        }
        if(vdimx % localRange[0] != 0)
        {
            return false;
        }
        if(vdimy % localRange[1] != 0)
        {
            return false;
        }
        return true;
    } else
    {
        return false;
    }
}

void PBCT2DLocalNDRangeFactory::checkLocalRange(cl::NDRange& localRange, std::string name) const
{

    size_t dim = localRange.dimensions();
    std::string ERR;
    if(dim == 0)
    {
        LOGD << io::xprintf("%s = cl::NDRange()", name.c_str());
    } else if(dim == 2)
    {
        uint64_t totalSize = localRange[0] * localRange[1];
        if(totalSize > maxWorkGroupSize)
        {
            ERR = io::xprintf("%s has total size %d exceeding maxWorkGroupSize=%d!", name.c_str(),
                              totalSize, maxWorkGroupSize);
            KCTERR(ERR);
        }
        if(totalSize == 0)
        {
            ERR = io::xprintf("There is 0 in %s definition!", name.c_str());
            KCTERR(ERR);
        }
        if(vdimx % localRange[0] != 0)
        {
            ERR = io::xprintf("%s vdimx %% localRange[0] != 0 %d %% %d!=0", name.c_str(), vdimx,
                              localRange[0]);
            KCTERR(ERR);
        }
        if(vdimy % localRange[1] != 0)
        {
            ERR = io::xprintf("%s vdimy %% localRange[1] != 0 %d %% %d!=0", name.c_str(), vdimy,
                              localRange[1]);
            KCTERR(ERR);
        }
        LOGD << io::xprintf("%s = cl::NDRange(%d, %d)", name.c_str(), localRange[0], localRange[1]);
    } else
    {
        ERR = io::xprintf("%s has dimension %d but it shall be 2!", name.c_str(), dim);
        KCTERR(ERR);
    }
}

bool PBCT2DLocalNDRangeFactory::isAdmissibleRange(uint32_t n0, uint32_t n1) const
{
    uint64_t totalSize = n0 * n1;
    if(totalSize > maxWorkGroupSize)
    {
        return false;
    }
    if(n0 == 0 || vdimx % n0 != 0)
    {
        return false;
    }
    if(n1 == 0 || vdimy % n1 != 0)
    {
        return false;
    }
    return true;
}

cl::NDRange PBCT2DLocalNDRangeFactory::guessProjectorLocalNDRange(bool barrierCalls) const
{

    cl::NDRange projectorLocalNDRange;
    if(barrierCalls)
    {
        if(isAdmissibleRange(128, 4))
        {
            projectorLocalNDRange = cl::NDRange(128, 4); // 17.02 Barrier
        } else if(isAdmissibleRange(256, 4))
        {
            projectorLocalNDRange = cl::NDRange(256, 4); // 17.18 Barrier
        } else if(isAdmissibleRange(128, 8))
        {
            projectorLocalNDRange = cl::NDRange(128, 8); // 18.18 Barrier
        } else if(isAdmissibleRange(64, 16))
        {
            projectorLocalNDRange = cl::NDRange(64, 16); // 20.54 Barrier
        } else if(isAdmissibleRange(32, 16))
        {
            projectorLocalNDRange = cl::NDRange(32, 16); // 21.54 Barrier
        } else if(isAdmissibleRange(32, 32))
        {
            projectorLocalNDRange = cl::NDRange(32, 32); // 23.29 Barrier
        } else if(isAdmissibleRange(16, 16))
        {
            projectorLocalNDRange = cl::NDRange(16, 16); // 31.98 Barrier
        } else if(isAdmissibleRange(8, 8))
        {
            projectorLocalNDRange = cl::NDRange(8, 8); //
        } else if(vdimx % 4 == 0 && vdimy % 4 == 0 && 4 * 4 <= maxWorkGroupSize)
        {
            projectorLocalNDRange = cl::NDRange(4, 4); //
        } else
        {
            projectorLocalNDRange = cl::NullRange; // 24.71 Barrier
        }
    } else
    {
        if(vdimx % 32 == 0 && vdimy % 32 == 0 && 32 * 32 <= maxWorkGroupSize)
        {
            projectorLocalNDRange = cl::NDRange(32, 32); // 3.8 Barrier
        } else if(vdimx % 16 == 0 && vdimy % 16 == 0 && 16 * 16 <= maxWorkGroupSize)
        {
            projectorLocalNDRange = cl::NDRange(16, 16); // 3.8 Barrier
        } else if(vdimx % 8 == 0 && vdimy % 8 == 0 && 8 * 8 <= maxWorkGroupSize)
        {
            projectorLocalNDRange = cl::NDRange(8, 8); // 10.9 Barrier
        } else if(vdimx % 4 == 0 && vdimy % 4 == 0 && 4 * 4 <= maxWorkGroupSize)
        {
            projectorLocalNDRange = cl::NDRange(4, 4); // 32.5 Barrier
        } else
        {
            projectorLocalNDRange = cl::NullRange;
        }
    }
    return projectorLocalNDRange;
}

cl::NDRange PBCT2DLocalNDRangeFactory::guessBackprojectorLocalNDRange() const
{
    cl::NDRange backprojectorLocalNDRange;
    if(isAdmissibleRange(32, 8))
    {
        backprojectorLocalNDRange = cl::NDRange(32, 2); // 6.72
    } else
    {
        backprojectorLocalNDRange = cl::NullRange; // 6.71
    }
    return backprojectorLocalNDRange;
}

} // namespace KCT
