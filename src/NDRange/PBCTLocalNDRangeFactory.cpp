#include "NDRange/PBCTLocalNDRangeFactory.hpp"

namespace KCT {

cl::NDRange PBCTLocalNDRangeFactory::getProjectorLocalNDRange(cl::NDRange defaultRange,
                                                              bool verbose) const
{
    std::size_t dim = defaultRange.dimensions();
    cl::NDRange out;
    bool barrier = false;
    bool guess = false;
    if(dim != 0)
    {
        if(dim == 3 && defaultRange[0] == 0 && defaultRange[1] == 0 && defaultRange[2] == 0)
        // Special value (0,0,0) returns NullRange
        {
            out = cl::NullRange;
        } else if(dim == 3 && defaultRange[0] == 0 && defaultRange[1] == 1 && defaultRange[2] == 1)
        // Special value (0, 1, 1) guess range
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

cl::NDRange PBCTLocalNDRangeFactory::getProjectorBarrierLocalNDRange(cl::NDRange defaultRange,
                                                                     bool verbose) const
{
    std::size_t dim = defaultRange.dimensions();
    cl::NDRange out;
    bool barrier = true;
    bool guess = false;
    if(dim != 0)
    {
        if(dim == 3 && defaultRange[0] == 0 && defaultRange[1] == 0 && defaultRange[2] == 0)
        // Special value (0,0,0) returns NullRange
        {
            out = cl::NullRange;
        } else if(dim == 3 && defaultRange[0] == 0 && defaultRange[1] == 1 && defaultRange[2] == 1)
        // Special value (0, 1, 1) guess range
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

cl::NDRange PBCTLocalNDRangeFactory::getBackprojectorLocalNDRange(cl::NDRange defaultRange,
                                                                  bool verbose) const
{
    std::size_t dim = defaultRange.dimensions();
    cl::NDRange out;
    bool guess = false;
    if(dim != 0)
    {
        if(dim == 3 && defaultRange[0] == 0 && defaultRange[1] == 0 && defaultRange[2] == 0)
        // Special value (0,0,0) returns NullRange
        {
            out = cl::NullRange;
        } else if(dim == 3 && defaultRange[0] == 0 && defaultRange[1] == 1 && defaultRange[2] == 1)
        // Special value (0, 1, 1) guess range
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

bool PBCTLocalNDRangeFactory::isLocalRangeAdmissible(cl::NDRange& localRange) const
{

    size_t dim = localRange.dimensions();
    if(dim == 0)
    {
        return false;
    } else if(dim == 3)
    {
        uint64_t totalSize = localRange[0] * localRange[1] * localRange[2];
        if(totalSize > maxWorkGroupSize)
        {
            return false;
        }
        if(localRange[0] == 0 || vdimx % localRange[0] != 0)
        {
            return false;
        }
        if(localRange[1] == 0 || vdimy % localRange[1] != 0)
        {
            return false;
        }
        if(localRange[2] == 0 || vdimz % localRange[2] != 0)
        {
            return false;
        }
        return true;
    } else
    {
        return false;
    }
}

void PBCTLocalNDRangeFactory::checkLocalRange(cl::NDRange& localRange, std::string name) const
{

    size_t dim = localRange.dimensions();
    std::string ERR;
    if(dim == 0)
    {
        LOGD << io::xprintf("%s = cl::NDRange()", name.c_str());
    } else if(dim == 3)
    {
        uint64_t totalSize = localRange[0] * localRange[1] * localRange[2];
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
        if(vdimz % localRange[2] != 0)
        {
            ERR = io::xprintf("%s vdimz %% localRange[2] != 0 %d %% %d!=0", name.c_str(), vdimz,
                              localRange[2]);
            KCTERR(ERR);
        }
        LOGD << io::xprintf("%s = cl::NDRange(%d, %d, %d)", name.c_str(), localRange[0],
                            localRange[1], localRange[2]);
    } else
    {
        ERR = io::xprintf("%s has dimension %d but it shall be 3!", name.c_str(), dim);
        KCTERR(ERR);
    }
}

bool PBCTLocalNDRangeFactory::isAdmissibleRange(uint32_t n0, uint32_t n1, uint32_t n2) const
{
    uint64_t totalSize = n0 * n1 * n2;
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
    if(n2 == 0 || vdimz % n2 != 0)
    {
        return false;
    }
    return true;
}

cl::NDRange PBCTLocalNDRangeFactory::guessProjectorLocalNDRange(bool barrierCalls) const
{

    cl::NDRange out;
    if(barrierCalls)
    {
        if(isAdmissibleRange(128, 2, 1))
        {
            out = cl::NDRange(128, 2, 1); // 60.6 Barrier
        } else
        {
            out = cl::NullRange; // 66.37 Barrier
        }
    } else
    {
        if(isAdmissibleRange(4, 64, 1))
        {
            out = cl::NDRange(4, 64, 1);
        } else if(isAdmissibleRange(16, 16, 1))
        {
            out = cl::NDRange(16, 16, 1);
        } else if(isAdmissibleRange(8, 8, 1))
        {
            out = cl::NDRange(8, 8, 1);
        } else if(isAdmissibleRange(4, 4, 1))
        {
            out = cl::NDRange(4, 4, 1);
        } else
        {
            out = cl::NullRange;
        }
    }
    return out;
}

cl::NDRange PBCTLocalNDRangeFactory::guessBackprojectorLocalNDRange() const
{
    cl::NDRange out;
    if(isAdmissibleRange(128, 2, 1))
    {
        out = cl::NDRange(128, 2, 1); // 34.97
    } else if(isAdmissibleRange(32, 4, 1))
    {
        out = cl::NDRange(32, 4, 1); // 35.16
    } else
    {
        out = cl::NullRange; // 35.22
    }
    return out;
}

} // namespace KCT
