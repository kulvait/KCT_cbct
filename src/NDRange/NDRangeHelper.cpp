#include "NDRange/NDRangeHelper.hpp"

namespace KCT {
cl::NDRange NDRangeHelper::flipNDRange(cl::NDRange& x)
{
    int dim = x.dimensions();
    if(dim == 2)
    {
        return cl::NDRange(x[1], x[0]);
    } else if(dim == 3)
    {
        return cl::NDRange(x[2], x[1], x[0]);
    } else
    { // Nothing to flip
        return x;
    }
}

std::string NDRangeHelper::NDRangeToString(cl::NDRange& x, std::string name)
{
    std::string str;
    int dim = x.dimensions();
    if(dim == 0)
    {
        str = "cl::NullRange";
    } else if(dim == 1)
    {
        str = io::xprintf("cl::NDRange(%d)", x[0]);
    } else if(dim == 2)
    {
        str = io::xprintf("cl::NDRange(%d, %d)", x[0], x[1]);
    } else
    {
        str = io::xprintf("cl::NDRange(%d, %d, %d)", x[0], x[1], x[2]);
    }
    if(name != "")
    {
        return io::xprintf("%s=%s", name.c_str(), str.c_str());
    } else
    {
        return str;
    }
}
} // namespace KCT
