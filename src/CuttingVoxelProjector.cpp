#include "CuttingVoxelProjector.hpp"

namespace CTL {

int CuttingVoxelProjector::initializeOpenCL()
{
    // List platforms and select the first one.
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if(all_platforms.size() == 0)
    {
        LOGE << "No platforms found. Check OpenCL installation!";
        return -1;
    }
    LOGD << io::xprintf("There exists %d OpenCL platforms on the PC.", all_platforms.size());
    cl::Platform default_platform = all_platforms[0];
    LOGI << "Using OpenCL platform: " << default_platform.getInfo<CL_PLATFORM_NAME>();
    // List devices and select the first one
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size() == 0)
    {
        LOGE << "No devices found on the platform. Check OpenCL installation!";
        return -2;
    }
    LOGD << io::xprintf("There exists %d OpenCL devices on the platform.", all_devices.size());
    cl::Device default_device = all_devices[0];
    LOGI << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>();
        return 0;
}

} // namespace CTL
