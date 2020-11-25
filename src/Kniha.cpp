#include "Kniha.hpp"

using namespace CTL;
namespace CTL {

int Kniha::initializeOpenCL(uint32_t platformId,
                            uint32_t* deviceIds,
                            uint32_t deviceIdsLength,
                            std::string xpath,
                            bool debug)
{
    if(openCLInitialized)
    {
        std::string err = "Could not initialize OpenCL platform twice.";
        LOGE << err;
        throw std::runtime_error(err.c_str());
    }
    // Select the first available platform.
    platform = util::OpenCLManager::getPlatform(platformId, true);
    if(platform == nullptr)
    {
        return -1;
    }
    // Select the first available device for given platform
    std::shared_ptr<cl::Device> dev;
    if(deviceIdsLength == 0)
    {
        LOGD << io::xprintf("Adding deviceID %d on the platform %d.", 0, platformId);
        dev = util::OpenCLManager::getDevice(*platform, 0, true);
        if(dev == nullptr)
        {
            return -2;
        }
        devices.push_back(*dev);
    } else
    {
        for(uint32_t i = 0; i != deviceIdsLength; i++)
        {
            LOGD << io::xprintf("Adding deviceID %d on the platform %d.", deviceIds[i], platformId);
            dev = util::OpenCLManager::getDevice(*platform, deviceIds[i], true);
            if(dev == nullptr)
            {
                return -2;
            }
            devices.push_back(*dev);
        }
    }
    cl::Context tmp(devices);
    context = std::make_shared<cl::Context>(tmp);

    // Debug info
    // https://software.intel.com/en-us/openclsdk-devguide-enabling-debugging-in-opencl-runtime
    std::string clFile;
    std::string sourceText;
    clFile = io::xprintf("%s/opencl/allsources.cl", xpath.c_str());
    std::vector<std::string> clFilesXpath;
    for(std::string f : CLFiles)
    {
        clFilesXpath.push_back(io::xprintf("%s/%s", xpath.c_str(), f.c_str()));
    }
    io::concatenateTextFiles(clFile, true, clFilesXpath);
    std::string projectorSource = io::fileToString(clFile);
    cl::Program program(*context, projectorSource);
    std::string options = "";
    if(debug)
    {
        options = io::xprintf("-g -s \"%s\"", clFile.c_str());
    } else
    {
        options = "-Werror";
    }
    LOGI << io::xprintf("Building file %s with options : %s", clFile.c_str(), options.c_str());
    if(program.build(devices, options.c_str()) != CL_SUCCESS)
    {
        LOGE << " Error building: ";
        for(cl::Device dev : devices)
        {
            cl_build_status s = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);
            std::string status = "CL_BUILD_SUCCESS";
            if(s == CL_BUILD_NONE)
            {
                status = "CL_BUILD_NONE";
            } else if(s == CL_BUILD_ERROR)
            {
                status = "CL_BUILD_ERROR";
            } else if(s == CL_BUILD_IN_PROGRESS)
            {
                status = "CL_BUILD_IN_PROGRESS";
            }
            std::string name = dev.getInfo<CL_DEVICE_NAME>();
            std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
            LOGE << io::xprintf("Device %s, status %s LOG:", name.c_str(), status.c_str())
                 << std::endl
                 << buildlog << std::endl;
        }
        return -3;
    }
    LOGI << io::xprintf("Build succesfull", options.c_str());
    // Unloading compiler to free resources is causing Segmentation fault on Intel platform
    // Was reported on
    // https://github.com/beagle-dev/beagle-lib/blob/master/libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp
    // Nvidia platform seems to be unaffected contrary to the report
    if(util::OpenCLManager::getPlatformName(platformId) != "Intel(R) OpenCL")
    {
        if(platform->unloadCompiler() != CL_SUCCESS)
        {
            LOGE << "Error compiler unloading";
        }
    }
    // OpenCL 1.2 got rid of KernelFunctor
    // https://forums.khronos.org/showthread.php/8317-cl-hpp-KernelFunctor-gone-replaced-with-KernelFunctorGlobal
    // https://stackoverflow.com/questions/23992369/what-should-i-use-instead-of-clkernelfunctor/54344990#54344990
    for(std::function<void(cl::Program)> f : callbacks)
    {
        f(program);
    }
    for(uint32_t i = 0; i != devices.size(); i++)
    {
        Q.push_back(std::make_shared<cl::CommandQueue>(*context, devices[i]));
    }
    openCLInitialized = true;
    return 0;
}

void Kniha::CLINCLUDEbackprojector()
{

    insertCLFile("opencl/backprojector.cl");
    callbacks.emplace_back([this](cl::Program program) {
        auto& ptr = FLOATcutting_voxel_backproject;
        std::string str = "FLOATcutting_voxel_backproject";
        if(ptr == nullptr)
        {
            ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                cl::Kernel(program, str.c_str()));
        };
    });
}

void Kniha::CLINCLUDEbackprojector_minmax()
{
    insertCLFile("opencl/backprojector_minmax.cl");
    callbacks.emplace_back([this](cl::Program program) {
        auto& ptr = FLOATcutting_voxel_minmaxbackproject;
        std::string str = "FLOATcutting_voxel_minmaxbackproject";
        if(ptr == nullptr)
        {
            ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                cl::Kernel(program, str.c_str()));
        };
    });
}

void Kniha::CLINCLUDEbackprojector_sidon()
{
    insertCLFile("opencl/backprojector_sidon.cl");
    callbacks.emplace_back([this](cl::Program program) {
        auto& ptr = FLOATsidon_backproject;
        std::string str = "FLOATsidon_backproject";
        if(ptr == nullptr)
        {
            ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                cl::Kernel(program, str.c_str()));
        };
    });
}

void Kniha::CLINCLUDEbackprojector_tt()
{
    insertCLFile("opencl/backprojector_tt.cl");
    callbacks.emplace_back([this](cl::Program program) {
        auto& ptr = FLOATta3_backproject;
        std::string str = "FLOATta3_backproject";
        if(ptr == nullptr)
        {
            ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                cl::Kernel(program, str.c_str()));
        };
    });
}

void Kniha::CLINCLUDEcenterVoxelProjector()
{
    insertCLFile("opencl/centerVoxelProjector.cl");
    callbacks.emplace_back([this](cl::Program program) {
        auto& ptr = FLOATcenter_voxel_project;
        std::string str = "FLOATcenter_voxel_project";
        if(ptr == nullptr)
        {
            ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                cl::Kernel(program, str.c_str()));
        };
    });
}

void Kniha::CLINCLUDEinclude() { insertCLFile("opencl/include.cl"); }

void Kniha::CLINCLUDEjacobiPreconditionedBackprojector()
{
    insertCLFile("opencl/jacobiPreconditionedBackprojector.cl");
    callbacks.emplace_back([this](cl::Program program) {
        auto& ptr = FLOATjacobiPreconditionedCutting_voxel_backproject;
        std::string str = "FLOATjacobiPreconditionedCutting_voxel_backproject";
        if(ptr == nullptr)
        {
            ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                cl::Kernel(program, str.c_str()));
        };
    });
}

void Kniha::CLINCLUDEjacobiPreconditionedProjector()
{
    insertCLFile("opencl/jacobiPreconditionedProjector.cl");
    callbacks.emplace_back([this](cl::Program program) {
        auto& ptr = FLOATjacobiPreconditionedCutting_voxel_project;
        std::string str = "FLOATjacobiPreconditionedCutting_voxel_project";
        if(ptr == nullptr)
        {
            ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                cl::Kernel(program, str.c_str()));
        };
    });
}

void Kniha::CLINCLUDEprecomputeJacobiPreconditioner()
{
    insertCLFile("opencl/precomputeJacobiPreconditioner.cl");
    callbacks.emplace_back([this](cl::Program program) {
        auto& ptr = FLOATcutting_voxel_jacobiPreconditionerVector;
        std::string str = "FLOATcutting_voxel_jacobiPreconditionerVector";
        if(ptr == nullptr)
        {
            ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                cl::Kernel(program, str.c_str()));
        };
    });
}

void Kniha::CLINCLUDEprojector()
{
    insertCLFile("opencl/projector.cl");
    callbacks.emplace_back([this](cl::Program program) {
        auto& ptr = FLOATcutting_voxel_project;
        std::string str = "FLOATcutting_voxel_project";
        if(ptr == nullptr)
        {
            ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                cl::Kernel(program, str.c_str()));
        };
    });
}

void Kniha::CLINCLUDEprojector_old()
{
    insertCLFile("opencl/projector_old.cl");
    callbacks.emplace_back([this](cl::Program program) {
        auto& ptr = OLD_FLOATcutting_voxel_project;
        std::string str = "OLD_FLOATcutting_voxel_project";
        if(ptr == nullptr)
        {
            ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                cl::Kernel(program, str.c_str()));
        };
    });
}

void Kniha::CLINCLUDEprojector_sidon()
{
    insertCLFile("opencl/projector_sidon.cl");
    callbacks.emplace_back([this](cl::Program program) {
        auto& ptr = FLOATsidon_project;
        std::string str = "FLOATsidon_project";
        if(ptr == nullptr)
        {
            ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                cl::Kernel(program, str.c_str()));
        };
    });
}

void Kniha::CLINCLUDEprojector_tt()
{
    insertCLFile("opencl/projector_tt.cl");
    callbacks.emplace_back([this](cl::Program program) {
        auto& ptr = FLOATta3_project;
        std::string str = "FLOATta3_project";
        if(ptr == nullptr)
        {
            ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                cl::Kernel(program, str.c_str()));
        };
    });
}

void Kniha::CLINCLUDErescaleProjections()
{
    insertCLFile("opencl/rescaleProjections.cl");
    callbacks.emplace_back([this](cl::Program program) {
        auto& ptr = FLOATrescale_projections_cos;
        std::string str = "FLOATrescale_projections_cos";
        if(ptr == nullptr)
        {
            ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                cl::Kernel(program, str.c_str()));
        };
        auto& ptr01 = FLOATrescale_projections_exact;
        str = "FLOATrescale_projections_exact";
        if(ptr01 == nullptr)
        {
            ptr01 = std::make_shared<std::remove_reference<decltype(*ptr01)>::type>(
                cl::Kernel(program, str.c_str()));
        };
    });
}

void Kniha::CLINCLUDEutils()
{
    insertCLFile("opencl/utils.cl");
    callbacks.emplace_back([this](cl::Program program) {
        auto& ptr = FLOATvector_NormSquarePartial;
        std::string str = "FLOATvector_NormSquarePartial";
        if(ptr == nullptr)
        {
            ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                cl::Kernel(program, str.c_str()));
        };
        auto& ptr01 = FLOATvector_SumPartial;
        str = "FLOATvector_SumPartial";
        if(ptr01 == nullptr)
        {
            ptr01 = std::make_shared<std::remove_reference<decltype(*ptr01)>::type>(
                cl::Kernel(program, str.c_str()));
        };
        auto& ptr02 = FLOATvector_NormSquarePartial_barier;
        str = "FLOATvector_NormSquarePartial_barier";
        if(ptr02 == nullptr)
        {
            ptr02 = std::make_shared<std::remove_reference<decltype(*ptr02)>::type>(
                cl::Kernel(program, str.c_str()));
        };
        auto& ptr03 = FLOATvector_SumPartial_barier;
        str = "FLOATvector_SumPartial_barier";
        if(ptr03 == nullptr)
        {
            ptr03 = std::make_shared<std::remove_reference<decltype(*ptr03)>::type>(
                cl::Kernel(program, str.c_str()));
        };
        auto& ptr04 = vector_NormSquarePartial;
        str = "vector_NormSquarePartial";
        if(ptr04 == nullptr)
        {
            ptr04 = std::make_shared<std::remove_reference<decltype(*ptr04)>::type>(
                cl::Kernel(program, str.c_str()));
        };
        auto& ptr05 = vector_SumPartial;
        str = "vector_SumPartial";
        if(ptr05 == nullptr)
        {
            ptr05 = std::make_shared<std::remove_reference<decltype(*ptr05)>::type>(
                cl::Kernel(program, str.c_str()));
        };
        auto& ptr06 = vector_NormSquarePartial_barier;
        str = "vector_NormSquarePartial_barier";
        if(ptr06 == nullptr)
        {
            ptr06 = std::make_shared<std::remove_reference<decltype(*ptr06)>::type>(
                cl::Kernel(program, str.c_str()));
        };
        auto& ptr07 = vector_SumPartial_barier;
        str = "vector_SumPartial_barier";
        if(ptr07 == nullptr)
        {
            ptr07 = std::make_shared<std::remove_reference<decltype(*ptr07)>::type>(
                cl::Kernel(program, str.c_str()));
        };
        auto& ptr08 = vector_ScalarProductPartial_barier;
        str = "vector_ScalarProductPartial_barier";
        if(ptr08 == nullptr)
        {
            ptr08 = std::make_shared<std::remove_reference<decltype(*ptr08)>::type>(
                cl::Kernel(program, str.c_str()));
        };
        auto& ptr09 = FLOATvector_zero;
        str = "FLOATvector_zero";
        if(ptr09 == nullptr)
        {
            ptr09 = std::make_shared<std::remove_reference<decltype(*ptr09)>::type>(
                cl::Kernel(program, str.c_str()));
        };
        auto& ptr10 = FLOATvector_zero_infinite_values;
        str = "FLOATvector_zero_infinite_values";
        if(ptr10 == nullptr)
        {
            ptr10 = std::make_shared<std::remove_reference<decltype(*ptr10)>::type>(
                cl::Kernel(program, str.c_str()));
        };
        auto& ptr11 = FLOATvector_scale;
        str = "FLOATvector_scale";
        if(ptr11 == nullptr)
        {
            ptr11 = std::make_shared<std::remove_reference<decltype(*ptr11)>::type>(
                cl::Kernel(program, str.c_str()));
        };
        auto& ptr12 = FLOATvector_sqrt;
        str = "FLOATvector_sqrt";
        if(ptr12 == nullptr)
        {
            ptr12 = std::make_shared<std::remove_reference<decltype(*ptr12)>::type>(
                cl::Kernel(program, str.c_str()));
        };
        auto& ptr13 = FLOATvector_invert;
        str = "FLOATvector_invert";
        if(ptr13 == nullptr)
        {
            ptr13 = std::make_shared<std::remove_reference<decltype(*ptr13)>::type>(
                cl::Kernel(program, str.c_str()));
        };
        auto& ptr14 = FLOATvector_invert_except_zero;
        str = "FLOATvector_invert_except_zero";
        if(ptr14 == nullptr)
        {
            ptr14 = std::make_shared<std::remove_reference<decltype(*ptr14)>::type>(
                cl::Kernel(program, str.c_str()));
        };
        auto& ptr15 = vector_SumPartial;
        str = "FLOATvector_substitute_greater_than";
        if(ptr15 == nullptr)
        {
            ptr15 = std::make_shared<std::remove_reference<decltype(*ptr15)>::type>(
                cl::Kernel(program, str.c_str()));
        };
        auto& ptr16 = vector_SumPartial;
        str = "FLOATvector_substitute_lower_than";
        if(ptr16 == nullptr)
        {
            ptr16 = std::make_shared<std::remove_reference<decltype(*ptr16)>::type>(
                cl::Kernel(program, str.c_str()));
        };
        {
            auto& ptr17 = FLOATvector_copy;
            str = "FLOATvector_copy";
            if(ptr17 == nullptr)
            {
                ptr17 = std::make_shared<std::remove_reference<decltype(*ptr17)>::type>(
                    cl::Kernel(program, str.c_str()));
            }
        }
        {
            auto& ptr18 = FLOATvector_copy_offset;
            str = "FLOATvector_copy_offset";
            if(ptr18 == nullptr)
            {
                ptr18 = std::make_shared<std::remove_reference<decltype(*ptr18)>::type>(
                    cl::Kernel(program, str.c_str()));
            }
        }
        {
            auto& ptr19 = FLOATvector_copy_offsets;
            str = "FLOATvector_copy_offsets";
            if(ptr19 == nullptr)
            {
                ptr19 = std::make_shared<std::remove_reference<decltype(*ptr19)>::type>(
                    cl::Kernel(program, str.c_str()));
            }
        }
        {
            auto& ptr = FLOATvector_A_equals_cB;
            str = "FLOATvector_A_equals_cB";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            }
        }
        {
            auto& ptr20 = FLOATvector_A_equals_A_plus_cB;
            str = "FLOATvector_A_equals_A_plus_cB";
            if(ptr20 == nullptr)
            {
                ptr20 = std::make_shared<std::remove_reference<decltype(*ptr20)>::type>(
                    cl::Kernel(program, str.c_str()));
            }
        }
        auto& ptr21 = FLOATvector_A_equals_Ac_plus_B;
        str = "FLOATvector_A_equals_Ac_plus_B";
        if(ptr21 == nullptr)
        {
            ptr21 = std::make_shared<std::remove_reference<decltype(*ptr21)>::type>(
                cl::Kernel(program, str.c_str()));
        };
        auto& ptr22 = FLOATvector_A_equals_A_times_B;
        str = "FLOATvector_A_equals_A_times_B";
        if(ptr22 == nullptr)
        {
            ptr22 = std::make_shared<std::remove_reference<decltype(*ptr22)>::type>(
                cl::Kernel(program, str.c_str()));
        };
        auto& ptr23 = FLOATvector_C_equals_A_times_B;
        str = "FLOATvector_C_equals_A_times_B";
        if(ptr23 == nullptr)
        {
            ptr23 = std::make_shared<std::remove_reference<decltype(*ptr23)>::type>(
                cl::Kernel(program, str.c_str()));
        };
        auto& ptr24 = FLOATvector_A_equals_A_plus_cB_offset;
        str = "FLOATvector_A_equals_A_plus_cB_offset";
        if(ptr24 == nullptr)
        {
            ptr24 = std::make_shared<std::remove_reference<decltype(*ptr24)>::type>(
                cl::Kernel(program, str.c_str()));
        };
        auto& ptr25 = FLOATvector_B_equals_A_plus_B_offsets;
        str = "FLOATvector_B_equals_A_plus_B_offsets";
        if(ptr25 == nullptr)
        {
            ptr25 = std::make_shared<std::remove_reference<decltype(*ptr25)>::type>(
                cl::Kernel(program, str.c_str()));
        };
        auto& ptr26 = FLOATvector_A_equals_A_plus_cB_offsets;
        str = "FLOATvector_A_equals_A_plus_cB_offsets";
        if(ptr26 == nullptr)
        {
            ptr26 = std::make_shared<std::remove_reference<decltype(*ptr26)>::type>(
                cl::Kernel(program, str.c_str()));
        }
    });
}

void Kniha::insertCLFile(std::string f)
{
    if(std::find(CLFiles.begin(), CLFiles.end(), f) == CLFiles.end())
    // Vector does not contain f yet
    {
        CLFiles.emplace_back(f);
    }
}

int Kniha::algFLOATvector_copy(cl::Buffer& A, cl::Buffer& B, uint64_t size, bool blocking)
{
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(size));
    auto exe = (*FLOATvector_copy)(eargs, A, B);
    if(blocking)
    {
        exe.wait();
    }
    return 0;
}

int Kniha::algFLOATvector_copy_offset(
    cl::Buffer& A, cl::Buffer& B, unsigned int offset, uint64_t size, bool blocking)
{
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(size));
    auto exe = (*FLOATvector_copy_offset)(eargs, A, B, offset);
    if(blocking)
    {
        exe.wait();
    }
    return 0;
}

int Kniha::algFLOATvector_copy_offsets(
    cl::Buffer& A, unsigned int oA, cl::Buffer& B, unsigned int oB, uint64_t size, bool blocking)
{
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(size));
    auto exe = (*FLOATvector_copy_offsets)(eargs, A, oA, B, oB);
    if(blocking)
    {
        exe.wait();
    }
    return 0;
}
int Kniha::algFLOATvector_A_equals_cB(
    cl::Buffer& A, cl::Buffer& B, float c, uint64_t size, bool blocking)
{
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(size));
    auto exe = (*FLOATvector_A_equals_cB)(eargs, A, B, c);
    if(blocking)
    {
        exe.wait();
    }
    return 0;
}
int Kniha::algFLOATvector_A_equals_A_plus_cB(
    cl::Buffer& A, cl::Buffer& B, float c, uint64_t size, bool blocking)
{
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(size));
    auto exe = (*FLOATvector_A_equals_A_plus_cB)(eargs, A, B, c);
    if(blocking)
    {
        exe.wait();
    }
    return 0;
}
int Kniha::algFLOATvector_A_equals_Ac_plus_B(
    cl::Buffer& A, cl::Buffer& B, float c, uint64_t size, bool blocking)
{
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(size));
    auto exe = (*FLOATvector_A_equals_Ac_plus_B)(eargs, A, B, c);
    if(blocking)
    {
        exe.wait();
    }
    return 0;
}

int Kniha::algFLOATvector_A_equals_A_plus_cB_offset(
    cl::Buffer& A, cl::Buffer& B, float c, unsigned int offset, uint64_t size, bool blocking)
{
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(size));
    auto exe = (*FLOATvector_A_equals_A_plus_cB_offset)(eargs, A, B, c, offset);
    if(blocking)
    {
        exe.wait();
    }
    return 0;
}
int Kniha::algFLOATvector_zero(cl::Buffer& A, uint64_t size, bool blocking)
{

    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(size));
    auto exe = (*FLOATvector_zero)(eargs, A);
    if(blocking)
    {
        exe.wait();
    }
    return 0;
}
int Kniha::algFLOATvector_zero_infinite_values(cl::Buffer& A,
                                               uint64_t size,
                                               bool blocking)
{
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(size));
    auto exe = (*FLOATvector_zero_infinite_values)(eargs, A);
    if(blocking)
    {
        exe.wait();
    }
    return 0;
}
int Kniha::algFLOATvector_scale(cl::Buffer& A, float c, uint64_t size, bool blocking)
{
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(size));
    auto exe = (*FLOATvector_scale)(eargs, A, c);
    if(blocking)
    {
        exe.wait();
    }
    return 0;
}
int Kniha::algFLOATvector_A_equals_A_plus_cB_offsets(cl::Buffer& A,
                                              unsigned int oA,
                                              cl::Buffer& B,
                                              unsigned int oB,
                                              float c,
                                              uint64_t size,
                                              bool blocking)
{
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(size));
    auto exe = (*FLOATvector_A_equals_A_plus_cB_offsets)(eargs, A, oA, B, oB, c);
    if(blocking)
    {
        exe.wait();
    }
    return 0;
}
int Kniha::algFLOATvector_B_equals_A_plus_B_offsets(cl::Buffer& A,
                                                                      unsigned int oA,
                                                                      cl::Buffer& B,
                                                                      unsigned int oB,
                                                                      uint64_t size,
                                                                      bool blocking)
{
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(size));
    auto exe = (*FLOATvector_B_equals_A_plus_B_offsets)(eargs, A, oA, B, oB);
    if(blocking)
    {
        exe.wait();
    }
    return 0;
}
int Kniha::algFLOATvector_invert(cl::Buffer& A, uint64_t size, bool blocking)
{
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(size));
    auto exe = (*FLOATvector_invert)(eargs, A);
    if(blocking)
    {
        exe.wait();
    }
    return 0;
}
int Kniha::algFLOATvector_invert_except_zero(cl::Buffer& A,
                                                       uint64_t size,
                                                       bool blocking)
{
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(size));
    auto exe = (*FLOATvector_invert_except_zero)(eargs, A);
    if(blocking)
    {
        exe.wait();
    }
    return 0;
}
int Kniha::algFLOATvector_A_equals_A_times_B(cl::Buffer& A,
                                                               cl::Buffer& B,
                                                               uint64_t size,
                                                               bool blocking)
{
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(size));
    auto exe = (*FLOATvector_A_equals_A_times_B)(eargs, A, B);
    if(blocking)
    {
        exe.wait();
    }
    return 0;
}
int Kniha::algFLOATvector_C_equals_A_times_B(
    cl::Buffer& A, cl::Buffer& B, cl::Buffer& C, uint64_t size, bool blocking)
{
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(size));
    auto exe = (*FLOATvector_C_equals_A_times_B)(eargs, A, B, C);
    if(blocking)
    {
        exe.wait();
    }
    return 0;
}
} // namespace CTL