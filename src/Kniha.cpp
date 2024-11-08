#include "Kniha.hpp"

using namespace KCT;
namespace KCT {

void Kniha::addOptString(std::string option)
{
    if(!openCLInitialized)
    {
        optstrings.emplace_back(option);
    }
}

int Kniha::initializeOpenCL(uint32_t platformID,
                            uint32_t* deviceIds,
                            uint32_t deviceIdsLength,
                            std::string xpath,
                            bool debug,
                            bool relaxed)
{
    if(openCLInitialized)
    {
        err = "Could not initialize OpenCL platform twice.";
        KCTERR(err);
    }
    // Select the first available platform.
    platform = util::OpenCLManager::getPlatform(platformID, true);
    if(platform == nullptr)
    {
        return -1;
    }
    // Select the first available device for given platform
    std::shared_ptr<cl::Device> dev;
    if(deviceIdsLength == 0)
    {
        LOGD << io::xprintf("Adding deviceID %d on the platform %d.", 0, platformID);
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
            LOGD << io::xprintf("Adding deviceID %d on the platform %d.", deviceIds[i], platformID);
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
    // clFile = io::xprintf("%s/opencl/allsources.cl", xpath.c_str());
    std::string tmpDir = std::experimental::filesystem::temp_directory_path().string();
    std::srand(std::time(nullptr));
    if(!debug)
    {
        unsigned int randomNumber = std::rand();
        clFile = io::xprintf("%s/allsources_%d.cl", tmpDir.c_str(), randomNumber);
    } else
    {
        clFile = io::xprintf("%s/allsources.cl", tmpDir.c_str());
    }
    std::vector<std::string> clFilesXpath;
    for(std::string f : CLFiles)
    {
        LOGD << io::xprintf("Including file %s", f.c_str());
        clFilesXpath.push_back(io::xprintf("%s/%s", xpath.c_str(), f.c_str()));
    }
    io::concatenateTextFiles(clFile, true, clFilesXpath);
    std::string allSources = io::fileToString(clFile);
    program = std::make_shared<cl::Program>(*context, allSources);
    if(relaxed)
    {
        optstrings.emplace_back("-DRELAXED");
        optstrings.emplace_back("-cl-fast-relaxed-math");
    }
    if(debug)
    {
        optstrings.emplace_back(io::xprintf("-g -s \"%s\"", clFile.c_str()));
    } else
    {
        optstrings.emplace_back("-Werror");
    }
    // Join operation
    std::string options
        = std::accumulate(std::begin(optstrings), std::end(optstrings), std::string(""),
                          [](const std::string& A, const std::string& B) {
                              return B.empty() ? A : (A.empty() ? B : A + " " + B);
                          });
    LOGI << io::xprintf("Building file %s with options : %s", clFile.c_str(), options.c_str());
    cl_int inf = program->build(devices, options.c_str());
    if(inf != CL_SUCCESS)
    {
        // Error codes can be found here
        // https://github.com/opencv/opencv/blob/master/3rdparty/include/opencl/1.2/CL/cl.h
        if(inf == CL_INVALID_COMPILER_OPTIONS)
        {
            if(debug)
            {
                LOGE << io::xprintf("Error CL_INVALID_COMPILER_OPTIONS when building. Not all "
                                    "platforms might support debugging!");
            } else
            {
                LOGE << io::xprintf("Error CL_INVALID_COMPILER_OPTIONS when building.");
            }
            return CL_INVALID_COMPILER_OPTIONS; //-66
        }
        LOGE << io::xprintf("Error building, program.build returned %d, see codes at "
                            "https://github.com/opencv/opencv/blob/master/3rdparty/include/opencl/"
                            "1.2/CL/cl.h.",
                            inf);
        for(cl::Device dev : devices)
        {
            cl_build_status s = program->getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);
            std::string status = "";
            if(s == CL_BUILD_NONE)
            {
                status = "CL_BUILD_NONE";
            } else if(s == CL_BUILD_ERROR)
            {
                status = "CL_BUILD_ERROR";
            } else if(s == CL_BUILD_IN_PROGRESS)
            {
                status = "CL_BUILD_IN_PROGRESS";
            } else if(s == CL_BUILD_SUCCESS)
            {
                status = "CL_BUILD_SUCCESS";
            } else
            {
                status = io::xprintf("Status code %d", s);
            }
            std::string name = dev.getInfo<CL_DEVICE_NAME>();
            std::string buildlog = program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
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
    if(util::OpenCLManager::getPlatformName(platformID) != "Intel(R) OpenCL")
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
        f(*program);
    }
    std::vector<uint64_t> devices_maxWorkGroupSize;
    std::vector<uint64_t> devices_localMemBytesize;
    for(uint32_t i = 0; i != devices.size(); i++)
    {
        Q.push_back(std::make_shared<cl::CommandQueue>(*context, devices[i]));
        devices_maxWorkGroupSize.push_back(util::OpenCLManager::maxWGS(devices[i]));
        devices_localMemBytesize.push_back(util::OpenCLManager::localMemSize(devices[i]));
    }
    maxWorkGroupSize
        = *std::min_element(devices_maxWorkGroupSize.begin(), devices_maxWorkGroupSize.end());
    localMemBytesize
        = *std::min_element(devices_localMemBytesize.begin(), devices_localMemBytesize.end());
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

void Kniha::CLINCLUDEprojector_cvp_barrier()
{
    insertCLFile("opencl/projector_cvp_barrier.cl");
    callbacks.emplace_back([this](cl::Program program) {
        auto& ptr = FLOATcutting_voxel_project_barrier;
        std::string str = "FLOATcutting_voxel_project_barrier";
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
        {
            auto& ptr = FLOATvector_MaxPartial;
            str = "FLOATvector_MaxPartial";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            }
        }
        auto& ptr02 = FLOATvector_NormSquarePartial_barrier;
        str = "FLOATvector_NormSquarePartial_barrier";
        if(ptr02 == nullptr)
        {
            cl_int err;
            ptr02 = std::make_shared<std::remove_reference<decltype(*ptr02)>::type>(
                cl::Kernel(program, str.c_str(), &err));
            if(err != CL_SUCCESS)
            {
                LOGE << io::xprintf("Error %d kernel %s", err, str.c_str());
            }
        };
        auto& ptr03 = FLOATvector_SumPartial_barrier;
        str = "FLOATvector_SumPartial_barrier";
        if(ptr03 == nullptr)
        {
            ptr03 = std::make_shared<std::remove_reference<decltype(*ptr03)>::type>(
                cl::Kernel(program, str.c_str()));
        };
        {
            auto& ptr = FLOATvector_MaxPartial_barrier;
            str = "FLOATvector_MaxPartial_barrier";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            }
        }
        {
            // std::shared_ptr<cl::make_kernel<cl::Buffer&, cl:Buffer&, cl::Buffer&,
            // cl::LocalSpaceArg&, unsigned long&>> FLOATvector_L1L2norm_barrier;
            auto& ptr = FLOATvector_L1L2norm_barrier;
            str = "FLOATvector_L1L2norm_barrier";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            }
        }
        {
            // std::shared_ptr<cl::make_kernel<cl::Buffer&, cl:Buffer&, cl::Buffer&,
            // cl::LocalSpaceArg&, unsigned long&>> vector_L1L2norm_barrier;
            auto& ptr = vector_L1L2norm_barrier;
            str = "vector_L1L2norm_barrier";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            }
        }
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
        auto& ptr06 = vector_NormSquarePartial_barrier;
        str = "vector_NormSquarePartial_barrier";
        if(ptr06 == nullptr)
        {
            ptr06 = std::make_shared<std::remove_reference<decltype(*ptr06)>::type>(
                cl::Kernel(program, str.c_str()));
        };
        auto& ptr07 = vector_SumPartial_barrier;
        str = "vector_SumPartial_barrier";
        if(ptr07 == nullptr)
        {
            ptr07 = std::make_shared<std::remove_reference<decltype(*ptr07)>::type>(
                cl::Kernel(program, str.c_str()));
        };
        auto& ptr08 = vector_ScalarProductPartial_barrier;
        str = "vector_ScalarProductPartial_barrier";
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
        {
            auto& ptr = FLOATvector_sqrt;
            str = "FLOATvector_sqrt";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            }
        }
        {
            auto& ptr = FLOATvector_square;
            str = "FLOATvector_square";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            }
        }
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
        auto& ptr15 = FLOATvector_substitute_greater_than;
        str = "FLOATvector_substitute_greater_than";
        if(ptr15 == nullptr)
        {
            ptr15 = std::make_shared<std::remove_reference<decltype(*ptr15)>::type>(
                cl::Kernel(program, str.c_str()));
        };
        auto& ptr16 = FLOATvector_substitute_lower_than;
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
        {
            auto& ptr = FLOATvector_C_equals_Ad_plus_Be;
            str = "FLOATvector_C_equals_Ad_plus_Be";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
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
        };
        auto& ptr27 = FLOATvector_A_equals_Ac_plus_Bd;
        str = "FLOATvector_A_equals_Ac_plus_Bd";
        if(ptr27 == nullptr)
        {
            ptr27 = std::make_shared<std::remove_reference<decltype(*ptr27)>::type>(
                cl::Kernel(program, str.c_str()));
        };
    });
}

void Kniha::CLINCLUDEconvolution()
{
    insertCLFile("opencl/convolution.cl");
    callbacks.emplace_back([this](cl::Program program) {
        {
            auto& ptr = FLOATvector_2Dconvolution3x3ZeroBoundary;
            std::string str = "FLOATvector_2Dconvolution3x3ZeroBoundary";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
        {
            auto& ptr = FLOATvector_2Dconvolution3x3ReflectionBoundary;
            std::string str = "FLOATvector_2Dconvolution3x3ReflectionBoundary";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
        {
            auto& ptr = FLOATvector_3DconvolutionGradientSobelFeldmanZeroBoundary;
            std::string str = "FLOATvector_3DconvolutionGradientSobelFeldmanZeroBoundary";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
        {
            auto& ptr = FLOATvector_3DconvolutionGradientSobelFeldmanReflectionBoundary;
            std::string str = "FLOATvector_3DconvolutionGradientSobelFeldmanReflectionBoundary";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
        {
            auto& ptr = FLOATvector_3DconvolutionLaplaceZeroBoundary;
            std::string str = "FLOATvector_3DconvolutionLaplaceZeroBoundary";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
        {
            auto& ptr = FLOATvector_3DconvolutionGradientFarid5x5x5;
            std::string str = "FLOATvector_3DconvolutionGradientFarid5x5x5";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
        {
            auto& ptr = FLOATvector_2DconvolutionGradientFarid5x5;
            std::string str = "FLOATvector_3DconvolutionGradientFarid5x5";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
        {
            auto& ptr = FLOATvector_3DisotropicGradient;
            std::string str = "FLOATvector_3DisotropicGradient";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
        {
            auto& ptr = FLOATvector_2DisotropicGradient;
            std::string str = "FLOATvector_2DisotropicGradient";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
        {
            auto& ptr = FLOATvector_isotropicBackDx;
            std::string str = "FLOATvector_isotropicBackDx";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
        {
            auto& ptr = FLOATvector_isotropicBackDy;
            std::string str = "FLOATvector_isotropicBackDy";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
        {
            auto& ptr = FLOATvector_isotropicBackDz;
            std::string str = "FLOATvector_isotropicBackDz";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
        {
            auto& ptr = FLOATvector_isotropicBackDivergence2D;
            std::string str = "FLOATvector_isotropicBackDivergence2D";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
    });
}

void Kniha::CLINCLUDEgradient()
{
    insertCLFile("opencl/gradient.cl");

    // Add to callbacks for kernel compilation after OpenCL runtime is ready
    callbacks.emplace_back([this](cl::Program program) {
        {
            auto& ptr = FLOATvector_Gradient2D_forwardDifference_2point;
            std::string str = "FLOATvector_Gradient2D_forwardDifference_2point";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
        {
            auto& ptr = FLOATvector_Gradient2D_forwardDifference_2point_adjoint;
            std::string str = "FLOATvector_Gradient2D_forwardDifference_2point_adjoint";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
        {
            auto& ptr = FLOATvector_Gradient2D_forwardDifference_3point;
            std::string str = "FLOATvector_Gradient2D_forwardDifference_3point";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
        {
            auto& ptr = FLOATvector_Gradient2D_forwardDifference_3point_adjoint;
            std::string str = "FLOATvector_Gradient2D_forwardDifference_3point_adjoint";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
        {
            auto& ptr = FLOATvector_Gradient2D_forwardDifference_4point;
            std::string str = "FLOATvector_Gradient2D_forwardDifference_4point";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
        {
            auto& ptr = FLOATvector_Gradient2D_forwardDifference_4point_adjoint;
            std::string str = "FLOATvector_Gradient2D_forwardDifference_4point_adjoint";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
        {
            auto& ptr = FLOATvector_Gradient2D_forwardDifference_5point;
            std::string str = "FLOATvector_Gradient2D_forwardDifference_5point";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
        {
            auto& ptr = FLOATvector_Gradient2D_forwardDifference_5point_adjoint;
            std::string str = "FLOATvector_Gradient2D_forwardDifference_5point_adjoint";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
        {
            auto& ptr = FLOATvector_Gradient2D_forwardDifference_6point;
            std::string str = "FLOATvector_Gradient2D_forwardDifference_6point";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
        {
            auto& ptr = FLOATvector_Gradient2D_forwardDifference_6point_adjoint;
            std::string str = "FLOATvector_Gradient2D_forwardDifference_6point_adjoint";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
        {
            auto& ptr = FLOATvector_Gradient2D_forwardDifference_7point;
            std::string str = "FLOATvector_Gradient2D_forwardDifference_7point";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
        {
            auto& ptr = FLOATvector_Gradient2D_forwardDifference_7point_adjoint;
            std::string str = "FLOATvector_Gradient2D_forwardDifference_7point_adjoint";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
        {
            auto& ptr = FLOATvector_Gradient2D_centralDifference_3point;
            std::string str = "FLOATvector_Gradient2D_centralDifference_3point";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
        {
            auto& ptr = FLOATvector_Gradient2D_centralDifference_3point_adjoint;
            std::string str = "FLOATvector_Gradient2D_centralDifference_3point_adjoint";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
        {
            auto& ptr = FLOATvector_Gradient2D_centralDifference_5point;
            std::string str = "FLOATvector_Gradient2D_centralDifference_5point";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
        {
            auto& ptr = FLOATvector_Gradient2D_centralDifference_5point_adjoint;
            std::string str = "FLOATvector_Gradient2D_centralDifference_5point_adjoint";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
    });
}

void Kniha::CLINCLUDEproximal()
{
    insertCLFile("opencl/proximal.cl");
    callbacks.emplace_back([this](cl::Program program) {
        {
            auto& ptr = FLOATvector_infProjectionToLambda2DBall;
            std::string str = "FLOATvector_infProjectionToLambda2DBall";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
        {
            auto& ptr = FLOATvector_distL1ProxSoftThreasholding;
            std::string str = "FLOATvector_distL1ProxSoftThreasholding";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
    });
}

void Kniha::CLINCLUDEpbct_cvp()
{
    insertCLFile("opencl/pbct_cvp.cl");
    callbacks.emplace_back([this](cl::Program program) {
        {
            auto& ptr = FLOAT_pbct_cutting_voxel_project;
            std::string str = "FLOAT_pbct_cutting_voxel_project";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
        {
            auto& ptr = FLOAT_pbct_cutting_voxel_backproject;
            std::string str = "FLOAT_pbct_cutting_voxel_backproject";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
    });
}

void Kniha::CLINCLUDEpbct_cvp_barrier()
{
    insertCLFile("opencl/pbct_cvp_barrier.cl");
    callbacks.emplace_back([this](cl::Program program) {
        {
            auto& ptr = FLOAT_pbct_cutting_voxel_project_barrier;
            std::string str = "FLOAT_pbct_cutting_voxel_project_barrier";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
    });
}

void Kniha::CLINCLUDEpbct2d_cvp()
{
    insertCLFile("opencl/pbct2d_cvp.cl");
    callbacks.emplace_back([this](cl::Program program) {
        {
            auto& ptr = FLOAT_pbct2d_cutting_voxel_project;
            std::string str = "FLOAT_pbct2d_cutting_voxel_project";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
        {
            auto& ptr = FLOAT_pbct2d_cutting_voxel_kaczmarz_product;
            std::string str = "FLOAT_pbct2d_cutting_voxel_kaczmarz_product";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
        {
            auto& ptr = FLOAT_pbct2d_cutting_voxel_backproject;
            std::string str = "FLOAT_pbct2d_cutting_voxel_backproject";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
        {
            auto& ptr = FLOAT_pbct2d_cutting_voxel_backproject_kaczmarz;
            std::string str = "FLOAT_pbct2d_cutting_voxel_backproject_kaczmarz";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
        {
            auto& ptr = FLOAT_pbct2d_cutting_voxel_jacobi_vector;
            std::string str = "FLOAT_pbct2d_cutting_voxel_jacobi_vector";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
        }
    });
}

void Kniha::CLINCLUDEpbct2d_cvp_barrier()
{
    insertCLFile("opencl/pbct2d_cvp_barrier.cl");
    callbacks.emplace_back([this](cl::Program program) {
        {
            auto& ptr = FLOAT_pbct2d_cutting_voxel_project_barrier;
            std::string str = "FLOAT_pbct2d_cutting_voxel_project_barrier";
            if(ptr == nullptr)
            {
                ptr = std::make_shared<std::remove_reference<decltype(*ptr)>::type>(
                    cl::Kernel(program, str.c_str()));
            };
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

std::string Kniha::infoString(cl_int cl_info_id)
{
    // See https://github.com/KhronosGroup/OpenCL-Headers/blob/main/CL/cl.h
    if(cl_info_id == CL_COMPLETE)
    {
        return "CL_COMPLETE";
    } else if(cl_info_id == CL_RUNNING)
    {
        return "CL_RUNNING";
    } else if(cl_info_id == CL_SUBMITTED)
    {
        return "CL_SUBMITTED";
    } else if(cl_info_id == CL_QUEUED)
    {
        return "CL_QUEUED";
    } else if(cl_info_id == CL_INVALID_VALUE)
    {
        return "CL_INVALID_VALUE";
    } else if(cl_info_id == CL_INVALID_CONTEXT)
    {
        return "CL_INVALID_CONTEXT";
    } else if(cl_info_id == CL_INVALID_EVENT)
    {
        return "CL_INVALID_EVENT";
    } else if(cl_info_id == CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)
    {
        return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    } else if(cl_info_id == CL_OUT_OF_RESOURCES)
    {
        return "CL_OUT_OF_RESOURCES";
    } else if(cl_info_id == CL_OUT_OF_HOST_MEMORY)
    {
        return "CL_OUT_OF_HOST_MEMORY";
    } else if(cl_info_id == CL_COMMAND_NDRANGE_KERNEL)
    {
        return "CL_COMMAND_NDRANGE_KERNEL";
    } else if(cl_info_id == CL_COMMAND_TASK)
    {
        return "CL_COMMAND_TASK";
    } else if(cl_info_id == CL_COMMAND_NATIVE_KERNEL)
    {
        return "CL_COMMAND_NATIVE_KERNEL";
    } else if(cl_info_id == CL_COMMAND_READ_BUFFER)
    {
        return "CL_COMMAND_READ_BUFFER";
    } else if(cl_info_id == CL_COMMAND_WRITE_BUFFER)
    {
        return "CL_COMMAND_WRITE_BUFFER";
    } else if(cl_info_id == CL_COMMAND_COPY_BUFFER)
    {
        return "CL_COMMAND_COPY_BUFFER";
    } else if(cl_info_id == CL_COMMAND_READ_IMAGE)
    {
        return "CL_COMMAND_READ_IMAGE";
    } else if(cl_info_id == CL_COMMAND_WRITE_IMAGE)
    {
        return "CL_COMMAND_WRITE_IMAGE";
    } else if(cl_info_id == CL_COMMAND_COPY_IMAGE)
    {
        return "CL_COMMAND_COPY_IMAGE";
    } else if(cl_info_id == CL_COMMAND_COPY_IMAGE_TO_BUFFER)
    {
        return "CL_COMMAND_COPY_IMAGE_TO_BUFFER";
    } else if(cl_info_id == CL_COMMAND_COPY_BUFFER_TO_IMAGE)
    {
        return "CL_COMMAND_COPY_BUFFER_TO_IMAGE";
    } else if(cl_info_id == CL_COMMAND_MAP_BUFFER)
    {
        return "CL_COMMAND_MAP_BUFFER";
    } else if(cl_info_id == CL_COMMAND_MAP_IMAGE)
    {
        return "CL_COMMAND_MAP_IMAGE";
    } else if(cl_info_id == CL_COMMAND_UNMAP_MEM_OBJECT)
    {
        return "CL_COMMAND_UNMAP_MEM_OBJECT";
    } else if(cl_info_id == CL_COMMAND_MARKER)
    {
        return "CL_COMMAND_MARKER";
    } else if(cl_info_id == CL_COMMAND_ACQUIRE_GL_OBJECTS)
    {
        return "CL_COMMAND_ACQUIRE_GL_OBJECTS";
    } else if(cl_info_id == CL_COMMAND_RELEASE_GL_OBJECTS)
    {
        return "CL_COMMAND_RELEASE_GL_OBJECTS";
    } else if(cl_info_id == CL_COMMAND_READ_BUFFER_RECT)
    {
        return "CL_COMMAND_READ_BUFFER_RECT";
    } else if(cl_info_id == CL_COMMAND_WRITE_BUFFER_RECT)
    {
        return "CL_COMMAND_WRITE_BUFFER_RECT";
    } else if(cl_info_id == CL_COMMAND_COPY_BUFFER_RECT)
    {
        return "CL_COMMAND_COPY_BUFFER_RECT";
    } else if(cl_info_id == CL_COMMAND_USER)
    {
        return "CL_COMMAND_USER";
    } else if(cl_info_id == CL_COMMAND_BARRIER)
    {
        return "CL_COMMAND_BARRIER";
    } else if(cl_info_id == CL_COMMAND_MIGRATE_MEM_OBJECTS)
    {
        return "CL_COMMAND_MIGRATE_MEM_OBJECTS";
    } else if(cl_info_id == CL_COMMAND_FILL_BUFFER)
    {
        return "CL_COMMAND_FILL_BUFFER";
    } else if(cl_info_id == CL_COMMAND_FILL_IMAGE)
    {
        return "CL_COMMAND_FILL_IMAGE";
    } /*else if(cl_info_id == CL_COMMAND_SVM_FREE)
    {
        return "CL_COMMAND_SVM_FREE";
    } else if(cl_info_id == CL_COMMAND_SVM_MEMCPY)
    {
        return "CL_COMMAND_SVM_MEMCPY";
    } else if(cl_info_id == CL_COMMAND_SVM_MEMFILL)
    {
        return "CL_COMMAND_SVM_MEMFILL";
    } else if(cl_info_id == CL_COMMAND_SVM_MAP)
    {
        return "CL_COMMAND_SVM_MAP";
    } else if(cl_info_id == CL_COMMAND_SVM_UNMAP)
    {
        return "CL_COMMAND_SVM_UNMAP";
    } */
    else if(cl_info_id == CL_DEVICE_NOT_FOUND)
    {
        return "CL_DEVICE_NOT_FOUND";
    } else if(cl_info_id == CL_DEVICE_NOT_AVAILABLE)
    {
        return "CL_DEVICE_NOT_AVAILABLE";
    } else if(cl_info_id == CL_COMPILER_NOT_AVAILABLE)
    {
        return "CL_COMPILER_NOT_AVAILABLE";
    } else if(cl_info_id == CL_MEM_OBJECT_ALLOCATION_FAILURE)
    {
        return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    } else if(cl_info_id == CL_OUT_OF_RESOURCES)
    {
        return "CL_OUT_OF_RESOURCES";
    } else if(cl_info_id == CL_OUT_OF_HOST_MEMORY)
    {
        return "CL_OUT_OF_HOST_MEMORY";
    } else if(cl_info_id == CL_PROFILING_INFO_NOT_AVAILABLE)
    {
        return "CL_PROFILING_INFO_NOT_AVAILABLE";
    } else if(cl_info_id == CL_MEM_COPY_OVERLAP)
    {
        return "CL_MEM_COPY_OVERLAP";
    } else if(cl_info_id == CL_IMAGE_FORMAT_MISMATCH)
    {
        return "CL_IMAGE_FORMAT_MISMATCH";
    } else if(cl_info_id == CL_IMAGE_FORMAT_NOT_SUPPORTED)
    {
        return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    } else if(cl_info_id == CL_BUILD_PROGRAM_FAILURE)
    {
        return "CL_BUILD_PROGRAM_FAILURE";
    } else if(cl_info_id == CL_MAP_FAILURE)
    {
        return "CL_MAP_FAILURE";
    } else if(cl_info_id == CL_MISALIGNED_SUB_BUFFER_OFFSET)
    {
        return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    } else if(cl_info_id == CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)
    {
        return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    } else if(cl_info_id == CL_COMPILE_PROGRAM_FAILURE)
    {
        return "CL_COMPILE_PROGRAM_FAILURE";
    } else if(cl_info_id == CL_LINKER_NOT_AVAILABLE)
    {
        return "CL_LINKER_NOT_AVAILABLE";
    } else if(cl_info_id == CL_LINK_PROGRAM_FAILURE)
    {
        return "CL_LINK_PROGRAM_FAILURE";
    } else if(cl_info_id == CL_DEVICE_PARTITION_FAILED)
    {
        return "CL_DEVICE_PARTITION_FAILED";
    } else if(cl_info_id == CL_KERNEL_ARG_INFO_NOT_AVAILABLE)
    {
        return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
    } else if(cl_info_id == CL_INVALID_VALUE)
    {
        return "CL_INVALID_VALUE";
    } else if(cl_info_id == CL_INVALID_DEVICE_TYPE)
    {
        return "CL_INVALID_DEVICE_TYPE";
    } else if(cl_info_id == CL_INVALID_PLATFORM)
    {
        return "CL_INVALID_PLATFORM";
    } else if(cl_info_id == CL_INVALID_DEVICE)
    {
        return "CL_INVALID_DEVICE";
    } else if(cl_info_id == CL_INVALID_CONTEXT)
    {
        return "CL_INVALID_CONTEXT";
    } else if(cl_info_id == CL_INVALID_QUEUE_PROPERTIES)
    {
        return "CL_INVALID_QUEUE_PROPERTIES";
    } else if(cl_info_id == CL_INVALID_COMMAND_QUEUE)
    {
        return "CL_INVALID_COMMAND_QUEUE";
    } else if(cl_info_id == CL_INVALID_HOST_PTR)
    {
        return "CL_INVALID_HOST_PTR";
    } else if(cl_info_id == CL_INVALID_MEM_OBJECT)
    {
        return "CL_INVALID_MEM_OBJECT";
    } else if(cl_info_id == CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
    {
        return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    } else if(cl_info_id == CL_INVALID_IMAGE_SIZE)
    {
        return "CL_INVALID_IMAGE_SIZE";
    } else if(cl_info_id == CL_INVALID_SAMPLER)
    {
        return "CL_INVALID_SAMPLER";
    } else if(cl_info_id == CL_INVALID_BINARY)
    {
        return "CL_INVALID_BINARY";
    } else if(cl_info_id == CL_INVALID_BUILD_OPTIONS)
    {
        return "CL_INVALID_BUILD_OPTIONS";
    } else if(cl_info_id == CL_INVALID_PROGRAM)
    {
        return "CL_INVALID_PROGRAM";
    } else if(cl_info_id == CL_INVALID_PROGRAM_EXECUTABLE)
    {
        return "CL_INVALID_PROGRAM_EXECUTABLE";
    } else if(cl_info_id == CL_INVALID_KERNEL_NAME)
    {
        return "CL_INVALID_KERNEL_NAME";
    } else if(cl_info_id == CL_INVALID_KERNEL_DEFINITION)
    {
        return "CL_INVALID_KERNEL_DEFINITION";
    } else if(cl_info_id == CL_INVALID_KERNEL)
    {
        return "CL_INVALID_KERNEL";
    } else if(cl_info_id == CL_INVALID_ARG_INDEX)
    {
        return "CL_INVALID_ARG_INDEX";
    } else if(cl_info_id == CL_INVALID_ARG_VALUE)
    {
        return "CL_INVALID_ARG_VALUE";
    } else if(cl_info_id == CL_INVALID_ARG_SIZE)
    {
        return "CL_INVALID_ARG_SIZE";
    } else if(cl_info_id == CL_INVALID_KERNEL_ARGS)
    {
        return "CL_INVALID_KERNEL_ARGS";
    } else if(cl_info_id == CL_INVALID_WORK_DIMENSION)
    {
        return "CL_INVALID_WORK_DIMENSION";
    } else if(cl_info_id == CL_INVALID_WORK_GROUP_SIZE)
    {
        return "CL_INVALID_WORK_GROUP_SIZE";
    } else if(cl_info_id == CL_INVALID_WORK_ITEM_SIZE)
    {
        return "CL_INVALID_WORK_ITEM_SIZE";
    } else if(cl_info_id == CL_INVALID_GLOBAL_OFFSET)
    {
        return "CL_INVALID_GLOBAL_OFFSET";
    } else if(cl_info_id == CL_INVALID_EVENT_WAIT_LIST)
    {
        return "CL_INVALID_EVENT_WAIT_LIST";
    } else if(cl_info_id == CL_INVALID_EVENT)
    {
        return "CL_INVALID_EVENT";
    } else if(cl_info_id == CL_INVALID_OPERATION)
    {
        return "CL_INVALID_OPERATION";
    } else if(cl_info_id == CL_INVALID_GL_OBJECT)
    {
        return "CL_INVALID_GL_OBJECT";
    } else if(cl_info_id == CL_INVALID_BUFFER_SIZE)
    {
        return "CL_INVALID_BUFFER_SIZE";
    } else if(cl_info_id == CL_INVALID_MIP_LEVEL)
    {
        return "CL_INVALID_MIP_LEVEL";
    } else if(cl_info_id == CL_INVALID_GLOBAL_WORK_SIZE)
    {
        return "CL_INVALID_GLOBAL_WORK_SIZE";
    } else if(cl_info_id == CL_INVALID_PROPERTY)
    {
        return "CL_INVALID_PROPERTY";
    } else if(cl_info_id == CL_INVALID_IMAGE_DESCRIPTOR)
    {
        return "CL_INVALID_IMAGE_DESCRIPTOR";
    } else if(cl_info_id == CL_INVALID_COMPILER_OPTIONS)
    {
        return "CL_INVALID_COMPILER_OPTIONS";
    } else if(cl_info_id == CL_INVALID_LINKER_OPTIONS)
    {
        return "CL_INVALID_LINKER_OPTIONS";
    } else if(cl_info_id == CL_INVALID_DEVICE_PARTITION_COUNT)
    {
        return "CL_INVALID_DEVICE_PARTITION_COUNT";
    } else if(cl_info_id == -9999)
    {
        return "Illegal read or write to a buffer on clEnqueueNDRangeKernel, NVIDIA specific error";
    } else
    {
        return io::xprintf("Unknown ID %d=0x%x, see "
                           "https://github.com/KhronosGroup/OpenCL-Headers/blob/main/CL/cl.h",
                           cl_info_id, cl_info_id);
    }
}

int Kniha::handleKernelExecution(cl::Event exe, bool blocking, std::string& errout)
{
    cl_int inf, ing;
    if(blocking)
    {
        inf = exe.wait();
        if(inf != CL_COMPLETE)
        {
            if(inf == CL_INVALID_EVENT)
            {
                cl_int command_type_int;
                inf = exe.getInfo(CL_EVENT_COMMAND_TYPE, &command_type_int);
                if(inf == CL_INVALID_EVENT)
                {
                    errout = "Blocking event wait and info CL_EVENT_COMMAND_TYPE returned "
                             "CL_INVALID_EVENT, might be out of memory.";
                } else
                {
                    std::string command_type_string = infoString(command_type_int);
                    errout = io::xprintf("Blocking event wait returned CL_INVALID_EVENT, "
                                         "CL_EVENT_COMMAND_TYPE=%s, might be out of memory.",
                                         command_type_string.c_str());
                }
                return 1;
            }
            cl_int command_type_int;
            exe.getInfo(CL_EVENT_COMMAND_TYPE, &command_type_int);
            std::string command_type_string = infoString(command_type_int);
            std::string status = infoString(inf);
            errout = io::xprintf(
                "Blocking CL_EVENT_COMMAND_TYPE=%s, COMMAND_EXECUTION_STATUS=%s that is "
                "different from CL_COMPLETE!",
                command_type_string.c_str(), status.c_str());
            return 1;
        }
    } else
    {
        ing = exe.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS, &inf);
        if(ing != CL_COMPLETE)
        {
            errout = io::xprintf("Unblocking event request to CL_EVENT_COMMAND_EXECUTION_STATUS "
                                 "failed with error code %s, might be out of memory.",
                                 infoString(ing).c_str());
            return 1;
        }
        if(inf != CL_COMPLETE && inf != CL_QUEUED && inf != CL_SUBMITTED && inf != CL_RUNNING)
        {
            cl_int command_type_int;
            ing = exe.getInfo(CL_EVENT_COMMAND_TYPE, &command_type_int);
            std::string command_type_string = infoString(command_type_int);
            std::string status = infoString(inf);
            if(ing == CL_COMPLETE)
            {
                errout
                    = io::xprintf("Unblocking CL_EVENT_COMMAND_TYPE=%s, "
                                  "COMMAND_EXECUTION_STATUS=%s that is different from CL_COMPLETE!",
                                  command_type_string.c_str(), status.c_str());
            } else
            {

                errout = io::xprintf(
                    "Unblocking COMMAND_EXECUTION_STATUS=%s that is different from CL_COMPLETE. "
                    "Request to CL_EVENT_COMMAND_TYPE failed with code %s!",
                    status.c_str(), infoString(ing).c_str());
            }
            return 2;
        }
    }
    return 0;
}

cl::NDRange Kniha::assignLocalRange(cl::NDRange localRange, cl::NDRange globalRange)
{
    size_t dim = globalRange.dimensions();
    if(localRange.dimensions() != 0)
    {
        if(dim != localRange.dimensions())
        {
            err = io::xprintf("Dimension mismatch between globalRange=%d and localRange=%d", dim,
                              localRange.dimensions());
            KCTERR(err);
        } else
        {
            for(unsigned int i = 0; i < dim; i++)
            {
                if(globalRange[i] < localRange[i])
                {
                    err = io::xprintf("globalRange[%d] < *localRange[i]  %d<%d.", i, globalRange[i],
                                      localRange[i]);
                    KCTERR(err);
                }
                if(globalRange[i] % localRange[i] != 0)
                {
                    err = io::xprintf(
                        "Global work size need to be multiple of work group size, see "
                        "https://stackoverflow.com/questions/3147940/"
                        "does-global-work-size-need-to-be-multiple-of-work-group-size-in-"
                        "opencl, but globalRange[%d]=%d, localRange[i]=%d.",
                        i, globalRange[i], localRange[i]);
                    KCTERR(err);
                }
            }
        }
        return localRange;
    }
    return cl::NullRange;
}

int Kniha::algFLOATcutting_voxel_minmaxbackproject(cl::Buffer& volume,
                                                   cl::Buffer& projection,
                                                   unsigned int& projectionOffset,
                                                   cl_double16& CM,
                                                   cl_double3& sourcePosition,
                                                   cl_double3& normalToDetector,
                                                   cl_int3& vdims,
                                                   cl_double3& voxelSizes,
                                                   cl_double3& volumeCenter,
                                                   cl_int2& pdims,
                                                   float globalScalingMultiplier,
                                                   cl::NDRange globalRange,
                                                   cl::NDRange _localRange,
                                                   bool blocking,
                                                   uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);
    cl_int2 dummy;
    auto exe = (*FLOATcutting_voxel_minmaxbackproject)(
        *eargs, volume, projection, projectionOffset, CM, sourcePosition, normalToDetector, vdims,
        voxelSizes, volumeCenter, pdims, globalScalingMultiplier, dummy);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOATcutting_voxel_project_barrier(cl::Buffer& volume,
                                                 cl::Buffer& projection,
                                                 unsigned int& projectionOffset,
                                                 cl_double16& CM,
                                                 cl_double3& sourcePosition,
                                                 cl_double3& normalToDetector,
                                                 cl_int3& vdims,
                                                 cl_double3& voxelSizes,
                                                 cl_double3& volumeCenter,
                                                 cl_int2& pdims,
                                                 float globalScalingMultiplier,
                                                 unsigned int LOCALARRAYSIZE,
                                                 cl::NDRange globalRange,
                                                 cl::NDRange _localRange,
                                                 bool blocking,
                                                 uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);
    cl::LocalSpaceArg localProjection = cl::Local(LOCALARRAYSIZE * sizeof(float));

    auto exe = (*FLOATcutting_voxel_project_barrier)(
        *eargs, volume, projection, localProjection, projectionOffset, CM, sourcePosition,
        normalToDetector, vdims, voxelSizes, volumeCenter, pdims, globalScalingMultiplier);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}
// utils.cl

int Kniha::algFLOATvector_NormSquarePartial(cl::Buffer& V,
                                            cl::Buffer& PARTIAL_OUT,
                                            unsigned int partialFrameSize,
                                            uint32_t partialFrameCount,
                                            bool blocking,
                                            uint32_t QID)
{
    cl::EnqueueArgs eargs(*Q[QID], cl::NDRange(partialFrameCount));
    auto exe = (*FLOATvector_NormSquarePartial)(eargs, V, PARTIAL_OUT, partialFrameSize);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOATvector_SumPartial(cl::Buffer& V,
                                     cl::Buffer& PARTIAL_OUT,
                                     unsigned int partialFrameSize,
                                     uint32_t partialFrameCount,
                                     bool blocking,
                                     uint32_t QID)
{
    cl::EnqueueArgs eargs(*Q[QID], cl::NDRange(partialFrameCount));
    auto exe = (*FLOATvector_SumPartial)(eargs, V, PARTIAL_OUT, partialFrameSize);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOATvector_MaxPartial(cl::Buffer& V,
                                     cl::Buffer& PARTIAL_OUT,
                                     unsigned int partialFrameSize,
                                     uint32_t partialFrameCount,
                                     bool blocking,
                                     uint32_t QID)
{
    cl::EnqueueArgs eargs(*Q[QID], cl::NDRange(partialFrameCount));
    auto exe = (*FLOATvector_MaxPartial)(eargs, V, PARTIAL_OUT, partialFrameSize);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algvector_NormSquarePartial(cl::Buffer& V,
                                       cl::Buffer& VSQUARE_OUT,
                                       unsigned int partialFrameSize,
                                       uint32_t partialFrameCount,
                                       bool blocking,
                                       uint32_t QID)
{
    cl::EnqueueArgs eargs(*Q[QID], cl::NDRange(partialFrameCount));
    auto exe = (*vector_NormSquarePartial)(eargs, V, VSQUARE_OUT, partialFrameSize);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algvector_SumPartial(cl::Buffer& V,
                                cl::Buffer& SUM_OUT,
                                unsigned int partialFrameSize,
                                uint32_t partialFrameCount,
                                bool blocking,
                                uint32_t QID)
{
    cl::EnqueueArgs eargs(*Q[QID], cl::NDRange(partialFrameCount));
    auto exe = (*vector_SumPartial)(eargs, V, SUM_OUT, partialFrameSize);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algvector_NormSquarePartial_barrier(cl::Buffer& V,
                                               cl::Buffer& V_red,
                                               unsigned long& VDIM,
                                               unsigned long& VDIM_ALIGNED,
                                               uint32_t workGroupSize,
                                               bool blocking,
                                               uint32_t QID)
{
    cl::EnqueueArgs eargs_red1(*Q[QID], cl::NDRange(VDIM_ALIGNED), cl::NDRange(workGroupSize));
    cl::LocalSpaceArg localsize = cl::Local(workGroupSize * sizeof(double));
    auto exe = (*vector_NormSquarePartial_barrier)(eargs_red1, V, V_red, localsize, VDIM);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algvector_SumPartial_barrier(cl::Buffer& V,
                                        cl::Buffer& V_red,
                                        unsigned long& VDIM,
                                        unsigned long& VDIM_ALIGNED,
                                        uint32_t workGroupSize,
                                        bool blocking,
                                        uint32_t QID)
{
    cl::EnqueueArgs eargs_red1(*Q[QID], cl::NDRange(VDIM_ALIGNED), cl::NDRange(workGroupSize));
    cl::LocalSpaceArg localsize = cl::Local(workGroupSize * sizeof(double));
    auto exe = (*vector_SumPartial_barrier)(eargs_red1, V, V_red, localsize, VDIM);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOATvector_copy(
    cl::Buffer& A, cl::Buffer& B, uint64_t size, bool blocking, uint32_t QID)
{
    cl::EnqueueArgs eargs(*Q[QID], cl::NDRange(size));
    auto exe = (*FLOATvector_copy)(eargs, A, B);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOATvector_copy_offset(
    cl::Buffer& A, cl::Buffer& B, unsigned long offset, uint64_t size, bool blocking, uint32_t QID)
{
    cl::EnqueueArgs eargs(*Q[QID], cl::NDRange(size));
    auto exe = (*FLOATvector_copy_offset)(eargs, A, B, offset);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOATvector_copy_offsets(cl::Buffer& A,
                                       unsigned long oA,
                                       cl::Buffer& B,
                                       unsigned long oB,
                                       uint64_t size,
                                       bool blocking,
                                       uint32_t QID)
{
    cl::EnqueueArgs eargs(*Q[QID], cl::NDRange(size));
    auto exe = (*FLOATvector_copy_offsets)(eargs, A, oA, B, oB);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}
int Kniha::algFLOATvector_A_equals_cB(
    cl::Buffer& A, cl::Buffer& B, float c, uint64_t size, bool blocking, uint32_t QID)
{
    cl::EnqueueArgs eargs(*Q[QID], cl::NDRange(size));
    auto exe = (*FLOATvector_A_equals_cB)(eargs, A, B, c);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}
int Kniha::algFLOATvector_A_equals_A_plus_cB(
    cl::Buffer& A, cl::Buffer& B, float c, uint64_t size, bool blocking, uint32_t QID)
{
    cl::EnqueueArgs eargs(*Q[QID], cl::NDRange(size));
    auto exe = (*FLOATvector_A_equals_A_plus_cB)(eargs, A, B, c);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}
int Kniha::algFLOATvector_A_equals_Ac_plus_B(
    cl::Buffer& A, cl::Buffer& B, float c, uint64_t size, bool blocking, uint32_t QID)
{
    cl::EnqueueArgs eargs(*Q[QID], cl::NDRange(size));
    auto exe = (*FLOATvector_A_equals_Ac_plus_B)(eargs, A, B, c);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}
int Kniha::algFLOATvector_A_equals_Ac_plus_Bd(
    cl::Buffer& A, cl::Buffer& B, float c, float d, uint64_t size, bool blocking, uint32_t QID)
{
    cl::EnqueueArgs eargs(*Q[QID], cl::NDRange(size));
    auto exe = (*FLOATvector_A_equals_Ac_plus_Bd)(eargs, A, B, c, d);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOATvector_C_equals_Ad_plus_Be(cl::Buffer& A,
                                              cl::Buffer& B,
                                              cl::Buffer& C,
                                              float d,
                                              float e,
                                              uint64_t size,
                                              bool blocking,
                                              uint32_t QID)
{
    cl::EnqueueArgs eargs(*Q[QID], cl::NDRange(size));
    auto exe = (*FLOATvector_C_equals_Ad_plus_Be)(eargs, A, B, C, d, e);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOATvector_A_equals_A_plus_cB_offset(cl::Buffer& A,
                                                    cl::Buffer& B,
                                                    float c,
                                                    unsigned long offset,
                                                    uint64_t size,
                                                    bool blocking,
                                                    uint32_t QID)
{
    cl::EnqueueArgs eargs(*Q[QID], cl::NDRange(size));
    auto exe = (*FLOATvector_A_equals_A_plus_cB_offset)(eargs, A, B, c, offset);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}
int Kniha::algFLOATvector_sqrt(cl::Buffer& A, uint64_t size, bool blocking, uint32_t QID)
{

    cl::EnqueueArgs eargs(*Q[QID], cl::NDRange(size));
    auto exe = (*FLOATvector_sqrt)(eargs, A);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}
int Kniha::algFLOATvector_square(cl::Buffer& A, uint64_t size, bool blocking, uint32_t QID)
{

    cl::EnqueueArgs eargs(*Q[QID], cl::NDRange(size));
    auto exe = (*FLOATvector_square)(eargs, A);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}
int Kniha::algFLOATvector_zero(cl::Buffer& A, uint64_t size, bool blocking, uint32_t QID)
{

    cl::EnqueueArgs eargs(*Q[QID], cl::NDRange(size));
    auto exe = (*FLOATvector_zero)(eargs, A);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}
int Kniha::algFLOATvector_zero_infinite_values(cl::Buffer& A,
                                               uint64_t size,
                                               bool blocking,
                                               uint32_t QID)
{
    cl::EnqueueArgs eargs(*Q[QID], cl::NDRange(size));
    auto exe = (*FLOATvector_zero_infinite_values)(eargs, A);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}
int Kniha::algFLOATvector_scale(cl::Buffer& A, float c, uint64_t size, bool blocking, uint32_t QID)
{
    cl::EnqueueArgs eargs(*Q[QID], cl::NDRange(size));
    auto exe = (*FLOATvector_scale)(eargs, A, c);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}
int Kniha::algFLOATvector_A_equals_A_plus_cB_offsets(cl::Buffer& A,
                                                     unsigned long oA,
                                                     cl::Buffer& B,
                                                     unsigned long oB,
                                                     float c,
                                                     uint64_t size,
                                                     bool blocking,
                                                     uint32_t QID)
{
    cl::EnqueueArgs eargs(*Q[QID], cl::NDRange(size));
    auto exe = (*FLOATvector_A_equals_A_plus_cB_offsets)(eargs, A, oA, B, oB, c);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}
int Kniha::algFLOATvector_B_equals_A_plus_B_offsets(cl::Buffer& A,
                                                    unsigned long oA,
                                                    cl::Buffer& B,
                                                    unsigned long oB,
                                                    uint64_t size,
                                                    bool blocking,
                                                    uint32_t QID)
{
    cl::EnqueueArgs eargs(*Q[QID], cl::NDRange(size));
    auto exe = (*FLOATvector_B_equals_A_plus_B_offsets)(eargs, A, oA, B, oB);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}
int Kniha::algFLOATvector_invert(cl::Buffer& A, uint64_t size, bool blocking, uint32_t QID)
{
    cl::EnqueueArgs eargs(*Q[QID], cl::NDRange(size));
    auto exe = (*FLOATvector_invert)(eargs, A);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}
int Kniha::algFLOATvector_invert_except_zero(cl::Buffer& A,
                                             uint64_t size,
                                             bool blocking,
                                             uint32_t QID)
{
    cl::EnqueueArgs eargs(*Q[QID], cl::NDRange(size));
    auto exe = (*FLOATvector_invert_except_zero)(eargs, A);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}
int Kniha::algFLOATvector_substitute_greater_than(
    cl::Buffer& A, float maxValue, float substitution, uint64_t size, bool blocking, uint32_t QID)
{
    cl::EnqueueArgs eargs(*Q[QID], cl::NDRange(size));
    auto exe = (*FLOATvector_substitute_greater_than)(eargs, A, maxValue, substitution);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}
int Kniha::algFLOATvector_substitute_lower_than(
    cl::Buffer& A, float minValue, float substitution, uint64_t size, bool blocking, uint32_t QID)
{
    cl::EnqueueArgs eargs(*Q[QID], cl::NDRange(size));
    auto exe = (*FLOATvector_substitute_lower_than)(eargs, A, minValue, substitution);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}
int Kniha::algFLOATvector_A_equals_A_times_B(
    cl::Buffer& A, cl::Buffer& B, uint64_t size, bool blocking, uint32_t QID)
{
    cl::EnqueueArgs eargs(*Q[QID], cl::NDRange(size));
    auto exe = (*FLOATvector_A_equals_A_times_B)(eargs, A, B);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}
int Kniha::algFLOATvector_C_equals_A_times_B(
    cl::Buffer& A, cl::Buffer& B, cl::Buffer& C, uint64_t size, bool blocking, uint32_t QID)
{
    cl::EnqueueArgs eargs(*Q[QID], cl::NDRange(size));
    auto exe = (*FLOATvector_C_equals_A_times_B)(eargs, A, B, C);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

// convolution.cl
int Kniha::algFLOATvector_2Dconvolution3x3ZeroBoundary(cl::Buffer& A,
                                                       cl::Buffer& B,
                                                       cl_int3& vdims,
                                                       cl_float16& convolutionKernel,
                                                       cl::NDRange globalRange,
                                                       cl::NDRange _localRange,
                                                       bool blocking,
                                                       uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);
    auto exe = (*FLOATvector_2Dconvolution3x3ZeroBoundary)(*eargs, A, B, vdims, convolutionKernel);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOATvector_2Dconvolution3x3ReflectionBoundary(cl::Buffer& A,
                                                             cl::Buffer& B,
                                                             cl_int3& vdims,
                                                             cl_float16& convolutionKernel,
                                                             cl::NDRange globalRange,
                                                             cl::NDRange _localRange,
                                                             bool blocking,
                                                             uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);
    auto exe
        = (*FLOATvector_2Dconvolution3x3ReflectionBoundary)(*eargs, A, B, vdims, convolutionKernel);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOATvector_3DconvolutionGradientSobelFeldmanReflectionBoundary(
    cl::Buffer& F,
    cl::Buffer& GX,
    cl::Buffer& GY,
    cl::Buffer& GZ,
    cl_int3& vdims,
    cl_float3& voxelSizes,
    cl::NDRange globalRange,
    cl::NDRange _localRange,
    bool blocking,
    uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);
    auto exe = (*FLOATvector_3DconvolutionGradientSobelFeldmanReflectionBoundary)(
        *eargs, F, GX, GY, GZ, vdims, voxelSizes);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOATvector_3DconvolutionGradientSobelFeldmanZeroBoundary(cl::Buffer& F,
                                                                        cl::Buffer& GX,
                                                                        cl::Buffer& GY,
                                                                        cl::Buffer& GZ,
                                                                        cl_int3& vdims,
                                                                        cl_float3& voxelSizes,
                                                                        cl::NDRange globalRange,
                                                                        cl::NDRange _localRange,
                                                                        bool blocking,
                                                                        uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);

    auto exe = (*FLOATvector_3DconvolutionGradientSobelFeldmanZeroBoundary)(*eargs, F, GX, GY, GZ,
                                                                            vdims, voxelSizes);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOATvector_3DconvolutionGradientFarid5x5x5(cl::Buffer& F,
                                                          cl::Buffer& GX,
                                                          cl::Buffer& GY,
                                                          cl::Buffer& GZ,
                                                          cl_int3& vdims,
                                                          cl_float3& voxelSizes,
                                                          int reflectionBoundary,
                                                          cl::NDRange globalRange,
                                                          cl::NDRange _localRange,
                                                          bool blocking,
                                                          uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);

    auto exe = (*FLOATvector_3DconvolutionGradientFarid5x5x5)(*eargs, F, GX, GY, GZ, vdims,
                                                              voxelSizes, reflectionBoundary);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOATvector_2DconvolutionGradientFarid5x5(cl::Buffer& F,
                                                        cl::Buffer& GX,
                                                        cl::Buffer& GY,
                                                        cl_int3& vdims,
                                                        cl_float3& voxelSizes,
                                                        int reflectionBoundary,
                                                        cl::NDRange globalRange,
                                                        cl::NDRange _localRange,
                                                        bool blocking,
                                                        uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);

    auto exe = (*FLOATvector_2DconvolutionGradientFarid5x5)(*eargs, F, GX, GY, vdims, voxelSizes,
                                                            reflectionBoundary);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOATvector_3DconvolutionLaplaceZeroBoundary(cl::Buffer& A,
                                                           cl::Buffer& B,
                                                           cl_int3& vdims,
                                                           cl_float3& voxelSizes,
                                                           cl::NDRange globalRange,
                                                           cl::NDRange _localRange,
                                                           bool blocking,
                                                           uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);

    auto exe = (*FLOATvector_3DconvolutionLaplaceZeroBoundary)(*eargs, A, B, vdims, voxelSizes);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOATvector_3DisotropicGradient(cl::Buffer& F,
                                              cl::Buffer& GX,
                                              cl::Buffer& GY,
                                              cl::Buffer& GZ,
                                              cl_int3& vdims,
                                              cl_float3& voxelSizes,
                                              cl::NDRange globalRange,
                                              cl::NDRange _localRange,
                                              bool blocking,
                                              uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);
    auto exe = (*FLOATvector_3DisotropicGradient)(*eargs, F, GX, GY, GZ, vdims, voxelSizes);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOATvector_2DisotropicGradient(cl::Buffer& F,
                                              cl::Buffer& GX,
                                              cl::Buffer& GY,
                                              cl_int3& vdims,
                                              cl_float3& voxelSizes,
                                              cl::NDRange globalRange,
                                              cl::NDRange _localRange,
                                              bool blocking,
                                              uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);
    auto exe = (*FLOATvector_2DisotropicGradient)(*eargs, F, GX, GY, vdims, voxelSizes);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOATvector_isotropicBackDx(cl::Buffer& F,
                                          cl::Buffer& DX,
                                          cl_int3& vdims,
                                          cl_float3& voxelSizes,
                                          cl::NDRange globalRange,
                                          cl::NDRange _localRange,
                                          bool blocking,
                                          uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);

    auto exe = (*FLOATvector_isotropicBackDx)(*eargs, F, DX, vdims, voxelSizes);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOATvector_isotropicBackDy(cl::Buffer& F,
                                          cl::Buffer& DY,
                                          cl_int3& vdims,
                                          cl_float3& voxelSizes,
                                          cl::NDRange globalRange,
                                          cl::NDRange _localRange,
                                          bool blocking,
                                          uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);

    auto exe = (*FLOATvector_isotropicBackDy)(*eargs, F, DY, vdims, voxelSizes);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOATvector_isotropicBackDz(cl::Buffer& F,
                                          cl::Buffer& DZ,
                                          cl_int3& vdims,
                                          cl_float3& voxelSizes,
                                          cl::NDRange globalRange,
                                          cl::NDRange _localRange,
                                          bool blocking,
                                          uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);

    auto exe = (*FLOATvector_isotropicBackDz)(*eargs, F, DZ, vdims, voxelSizes);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOATvector_isotropicBackDivergence2D(cl::Buffer& FX,
                                                    cl::Buffer& FY,
                                                    cl::Buffer& DIV,
                                                    cl_int3& vdims,
                                                    cl_float3& voxelSizes,
                                                    cl::NDRange globalRange,
                                                    cl::NDRange _localRange,
                                                    bool blocking,
                                                    uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);

    auto exe = (*FLOATvector_isotropicBackDivergence2D)(*eargs, FX, FY, DIV, vdims, voxelSizes);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

// pbct_cvp.cl
int Kniha::algFLOAT_pbct_cutting_voxel_project(cl::Buffer& volume,
                                               cl::Buffer& projection,
                                               unsigned long& projectionOffset,
                                               cl_double8& CM,
                                               cl_int3& vdims,
                                               cl_double3& voxelSizes,
                                               cl_double3& volumeCenter,
                                               cl_int2& pdims,
                                               float globalScalingMultiplier,
                                               cl::NDRange globalRange,
                                               cl::NDRange _localRange, // default cl::NullRange
                                               bool blocking,
                                               uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);

    auto exe = (*FLOAT_pbct_cutting_voxel_project)(*eargs, volume, projection, projectionOffset, CM,
                                                   vdims, voxelSizes, volumeCenter, pdims,
                                                   globalScalingMultiplier);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOAT_pbct_cutting_voxel_backproject(cl::Buffer& volume,
                                                   cl::Buffer& projection,
                                                   unsigned long& projectionOffset,
                                                   cl_double8& CM,
                                                   cl_int3& vdims,
                                                   cl_double3& voxelSizes,
                                                   cl_double3& volumeCenter,
                                                   cl_int2& pdims,
                                                   float globalScalingMultiplier,
                                                   cl::NDRange globalRange,
                                                   cl::NDRange _localRange, // default cl::NullRange
                                                   bool blocking,
                                                   uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);

    auto exe = (*FLOAT_pbct_cutting_voxel_backproject)(*eargs, volume, projection, projectionOffset,
                                                       CM, vdims, voxelSizes, volumeCenter, pdims,
                                                       globalScalingMultiplier);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

// pbct_cvp_barrier.cl
int Kniha::algFLOAT_pbct_cutting_voxel_project_barrier(cl::Buffer& volume,
                                                       cl::Buffer& projection,
                                                       unsigned long& projectionOffset,
                                                       cl_double8& CM,
                                                       cl_int3& vdims,
                                                       cl_double3& voxelSizes,
                                                       cl_double3& volumeCenter,
                                                       cl_int2& pdims,
                                                       float globalScalingMultiplier,
                                                       unsigned int LOCALARRAYSIZE,
                                                       cl::NDRange globalRange,
                                                       cl::NDRange _localRange,
                                                       bool blocking,
                                                       uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    cl::LocalSpaceArg localProjection = cl::Local(LOCALARRAYSIZE * sizeof(float));
    /*eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);
        auto exe = (*FLOAT_pbct_cutting_voxel_project_barrier)(
            *eargs, volume, projection, localProjection, projectionOffset, CM, vdims, voxelSizes,
            volumeCenter, pdims, globalScalingMultiplier);*/
    cl::Kernel kernel = cl::Kernel(*program, "FLOAT_pbct_cutting_voxel_project_barrier");
    kernel.setArg(0, volume);
    kernel.setArg(1, projection);
    kernel.setArg(2, localProjection);
    kernel.setArg(3, projectionOffset);
    kernel.setArg(4, CM);
    kernel.setArg(5, vdims);
    kernel.setArg(6, voxelSizes);
    kernel.setArg(7, volumeCenter);
    kernel.setArg(8, pdims);
    kernel.setArg(9, globalScalingMultiplier);
    cl::NDRange nulloffset = cl::NullRange;
    cl::Event exe;
    cl_int cl_info_id
        = Q[QID]->enqueueNDRangeKernel(kernel, nulloffset, globalRange, localRange, nullptr, &exe);
    if(cl_info_id != CL_SUCCESS)
    {
        std::string command_type_string = infoString(cl_info_id);
        err = io::xprintf(
            "Error in enqueueNDRangeKernel FLOAT_pbct_cutting_voxel_project_barrier cl_info_id=%d, "
            "command_type_string=%s!",
            cl_info_id, command_type_string.c_str());
        KCTERR(err);
    }
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

// pbct2d_cvp.cl
int Kniha::algFLOAT_pbct2d_cutting_voxel_project(cl::Buffer& volume,
                                                 cl::Buffer& projection,
                                                 unsigned long projectionOffset,
                                                 cl_double3& CM,
                                                 cl_int3& vdims,
                                                 cl_double3& voxelSizes,
                                                 cl_double2& volumeCenter,
                                                 cl_int2& pdims,
                                                 float& globalScalingMultiplier,
                                                 int& k_from,
                                                 int& k_count,
                                                 cl::NDRange globalRange,
                                                 cl::NDRange _localRange, // default cl::NullRange
                                                 bool blocking,
                                                 uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);
    /*
        auto exe = (*FLOAT_pbct2d_cutting_voxel_project)(*eargs, volume, projection,
       projectionOffset, CM, vdims, voxelSizes, volumeCenter, pdims, globalScalingMultiplier,
       k_from, k_count);
    */
    // Improve error handling, see
    // https://stackoverflow.com/questions/14088030/opencl-returns-error-58-while-executing-larga-amount-of-data
    cl::Kernel kernel = cl::Kernel(*program, "FLOAT_pbct2d_cutting_voxel_project");
    kernel.setArg(0, volume);
    kernel.setArg(1, projection);
    kernel.setArg(2, projectionOffset);
    kernel.setArg(3, CM);
    kernel.setArg(4, vdims);
    kernel.setArg(5, voxelSizes);
    kernel.setArg(6, volumeCenter);
    kernel.setArg(7, pdims);
    kernel.setArg(8, globalScalingMultiplier);
    kernel.setArg(9, k_from);
    kernel.setArg(10, k_count);
    cl::NDRange nulloffset = cl::NullRange;
    cl::Event exe;
    cl_int cl_info_id
        = Q[QID]->enqueueNDRangeKernel(kernel, nulloffset, globalRange, localRange, nullptr, &exe);
    if(cl_info_id != CL_SUCCESS)
    {
        std::string command_type_string = infoString(cl_info_id);
        err = io::xprintf(
            "Error in enqueueNDRangeKernel FLOAT_pbct2d_cutting_voxel_project cl_info_id=%d, "
            "command_type_string=%s!",
            cl_info_id, command_type_string.c_str());
        KCTERR(err);
    }

    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOAT_pbct2d_cutting_voxel_kaczmarz_product(
    cl::Buffer& projection,
    unsigned long projectionOffset,
    cl_double3& CM,
    cl_int3& vdims,
    cl_double3& voxelSizes,
    cl_double2& volumeCenter,
    cl_int2& pdims,
    float& globalScalingMultiplier,
    int& k_from,
    int& k_count,
    cl::NDRange globalRange,
    cl::NDRange _localRange, // default cl::NullRange
    bool blocking,
    uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);
    /*
        auto exe = (*FLOAT_pbct2d_cutting_voxel_project)(*eargs, volume, projection,
       projectionOffset, CM, vdims, voxelSizes, volumeCenter, pdims, globalScalingMultiplier,
       k_from, k_count);
    */
    // Improve error handling, see
    // https://stackoverflow.com/questions/14088030/opencl-returns-error-58-while-executing-larga-amount-of-data
    cl::Kernel kernel = cl::Kernel(*program, "FLOAT_pbct2d_cutting_voxel_kaczmarz_product");
    kernel.setArg(0, projection);
    kernel.setArg(1, projectionOffset);
    kernel.setArg(2, CM);
    kernel.setArg(3, vdims);
    kernel.setArg(4, voxelSizes);
    kernel.setArg(5, volumeCenter);
    kernel.setArg(6, pdims);
    kernel.setArg(7, globalScalingMultiplier);
    kernel.setArg(8, k_from);
    kernel.setArg(9, k_count);
    cl::NDRange nulloffset = cl::NullRange;
    cl::Event exe;
    cl_int cl_info_id
        = Q[QID]->enqueueNDRangeKernel(kernel, nulloffset, globalRange, localRange, nullptr, &exe);
    if(cl_info_id != CL_SUCCESS)
    {
        std::string command_type_string = infoString(cl_info_id);
        err = io::xprintf(
            "Error in enqueueNDRangeKernel FLOAT_pbct2d_cutting_voxel_project cl_info_id=%d, "
            "command_type_string=%s!",
            cl_info_id, command_type_string.c_str());
        KCTERR(err);
    }

    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOAT_pbct2d_cutting_voxel_backproject(
    cl::Buffer& volume,
    cl::Buffer& projection,
    unsigned long projectionOffset,
    cl_double3& CM,
    cl_int3& vdims,
    cl_double3& voxelSizes,
    cl_double2& volumeCenter,
    cl_int2& pdims,
    float& globalScalingMultiplier,
    int& k_from,
    int& k_count,
    cl::NDRange globalRange,
    cl::NDRange _localRange, // default cl::NullRange
    bool blocking,
    uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);

    auto exe = (*FLOAT_pbct2d_cutting_voxel_backproject)(
        *eargs, volume, projection, projectionOffset, CM, vdims, voxelSizes, volumeCenter, pdims,
        globalScalingMultiplier, k_from, k_count);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOAT_pbct2d_cutting_voxel_backproject_kaczmarz(
    cl::Buffer& volume,
    cl::Buffer& projection,
    unsigned long projectionOffset,
    cl_double3& CM,
    cl_int3& vdims,
    cl_double3& voxelSizes,
    cl_double2& volumeCenter,
    cl_int2& pdims,
    float& globalScalingMultiplier,
    int& k_from,
    int& k_count,
    cl::NDRange globalRange,
    cl::NDRange _localRange, // default cl::NullRange
    bool blocking,
    uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);

    auto exe = (*FLOAT_pbct2d_cutting_voxel_backproject_kaczmarz)(
        *eargs, volume, projection, projectionOffset, CM, vdims, voxelSizes, volumeCenter, pdims,
        globalScalingMultiplier, k_from, k_count);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOAT_pbct2d_cutting_voxel_jacobi_vector(
    cl::Buffer& volume,
    cl_double3& CM,
    cl_int3& vdims,
    cl_double3& voxelSizes,
    cl_double2& volumeCenter,
    cl_int2& pdims,
    float& globalScalingMultiplier,
    int& k_from,
    int& k_count,
    cl::NDRange globalRange,
    cl::NDRange _localRange, // default cl::NullRange
    bool blocking,
    uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);

    auto exe = (*FLOAT_pbct2d_cutting_voxel_jacobi_vector)(
        *eargs, volume, CM, vdims, voxelSizes, volumeCenter, pdims, globalScalingMultiplier, k_from,
        k_count);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

// pbct2d_cvp_barrier.cl
int Kniha::algFLOAT_pbct2d_cutting_voxel_project_barrier(
    cl::Buffer& volume,
    cl::Buffer& projection,
    unsigned long& projectionOffset,
    cl_double3& CM,
    cl_int3& vdims,
    cl_double3& voxelSizes,
    cl_double2& volumeCenter,
    cl_int2& pdims,
    float globalScalingMultiplier,
    int& k_from,
    int& k_count,
    unsigned int LOCALARRAYSIZE,
    cl::NDRange globalRange,
    cl::NDRange _localRange, // default cl::NullRange
    bool blocking, // default false
    uint32_t QID) // default 0
{
    // https://stackoverflow.com/questions/14088030/opencl-returns-error-58-while-executing-larga-amount-of-data
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    cl::LocalSpaceArg localProjection = cl::Local(LOCALARRAYSIZE * sizeof(float));
    cl::Kernel kernel = cl::Kernel(*program, "FLOAT_pbct2d_cutting_voxel_project_barrier");
    kernel.setArg(0, volume);
    kernel.setArg(1, projection);
    kernel.setArg(2, localProjection);
    kernel.setArg(3, projectionOffset);
    kernel.setArg(4, CM);
    kernel.setArg(5, vdims);
    kernel.setArg(6, voxelSizes);
    kernel.setArg(7, volumeCenter);
    kernel.setArg(8, pdims);
    kernel.setArg(9, globalScalingMultiplier);
    kernel.setArg(10, k_from);
    kernel.setArg(11, k_count);
    cl::NDRange nulloffset = cl::NullRange;
    cl::Event exe;
    cl_int cl_info_id
        = Q[QID]->enqueueNDRangeKernel(kernel, nulloffset, globalRange, localRange, nullptr, &exe);
    if(cl_info_id != CL_SUCCESS)
    {
        std::string command_type_string = infoString(cl_info_id);
        err = io::xprintf("Error in enqueueNDRangeKernel "
                          "FLOAT_pbct2d_cutting_voxel_project_barrier cl_info_id=%d, "
                          "command_type_string=%s!",
                          cl_info_id, command_type_string.c_str());
        KCTERR(err);
    }
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

// proximal.cl
int Kniha::algFLOATvector_infProjectionToLambda2DBall(
    cl::Buffer& G1, cl::Buffer& G2, float lambda, uint64_t size, bool blocking, uint32_t QID)
{
    cl::EnqueueArgs eargs(*Q[QID], cl::NDRange(size));
    auto exe = (*FLOATvector_infProjectionToLambda2DBall)(eargs, G1, G2, lambda);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOATvector_distL1ProxSoftThreasholding(
    cl::Buffer& U0, cl::Buffer& XPROX, float omega, uint64_t size, bool blocking, uint32_t QID)
{
    cl::EnqueueArgs eargs(*Q[QID], cl::NDRange(size));
    auto exe = (*FLOATvector_distL1ProxSoftThreasholding)(eargs, U0, XPROX, omega);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}



// gradient.cl

int Kniha::algFLOATvector_Gradient2D_forwardDifference_2point(cl::Buffer& F,
                                                              cl::Buffer& GX,
                                                              cl::Buffer& GY,
                                                              cl_int3& vdims,
                                                              cl_float3& voxelSizes,
                                                              cl::NDRange globalRange,
                                                              cl::NDRange _localRange,
                                                              bool blocking,
                                                              uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);
    auto exe
        = (*FLOATvector_Gradient2D_forwardDifference_2point)(*eargs, F, GX, GY, vdims, voxelSizes);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

// Similarly, define the rest for other kernels

int Kniha::algFLOATvector_Gradient2D_forwardDifference_2point_adjoint(cl::Buffer& GX,
                                                                      cl::Buffer& GY,
                                                                      cl::Buffer& D,
                                                                      cl_int3& vdims,
                                                                      cl_float3& voxelSizes,
                                                                      cl::NDRange globalRange,
                                                                      cl::NDRange _localRange,
                                                                      bool blocking,
                                                                      uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);
    auto exe = (*FLOATvector_Gradient2D_forwardDifference_2point_adjoint)(*eargs, GX, GY, D, vdims,
                                                                          voxelSizes);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOATvector_Gradient2D_forwardDifference_3point(cl::Buffer& F,
                                                              cl::Buffer& GX,
                                                              cl::Buffer& GY,
                                                              cl_int3& vdims,
                                                              cl_float3& voxelSizes,
                                                              cl::NDRange globalRange,
                                                              cl::NDRange _localRange,
                                                              bool blocking,
                                                              uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);
    auto exe
        = (*FLOATvector_Gradient2D_forwardDifference_3point)(*eargs, F, GX, GY, vdims, voxelSizes);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOATvector_Gradient2D_forwardDifference_3point_adjoint(cl::Buffer& GX,
                                                                      cl::Buffer& GY,
                                                                      cl::Buffer& D,
                                                                      cl_int3& vdims,
                                                                      cl_float3& voxelSizes,
                                                                      cl::NDRange globalRange,
                                                                      cl::NDRange _localRange,
                                                                      bool blocking,
                                                                      uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);
    auto exe = (*FLOATvector_Gradient2D_forwardDifference_3point_adjoint)(*eargs, GX, GY, D, vdims,
                                                                          voxelSizes);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOATvector_Gradient2D_forwardDifference_4point(cl::Buffer& F,
                                                              cl::Buffer& GX,
                                                              cl::Buffer& GY,
                                                              cl_int3& vdims,
                                                              cl_float3& voxelSizes,
                                                              cl::NDRange globalRange,
                                                              cl::NDRange _localRange,
                                                              bool blocking,
                                                              uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);
    auto exe
        = (*FLOATvector_Gradient2D_forwardDifference_4point)(*eargs, F, GX, GY, vdims, voxelSizes);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOATvector_Gradient2D_forwardDifference_4point_adjoint(cl::Buffer& GX,
                                                                      cl::Buffer& GY,
                                                                      cl::Buffer& D,
                                                                      cl_int3& vdims,
                                                                      cl_float3& voxelSizes,
                                                                      cl::NDRange globalRange,
                                                                      cl::NDRange _localRange,
                                                                      bool blocking,
                                                                      uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);
    auto exe = (*FLOATvector_Gradient2D_forwardDifference_4point_adjoint)(*eargs, GX, GY, D, vdims,
                                                                          voxelSizes);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOATvector_Gradient2D_forwardDifference_5point(cl::Buffer& F,
                                                              cl::Buffer& GX,
                                                              cl::Buffer& GY,
                                                              cl_int3& vdims,
                                                              cl_float3& voxelSizes,
                                                              cl::NDRange globalRange,
                                                              cl::NDRange _localRange,
                                                              bool blocking,
                                                              uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);
    auto exe
        = (*FLOATvector_Gradient2D_forwardDifference_5point)(*eargs, F, GX, GY, vdims, voxelSizes);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOATvector_Gradient2D_forwardDifference_5point_adjoint(cl::Buffer& GX,
                                                                      cl::Buffer& GY,
                                                                      cl::Buffer& D,
                                                                      cl_int3& vdims,
                                                                      cl_float3& voxelSizes,
                                                                      cl::NDRange globalRange,
                                                                      cl::NDRange _localRange,
                                                                      bool blocking,
                                                                      uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);
    auto exe = (*FLOATvector_Gradient2D_forwardDifference_5point_adjoint)(*eargs, GX, GY, D, vdims,
                                                                          voxelSizes);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOATvector_Gradient2D_forwardDifference_6point(cl::Buffer& F,
                                                              cl::Buffer& GX,
                                                              cl::Buffer& GY,
                                                              cl_int3& vdims,
                                                              cl_float3& voxelSizes,
                                                              cl::NDRange globalRange,
                                                              cl::NDRange _localRange,
                                                              bool blocking,
                                                              uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);
    auto exe
        = (*FLOATvector_Gradient2D_forwardDifference_6point)(*eargs, F, GX, GY, vdims, voxelSizes);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOATvector_Gradient2D_forwardDifference_6point_adjoint(cl::Buffer& GX,
                                                                      cl::Buffer& GY,
                                                                      cl::Buffer& D,
                                                                      cl_int3& vdims,
                                                                      cl_float3& voxelSizes,
                                                                      cl::NDRange globalRange,
                                                                      cl::NDRange _localRange,
                                                                      bool blocking,
                                                                      uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);
    auto exe = (*FLOATvector_Gradient2D_forwardDifference_6point_adjoint)(*eargs, GX, GY, D, vdims,
                                                                          voxelSizes);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOATvector_Gradient2D_forwardDifference_7point(cl::Buffer& F,
                                                              cl::Buffer& GX,
                                                              cl::Buffer& GY,
                                                              cl_int3& vdims,
                                                              cl_float3& voxelSizes,
                                                              cl::NDRange globalRange,
                                                              cl::NDRange _localRange,
                                                              bool blocking,
                                                              uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);
    auto exe
        = (*FLOATvector_Gradient2D_forwardDifference_7point)(*eargs, F, GX, GY, vdims, voxelSizes);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOATvector_Gradient2D_forwardDifference_7point_adjoint(cl::Buffer& GX,
                                                                      cl::Buffer& GY,
                                                                      cl::Buffer& D,
                                                                      cl_int3& vdims,
                                                                      cl_float3& voxelSizes,
                                                                      cl::NDRange globalRange,
                                                                      cl::NDRange _localRange,
                                                                      bool blocking,
                                                                      uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);
    auto exe = (*FLOATvector_Gradient2D_forwardDifference_7point_adjoint)(*eargs, GX, GY, D, vdims,
                                                                          voxelSizes);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOATvector_Gradient2D_centralDifference_3point(cl::Buffer& F,
                                                              cl::Buffer& GX,
                                                              cl::Buffer& GY,
                                                              cl_int3& vdims,
                                                              cl_float3& voxelSizes,
                                                              cl::NDRange globalRange,
                                                              cl::NDRange _localRange,
                                                              bool blocking,
                                                              uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);
    auto exe
        = (*FLOATvector_Gradient2D_centralDifference_3point)(*eargs, F, GX, GY, vdims, voxelSizes);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

// Similarly, define the rest for other kernels

int Kniha::algFLOATvector_Gradient2D_centralDifference_3point_adjoint(cl::Buffer& GX,
                                                                      cl::Buffer& GY,
                                                                      cl::Buffer& D,
                                                                      cl_int3& vdims,
                                                                      cl_float3& voxelSizes,
                                                                      cl::NDRange globalRange,
                                                                      cl::NDRange _localRange,
                                                                      bool blocking,
                                                                      uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);
    auto exe = (*FLOATvector_Gradient2D_centralDifference_3point_adjoint)(*eargs, GX, GY, D, vdims,
                                                                          voxelSizes);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

int Kniha::algFLOATvector_Gradient2D_centralDifference_5point(cl::Buffer& F,
                                                              cl::Buffer& GX,
                                                              cl::Buffer& GY,
                                                              cl_int3& vdims,
                                                              cl_float3& voxelSizes,
                                                              cl::NDRange globalRange,
                                                              cl::NDRange _localRange,
                                                              bool blocking,
                                                              uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);
    auto exe
        = (*FLOATvector_Gradient2D_centralDifference_5point)(*eargs, F, GX, GY, vdims, voxelSizes);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

// Similarly, define the rest for other kernels

int Kniha::algFLOATvector_Gradient2D_centralDifference_5point_adjoint(cl::Buffer& GX,
                                                                      cl::Buffer& GY,
                                                                      cl::Buffer& D,
                                                                      cl_int3& vdims,
                                                                      cl_float3& voxelSizes,
                                                                      cl::NDRange globalRange,
                                                                      cl::NDRange _localRange,
                                                                      bool blocking,
                                                                      uint32_t QID)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    cl::NDRange localRange = assignLocalRange(_localRange, globalRange);
    eargs = std::make_shared<cl::EnqueueArgs>(*Q[QID], globalRange, localRange);
    auto exe = (*FLOATvector_Gradient2D_centralDifference_5point_adjoint)(*eargs, GX, GY, D, vdims,
                                                                          voxelSizes);
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}

} // namespace KCT
