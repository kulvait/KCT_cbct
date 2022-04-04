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

int Kniha::initializeOpenCL(uint32_t platformId,
                            uint32_t* deviceIds,
                            uint32_t deviceIdsLength,
                            std::string xpath,
                            bool debug,
                            bool relaxed)
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
    std::string projectorSource = io::fileToString(clFile);
    cl::Program program(*context, projectorSource);
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
    cl_int inf = program.build(devices, options.c_str());
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
            cl_build_status s = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);
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

void Kniha::CLINCLUDEconvolution()
{
    insertCLFile("opencl/convolution.cl");
    callbacks.emplace_back([this](cl::Program program) {
        {
            auto& ptr = FLOATvector_2Dconvolution3x3;
            std::string str = "FLOATvector_2Dconvolution3x3";
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

void Kniha::insertCLFile(std::string f)
{
    if(std::find(CLFiles.begin(), CLFiles.end(), f) == CLFiles.end())
    // Vector does not contain f yet
    {
        CLFiles.emplace_back(f);
    }
}

int Kniha::handleKernelExecution(cl::Event exe, bool blocking, std::string& errout)
{
    cl_int inf;
    std::string kernelName;
    if(blocking)
    {
        exe.wait();
    }
    exe.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS, &inf);
    if(blocking)
    {
        if(inf != CL_COMPLETE)
        {
            exe.getInfo(CL_KERNEL_FUNCTION_NAME, &kernelName);
            errout = io::xprintf(
                "Kernel %s COMMAND_EXECUTION_STATUS is %d that is different from CL_COMPLETE!",
                kernelName.c_str(), inf);
            return 1;
        }
    } else if(inf != CL_COMPLETE && inf != CL_QUEUED && inf != CL_SUBMITTED && inf != CL_RUNNING)
    {
        exe.getInfo(CL_KERNEL_FUNCTION_NAME, &kernelName);
        errout = io::xprintf("Kernel %s COMMAND_EXECUTION_STATUS is %d implying an error!",
                             kernelName.c_str(), inf);
        return 2;
    }
    return 0;
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
                                                   cl::NDRange& globalRange,
                                                   std::shared_ptr<cl::NDRange> localRange,
                                                   bool blocking)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    if(localRange != nullptr)
    {
        eargs = std::make_shared<cl::EnqueueArgs>(*Q[0], globalRange, *localRange);
    } else
    {
        eargs = std::make_shared<cl::EnqueueArgs>(*Q[0], globalRange);
    }
    cl_int2 dummy;
    auto exe = (*FLOATcutting_voxel_minmaxbackproject)(
        *eargs, volume, projection, projectionOffset, CM, sourcePosition, normalToDetector, vdims,
        voxelSizes, volumeCenter, pdims, globalScalingMultiplier, dummy);
    if(blocking)
    {
        exe.wait();
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
                                                 cl::NDRange& globalRange,
                                                 std::shared_ptr<cl::NDRange> localRange,
                                                 bool blocking)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    if(localRange != nullptr)
    {
        eargs = std::make_shared<cl::EnqueueArgs>(*Q[0], globalRange, *localRange);
    } else
    {
        eargs = std::make_shared<cl::EnqueueArgs>(*Q[0], globalRange);
    }
    cl::LocalSpaceArg localProjection = cl::Local(LOCALARRAYSIZE * sizeof(float));
    auto lambda = [](cl_event e, cl_int status, void* data) {
        if(status != CL_COMPLETE)
        {
            LOGE << io::xprintf("Terminated with the status different than CL_COMPLETE");
        }
    };

    auto exe = (*FLOATcutting_voxel_project_barrier)(
        *eargs, volume, projection, localProjection, projectionOffset, CM, sourcePosition,
        normalToDetector, vdims, voxelSizes, volumeCenter, pdims, globalScalingMultiplier);
    exe.setCallback(CL_COMPLETE, lambda);
    if(blocking)
    {
        exe.wait();
    }
    return 0;
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
int Kniha::algFLOATvector_zero_infinite_values(cl::Buffer& A, uint64_t size, bool blocking)
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
int Kniha::algFLOATvector_B_equals_A_plus_B_offsets(
    cl::Buffer& A, unsigned int oA, cl::Buffer& B, unsigned int oB, uint64_t size, bool blocking)
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
int Kniha::algFLOATvector_invert_except_zero(cl::Buffer& A, uint64_t size, bool blocking)
{
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(size));
    auto exe = (*FLOATvector_invert_except_zero)(eargs, A);
    if(blocking)
    {
        exe.wait();
    }
    return 0;
}
int Kniha::algFLOATvector_substitute_greater_than(
    cl::Buffer& A, float maxValue, float substitution, uint64_t size, bool blocking)
{
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(size));
    auto exe = (*FLOATvector_substitute_greater_than)(eargs, A, maxValue, substitution);
    if(blocking)
    {
        exe.wait();
    }
    return 0;
}
int Kniha::algFLOATvector_substitute_lower_than(
    cl::Buffer& A, float minValue, float substitution, uint64_t size, bool blocking)
{
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(size));
    auto exe = (*FLOATvector_substitute_lower_than)(eargs, A, minValue, substitution);
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
int Kniha::algvector_NormSquarePartial_barrier(cl::Buffer& V,
                                               cl::Buffer& V_red,
                                               unsigned int& VDIM,
                                               unsigned int& VDIM_ALIGNED,
                                               uint32_t workGroupSize,
                                               bool blocking)
{
    cl::EnqueueArgs eargs_red1(*Q[0], cl::NDRange(VDIM_ALIGNED), cl::NDRange(workGroupSize));
    cl::LocalSpaceArg localsize = cl::Local(workGroupSize * sizeof(double));
    auto exe = (*vector_NormSquarePartial_barrier)(eargs_red1, V, V_red, localsize, VDIM);
    std::string err;
    if(handleKernelExecution(exe, blocking, err))
    {
        KCTERR(err);
    }
    return 0;
}
int Kniha::algvector_SumPartial_barrier(cl::Buffer& V,
                                        cl::Buffer& V_red,
                                        unsigned int& VDIM,
                                        unsigned int& VDIM_ALIGNED,
                                        uint32_t workGroupSize,
                                        bool blocking)
{
    cl_int inf;
    std::string err;
    cl::EnqueueArgs eargs_red1(*Q[0], cl::NDRange(VDIM_ALIGNED), cl::NDRange(workGroupSize));
    cl::LocalSpaceArg localsize = cl::Local(workGroupSize * sizeof(double));
    auto exe = (*vector_SumPartial_barrier)(eargs_red1, V, V_red, localsize, VDIM);
    if(blocking)
    {
        exe.wait();
    }
    exe.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS, &inf);
    if((blocking && inf != CL_COMPLETE)
       || (inf != CL_QUEUED && inf != CL_SUBMITTED && inf != CL_RUNNING))
    {
        err = io::xprintf("COMMAND_EXECUTION_STATUS is %d", inf);
        KCTERR(err);
    }
    return 0;
}
// convolution.cl
int Kniha::algFLOATvector_2Dconvolution3x3(cl::Buffer& A,
                                           cl::Buffer& B,
                                           cl_int3& vdims,
                                           cl_float16& convolutionKernel,
                                           cl::NDRange& globalRange,
                                           std::shared_ptr<cl::NDRange> localRange,
                                           bool blocking)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    if(localRange != nullptr)
    {
        eargs = std::make_shared<cl::EnqueueArgs>(*Q[0], globalRange, *localRange);
    } else
    {
        eargs = std::make_shared<cl::EnqueueArgs>(*Q[0], globalRange);
    }
    auto lambda = [](cl_event e, cl_int status, void* data) {
        if(status != CL_COMPLETE)
        {
            LOGE << io::xprintf("Terminated with the status different than CL_COMPLETE");
        }
    };

    auto exe = (*FLOATvector_2Dconvolution3x3)(*eargs, A, B, vdims, convolutionKernel);
    exe.setCallback(CL_COMPLETE, lambda);
    if(blocking)
    {
        exe.wait();
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
    cl::NDRange& globalRange,
    std::shared_ptr<cl::NDRange> localRange,
    bool blocking)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    if(localRange != nullptr)
    {
        eargs = std::make_shared<cl::EnqueueArgs>(*Q[0], globalRange, *localRange);
    } else
    {
        eargs = std::make_shared<cl::EnqueueArgs>(*Q[0], globalRange);
    }
    auto lambda = [](cl_event e, cl_int status, void* data) {
        if(status != CL_COMPLETE)
        {
            LOGE << io::xprintf("Terminated with the status different than CL_COMPLETE");
        }
    };
    auto exe = (*FLOATvector_3DconvolutionGradientSobelFeldmanReflectionBoundary)(
        *eargs, F, GX, GY, GZ, vdims, voxelSizes);
    exe.setCallback(CL_COMPLETE, lambda);
    if(blocking)
    {
        exe.wait();
    }
    return 0;
}

int Kniha::algFLOATvector_3DconvolutionGradientSobelFeldmanZeroBoundary(
    cl::Buffer& F,
    cl::Buffer& GX,
    cl::Buffer& GY,
    cl::Buffer& GZ,
    cl_int3& vdims,
    cl_float3& voxelSizes,
    cl::NDRange& globalRange,
    std::shared_ptr<cl::NDRange> localRange,
    bool blocking)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    if(localRange != nullptr)
    {
        eargs = std::make_shared<cl::EnqueueArgs>(*Q[0], globalRange, *localRange);
    } else
    {
        eargs = std::make_shared<cl::EnqueueArgs>(*Q[0], globalRange);
    }
    auto lambda = [](cl_event e, cl_int status, void* data) {
        if(status != CL_COMPLETE)
        {
            LOGE << io::xprintf("Terminated with the status different than CL_COMPLETE");
        }
    };
    auto exe = (*FLOATvector_3DconvolutionGradientSobelFeldmanZeroBoundary)(*eargs, F, GX, GY, GZ,
                                                                            vdims, voxelSizes);
    exe.setCallback(CL_COMPLETE, lambda);
    if(blocking)
    {
        exe.wait();
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
                                                          cl::NDRange& globalRange,
                                                          std::shared_ptr<cl::NDRange> localRange,
                                                          bool blocking)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    if(localRange != nullptr)
    {
        eargs = std::make_shared<cl::EnqueueArgs>(*Q[0], globalRange, *localRange);
    } else
    {
        eargs = std::make_shared<cl::EnqueueArgs>(*Q[0], globalRange);
    }
    auto lambda = [](cl_event e, cl_int status, void* data) {
        if(status != CL_COMPLETE)
        {
            LOGE << io::xprintf("Terminated with the status different than CL_COMPLETE");
        }
    };
    auto exe = (*FLOATvector_3DconvolutionGradientFarid5x5x5)(*eargs, F, GX, GY, GZ, vdims,
                                                              voxelSizes, reflectionBoundary);
    exe.setCallback(CL_COMPLETE, lambda);
    if(blocking)
    {
        exe.wait();
    }
    return 0;
}

int Kniha::algFLOATvector_2DconvolutionGradientFarid5x5(cl::Buffer& F,
                                                        cl::Buffer& GX,
                                                        cl::Buffer& GY,
                                                        cl_int3& vdims,
                                                        cl_float3& voxelSizes,
                                                        int reflectionBoundary,
                                                        cl::NDRange& globalRange,
                                                        std::shared_ptr<cl::NDRange> localRange,
                                                        bool blocking)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    if(localRange != nullptr)
    {
        eargs = std::make_shared<cl::EnqueueArgs>(*Q[0], globalRange, *localRange);
    } else
    {
        eargs = std::make_shared<cl::EnqueueArgs>(*Q[0], globalRange);
    }
    auto lambda = [](cl_event e, cl_int status, void* data) {
        if(status != CL_COMPLETE)
        {
            LOGE << io::xprintf("Terminated with the status different than CL_COMPLETE");
        }
    };
    auto exe = (*FLOATvector_2DconvolutionGradientFarid5x5)(*eargs, F, GX, GY, vdims, voxelSizes,
                                                            reflectionBoundary);
    exe.setCallback(CL_COMPLETE, lambda);
    if(blocking)
    {
        exe.wait();
    }
    return 0;
}

int Kniha::algFLOATvector_3DconvolutionLaplaceZeroBoundary(cl::Buffer& A,
                                                           cl::Buffer& B,
                                                           cl_int3& vdims,
                                                           cl_float3& voxelSizes,
                                                           cl::NDRange& globalRange,
                                                           std::shared_ptr<cl::NDRange> localRange,
                                                           bool blocking)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    if(localRange != nullptr)
    {
        eargs = std::make_shared<cl::EnqueueArgs>(*Q[0], globalRange, *localRange);
    } else
    {
        eargs = std::make_shared<cl::EnqueueArgs>(*Q[0], globalRange);
    }
    auto lambda = [](cl_event e, cl_int status, void* data) {
        if(status != CL_COMPLETE)
        {
            LOGE << io::xprintf("Terminated with the status different than CL_COMPLETE");
        }
    };

    auto exe = (*FLOATvector_3DconvolutionLaplaceZeroBoundary)(*eargs, A, B, vdims, voxelSizes);
    exe.setCallback(CL_COMPLETE, lambda);
    if(blocking)
    {
        exe.wait();
    }
    return 0;
}

int Kniha::algFLOATvector_3DisotropicGradient(cl::Buffer& F,
                                              cl::Buffer& GX,
                                              cl::Buffer& GY,
                                              cl::Buffer& GZ,
                                              cl_int3& vdims,
                                              cl_float3& voxelSizes,
                                              cl::NDRange& globalRange,
                                              std::shared_ptr<cl::NDRange> localRange,
                                              bool blocking)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    if(localRange != nullptr)
    {
        eargs = std::make_shared<cl::EnqueueArgs>(*Q[0], globalRange, *localRange);
    } else
    {
        eargs = std::make_shared<cl::EnqueueArgs>(*Q[0], globalRange);
    }
    auto lambda = [](cl_event e, cl_int status, void* data) {
        if(status != CL_COMPLETE)
        {
            LOGE << io::xprintf("Terminated with the status different than CL_COMPLETE");
        }
    };
    auto exe = (*FLOATvector_3DisotropicGradient)(*eargs, F, GX, GY, GZ, vdims, voxelSizes);
    exe.setCallback(CL_COMPLETE, lambda);
    if(blocking)
    {
        exe.wait();
    }
    return 0;
}

int Kniha::algFLOATvector_2DisotropicGradient(cl::Buffer& F,
                                              cl::Buffer& GX,
                                              cl::Buffer& GY,
                                              cl_int3& vdims,
                                              cl_float3& voxelSizes,
                                              cl::NDRange& globalRange,
                                              std::shared_ptr<cl::NDRange> localRange,
                                              bool blocking)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    if(localRange != nullptr)
    {
        eargs = std::make_shared<cl::EnqueueArgs>(*Q[0], globalRange, *localRange);
    } else
    {
        eargs = std::make_shared<cl::EnqueueArgs>(*Q[0], globalRange);
    }
    auto lambda = [](cl_event e, cl_int status, void* data) {
        if(status != CL_COMPLETE)
        {
            LOGE << io::xprintf("Terminated with the status different than CL_COMPLETE");
        }
    };
    auto exe = (*FLOATvector_2DisotropicGradient)(*eargs, F, GX, GY, vdims, voxelSizes);
    exe.setCallback(CL_COMPLETE, lambda);
    if(blocking)
    {
        exe.wait();
    }
    return 0;
}

int Kniha::algFLOATvector_isotropicBackDx(cl::Buffer& F,
                                          cl::Buffer& DX,
                                          cl_int3& vdims,
                                          cl_float3& voxelSizes,
                                          cl::NDRange& globalRange,
                                          std::shared_ptr<cl::NDRange> localRange,
                                          bool blocking)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    if(localRange != nullptr)
    {
        eargs = std::make_shared<cl::EnqueueArgs>(*Q[0], globalRange, *localRange);
    } else
    {
        eargs = std::make_shared<cl::EnqueueArgs>(*Q[0], globalRange);
    }
    auto lambda = [](cl_event e, cl_int status, void* data) {
        if(status != CL_COMPLETE)
        {
            LOGE << io::xprintf("Terminated with the status different than CL_COMPLETE");
        }
    };

    auto exe = (*FLOATvector_isotropicBackDx)(*eargs, F, DX, vdims, voxelSizes);
    exe.setCallback(CL_COMPLETE, lambda);
    if(blocking)
    {
        exe.wait();
    }
    return 0;
}

int Kniha::algFLOATvector_isotropicBackDy(cl::Buffer& F,
                                          cl::Buffer& DY,
                                          cl_int3& vdims,
                                          cl_float3& voxelSizes,
                                          cl::NDRange& globalRange,
                                          std::shared_ptr<cl::NDRange> localRange,
                                          bool blocking)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    if(localRange != nullptr)
    {
        eargs = std::make_shared<cl::EnqueueArgs>(*Q[0], globalRange, *localRange);
    } else
    {
        eargs = std::make_shared<cl::EnqueueArgs>(*Q[0], globalRange);
    }
    auto lambda = [](cl_event e, cl_int status, void* data) {
        if(status != CL_COMPLETE)
        {
            LOGE << io::xprintf("Terminated with the status different than CL_COMPLETE");
        }
    };

    auto exe = (*FLOATvector_isotropicBackDy)(*eargs, F, DY, vdims, voxelSizes);
    exe.setCallback(CL_COMPLETE, lambda);
    if(blocking)
    {
        exe.wait();
    }
    return 0;
}

int Kniha::algFLOATvector_isotropicBackDz(cl::Buffer& F,
                                          cl::Buffer& DZ,
                                          cl_int3& vdims,
                                          cl_float3& voxelSizes,
                                          cl::NDRange& globalRange,
                                          std::shared_ptr<cl::NDRange> localRange,
                                          bool blocking)
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    if(localRange != nullptr)
    {
        eargs = std::make_shared<cl::EnqueueArgs>(*Q[0], globalRange, *localRange);
    } else
    {
        eargs = std::make_shared<cl::EnqueueArgs>(*Q[0], globalRange);
    }
    auto lambda = [](cl_event e, cl_int status, void* data) {
        if(status != CL_COMPLETE)
        {
            LOGE << io::xprintf("Terminated with the status different than CL_COMPLETE");
        }
    };

    auto exe = (*FLOATvector_isotropicBackDz)(*eargs, F, DZ, vdims, voxelSizes);
    exe.setCallback(CL_COMPLETE, lambda);
    if(blocking)
    {
        exe.wait();
    }
    return 0;
}

int Kniha::algFLOAT_pbct_cutting_voxel_project(
    cl::Buffer& volume,
    cl::Buffer& projection,
    unsigned long& projectionOffset,
    cl_double8& CM,
    cl_int3& vdims,
    cl_double3& voxelSizes,
    cl_double3& volumeCenter,
    cl_int2& pdims,
    float globalScalingMultiplier,
    cl::NDRange& globalRange,
    std::shared_ptr<cl::NDRange> localRange, // default nullptr
    bool blocking) // default false
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    if(localRange != nullptr)
    {
        eargs = std::make_shared<cl::EnqueueArgs>(*Q[0], globalRange, *localRange);
    } else
    {
        eargs = std::make_shared<cl::EnqueueArgs>(*Q[0], globalRange);
    }
    auto lambda = [](cl_event e, cl_int status, void* data) {
        if(status != CL_COMPLETE)
        {
            LOGE << io::xprintf("Terminated with the status different than CL_COMPLETE");
        }
    };

    auto exe = (*FLOAT_pbct_cutting_voxel_project)(*eargs, volume, projection, projectionOffset, CM,
                                                   vdims, voxelSizes, volumeCenter, pdims,
                                                   globalScalingMultiplier);
    exe.setCallback(CL_COMPLETE, lambda);
    if(blocking)
    {
        exe.wait();
    }
    return 0;
}

int Kniha::algFLOAT_pbct_cutting_voxel_backproject(
    cl::Buffer& volume,
    cl::Buffer& projection,
    unsigned long& projectionOffset,
    cl_double8& CM,
    cl_int3& vdims,
    cl_double3& voxelSizes,
    cl_double3& volumeCenter,
    cl_int2& pdims,
    float globalScalingMultiplier,
    cl::NDRange& globalRange,
    std::shared_ptr<cl::NDRange> localRange, // default nullptr
    bool blocking) // default false
{
    std::shared_ptr<cl::EnqueueArgs> eargs;
    if(localRange != nullptr)
    {
        eargs = std::make_shared<cl::EnqueueArgs>(*Q[0], globalRange, *localRange);
    } else
    {
        eargs = std::make_shared<cl::EnqueueArgs>(*Q[0], globalRange);
    }
    auto lambda = [](cl_event e, cl_int status, void* data) {
        if(status != CL_COMPLETE)
        {
            LOGE << io::xprintf("Terminated with the status different than CL_COMPLETE");
        }
    };

    auto exe = (*FLOAT_pbct_cutting_voxel_backproject)(*eargs, volume, projection, projectionOffset,
                                                       CM, vdims, voxelSizes, volumeCenter, pdims,
                                                       globalScalingMultiplier);
    exe.setCallback(CL_COMPLETE, lambda);
    if(blocking)
    {
        exe.wait();
    }
    return 0;
}

} // namespace KCT
