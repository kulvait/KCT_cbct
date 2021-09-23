#pragma once
#include <cctype>
#include <string>

#include "OPENCL/OpenCLManager.hpp"
#include "PROG/Arguments.hpp"
#include "rawop.h"

namespace KCT::util {

class CArmArguments : public virtual Arguments
{
public:
    // Dimensions
    CLI::Option_group* og_geometry = nullptr;
    uint32_t volumeSizeX = 256;
    uint32_t volumeSizeY = 256;
    uint32_t volumeSizeZ = 199;
    uint32_t projectionSizeX = 616;
    uint32_t projectionSizeY = 480;
    uint32_t projectionSizeZ = 248;
    // Discretization
    double voxelSizeX = 1.0;
    double voxelSizeY = 1.0;
    double voxelSizeZ = 1.0;
    double volumeCenterX = 0.0;
    double volumeCenterY = 0.0;
    double volumeCenterZ = 0.0;
    double pixelSizeX = 0.616;
    double pixelSizeY = 0.616;

    // Basis specification
    bool useLegendrePolynomials = false;
    bool useChebyshevPolynomials = false;
    bool useFourierBasis = false;
    std::string engineerBasis = "";
    // Basis and timings
    uint32_t basisSize = 7;
    float pause_size = 1171;
    float frame_time = 16.8;
    float start_offset = 0.0, end_offset = 0.0;

    // Settings
    uint32_t maxIterationCount = 40;
    double stoppingRelativeError = 0.00025;
    uint32_t reportKthIteration = 0;

    // Projector settings
    bool useCVPProjector = false;
    bool useExactScaling = true;
    bool useCosScaling = false;
    bool useNoScaling = false;
    bool useBarrierCalls = false;
    bool useElevationCorrection = false;
    uint32_t barrierArraySize = 7680; // To fit Intel
    bool useCenterVoxelProjector = false;
    bool useSidonProjector = false;
    uint32_t probesPerEdge = 1;
    bool useTTProjector = false;
    // OpenCL
    std::string CLplatformString = "";
    uint32_t CLplatformID = 0;
    std::vector<uint32_t> CLdeviceIDs;
    std::vector<uint32_t> backprojectorLocalNDRange = { 0, 0, 1 };
    std::vector<uint32_t> projectorLocalNDRange = { 0, 0, 1 };
    bool CLdebug = false;
    bool CLrelaxed = false;
    uint32_t CLitemsPerWorkgroup = 256;

    void parsePlatformString(bool verbose = false);

protected:
    CArmArguments(int argc, char* argv[], std::string appName);

    // From parsePlatformString()
    void insertDeviceID(uint32_t deviceID, uint32_t devicesOnPlatform);
    void fillDevicesList(std::string commaSeparatedEntries, uint32_t CLplatformID);

    void addGeometryGroup();
    void addVolumeSizeArgs();
    void addVoxelSizeArgs();
    void addVolumeCenterArgs();
    void addProjectionSizeArgs();
    void addPixelSizeArgs();

    void addBasisGroup();
    void addBasisSpecificationArgs(bool includeBasisSize = true);

    void addSettingsGroup();
    void addCLSettingsGroup();
    void addSettingsArgs();
    void addCLSettingsArgs();
    void addProjectorLocalNDRangeArgs();
    void addBackprojectorLocalNDRangeArgs();
    void addRelaxedArg();

    // Projector setup
    void addProjectorSettingsGroups();
    void addCuttingVoxelProjectorArgs(bool includeNoScaling = false);
    void addTTProjectorArgs();
    void addSidonProjectorArgs();
    void addCenterVoxelProjectorArgs();
    void addProjectorArgs();

    CLI::Option_group* og_basis = nullptr;
    CLI::Option_group* og_settings = nullptr;
    CLI::Option_group* og_projectorsettings = nullptr;
    CLI::Option_group* og_projectortypesettings = nullptr;
    CLI::Option_group* og_cl_settings = nullptr;

    CLI::Option* opt_cl_backprojectorlocalrange = nullptr;
    CLI::Option* opt_cl_projectorlocalrange = nullptr;
    CLI::Option* opt_cl_localarraysize = nullptr;
};
} // namespace KCT::util
