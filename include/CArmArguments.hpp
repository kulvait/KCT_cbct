#pragma once
#include <cctype>
#include <string>

#include "OPENCL/OpenCLManager.hpp"
#include "PROG/Arguments.hpp"
#include "rawop.h"

namespace CTL::util {

class CArmArguments : public Arguments
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
    double pixelSizeX = 0.616;
    double pixelSizeY = 0.616;

    // Basis specification
    CLI::Option_group* og_basis = nullptr;
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
    CLI::Option_group* og_settings = nullptr;
    uint32_t maxIterationCount = 40;
    double stoppingRelativeError = 0.00025;
    uint32_t reportKthIteration = 0;

    // Projector settings
    bool useSidonProjector = false;
    uint32_t probesPerEdge = 1;
    bool useTTProjector = false;
    bool useCVPProjector = true;
    bool useExactScaling = true;
    // OpenCL
    CLI::Option_group* og_cl_settings = nullptr;
    std::string CLplatformString = "";
    uint32_t CLplatformID = 0;
    std::vector<uint32_t> CLdeviceIDs;
    bool CLdebug = false;
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
    void addProjectionSizeArgs();
    void addPixelSizeArgs();

    void addBasisGroup();
    void addBasisSpecificationArgs(bool includeBasisSize = true);

    void addSettingsGroup();
    void addCLSettingsGroup();
    void addSettingsArgs();
    void addProjectorArgs();
};
} // namespace CTL::util
