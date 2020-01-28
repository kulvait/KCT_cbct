#pragma once

#include "PROG/Arguments.hpp"

namespace CTL::util {

class CArmArguments : public Arguments
{
public:
    // Dimensions
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
    bool useLegendrePolynomials = false;
    bool useChebyshevPolynomials = false;
    bool useFourierBasis = false;
    std::string engineerBasis = "";
    // Basis and timings
    uint32_t basisSize = 7;
    float pause_size = 1171;
    float frame_time = 16.8;
    float start_offset = 0.0, end_offset = 0.0;

    CLI::Option_group* og_geometry = nullptr;
    CLI::Option_group* og_basis = nullptr;

protected:
    CArmArguments(int argc, char* argv[], std::string appName);
    void addVolumeSizeArgs();
    void addVoxelSizeArgs();
    void addProjectionSizeArgs();
    void addPixelSizeArgs();
    void addBasisSpecificationArgs(bool includeBasisSize = true);
};
} // namespace CTL::util
