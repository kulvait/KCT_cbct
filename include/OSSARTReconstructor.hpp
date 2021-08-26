#pragma once

// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <CL/cl.hpp>
#include <chrono>
#include <ctime>
#include <iostream>

// Internal libraries
#include "BaseReconstructor.hpp"
#include "DEN/DenProjectionMatrixReader.hpp"
#include "MATRIX/LightProjectionMatrix.hpp"
#include "MATRIX/ProjectionMatrix.hpp"
#include "OPENCL/OpenCLManager.hpp"
#include "rawop.h"
#include "stringFormatter.h"

namespace KCT {

class OSSARTReconstructor : public BaseReconstructor
{
public:
    /**
     * Initialize Cutting Voxel Projector
     *
     * @param volume Pointer to volume file
     * @param vdimx Volume x dimension
     * @param vdimy Volume y dimension
     * @param vdimz Volume z dimension
     * @param xpath Path of cl kernel files
     * @param debug Should debugging be used by suppliing source and -g as options
     */
    OSSARTReconstructor(uint32_t pdimx,
                        uint32_t pdimy,
                        uint32_t pdimz,
                        uint32_t vdimx,
                        uint32_t vdimy,
                        uint32_t vdimz,
                        uint32_t workGroupSize = 256)
        : BaseReconstructor(pdimx, pdimy, pdimz, vdimx, vdimy, vdimz, workGroupSize)
    {
        setup(0.9, 1);
        removeUpperBoxCondition();
        removeLowerBoxCondition();
    }

    void setup(float relaxationParameter, uint32_t subsetCount);

    void addUpperBoxCondition(float upperBound, float upperBoxSubstitution);

    void removeUpperBoxCondition();

    void addLowerBoxCondition(float lowerBound, float lowerBoxSubstitution);

    void removeLowerBoxCondition();

    virtual int reconstruct(uint32_t maxIterations = 100, float errCondition = 0.01);

private:
    float relaxationParameter;
    uint32_t subsetCount;
    bool lowerBoxCondition, upperBoxCondition;
    float lowerBound, upperBound;
    float lowerBoundSubstitution, upperBoundSubstitution;
};

} // namespace KCT
