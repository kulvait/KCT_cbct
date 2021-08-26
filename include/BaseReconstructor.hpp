#pragma once

// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <CL/cl.hpp>
#include <chrono>
#include <ctime>
#include <iostream>

// Internal libraries
#include "AlgorithmsBarrierBuffers.hpp"
#include "DEN/DenProjectionMatrixReader.hpp"
#include "Kniha.hpp"
#include "MATRIX/LightProjectionMatrix.hpp"
#include "MATRIX/ProjectionMatrix.hpp"
#include "MATRIX/utils.hpp"
#include "OPENCL/OpenCLManager.hpp"
#include "rawop.h"
#include "stringFormatter.h"

using namespace KCT::matrix;
namespace KCT {

class BaseReconstructor : public virtual Kniha, public AlgorithmsBarrierBuffers
{
public:
    cl::NDRange guessProjectionLocalNDRange(bool barrierCalls)
    {
        cl::NDRange projectorLocalNDRange;
        if(barrierCalls)
        {

            if(vdimx % 64 == 0 && vdimy % 4 == 0 && workGroupSize >= 256)
            {
                projectorLocalNDRange = cl::NDRange(64, 4, 1); // 9.45 Barrier
            } else
            {
                projectorLocalNDRange = cl::NDRange();
            }
        } else
        {
            if(vdimz % 4 == 0 && vdimy % 64 == 0 && workGroupSize >= 256)
            {
                projectorLocalNDRange = cl::NDRange(4, 64, 1); // 23.23 RELAXED
            } else
            {
                projectorLocalNDRange = cl::NDRange();
            }
            /*
                        // ZYX
                        localRangeProjection = cl::NDRange(4, 64, 1); // 23.23 RELAXED
                        localRangeProjection = cl::NDRange(); // 27.58 RELAXED
                        localRangeProjection = cl::NDRange(256, 1, 1); // 31.4 RELAXED
                        localRangeProjection = cl::NDRange(1, 256, 1); // 42.27 RELAXED
                        localRangeProjection = cl::NDRange(1, 1, 256); // 42.52 RELAXED
                        localRangeProjection = cl::NDRange(128, 2, 1); // 27.52 RELAXED
                        localRangeProjection = cl::NDRange(128, 1, 2); // 38.1 RELAXED
                        localRangeProjection = cl::NDRange(2, 128, 1); // 24.53 RELAXED
                        localRangeProjection = cl::NDRange(1, 128, 2); // 30.5 RELAXED
                        localRangeProjection = cl::NDRange(1, 2, 128); // 36.17 RELAXED
                        localRangeProjection = cl::NDRange(2, 1, 128); // 26.57 RELAXED
                        localRangeProjection = cl::NDRange(64, 4, 1); // 33.59 RELAXED
                        localRangeProjection = cl::NDRange(64, 1, 4); // 30.79 RELAXED
                        localRangeProjection = cl::NDRange(4, 64, 1); // 23.29 RELAXED
                        localRangeProjection = cl::NDRange(1, 64, 4); // 31.22 RELAXED
                        localRangeProjection = cl::NDRange(4, 1, 64); // 25.24 RELAXED
                        localRangeProjection = cl::NDRange(1, 4, 64); // 43.15 RELAXED
                        localRangeProjection = cl::NDRange(64, 2, 2); // 42.27 RELAXED
                        localRangeProjection = cl::NDRange(2, 64, 2); // 26.31 RELAXED
                        localRangeProjection = cl::NDRange(2, 2, 64); // 30.83 RELAXED
                        localRangeProjection = cl::NDRange(1, 16, 16); // 42.28 RELAXED
                        localRangeProjection = cl::NDRange(32, 4, 2); // 43.86 RELAXED
                        localRangeProjection = cl::NDRange(32, 2, 4); // 31.32 RELAXED
                        localRangeProjection = cl::NDRange(4, 32, 2); // 24.13 RELAXED
                        localRangeProjection = cl::NDRange(2, 32, 4); // 25.48 RELAXED
                        localRangeProjection = cl::NDRange(2, 4, 32); // 34.53 RELAXED
                        localRangeProjection = cl::NDRange(4, 2, 32); // 27.54 RELAXED
                        localRangeProjection = cl::NDRange(32, 8, 1); // 36.83 RELAXED
                        localRangeProjection = cl::NDRange(32, 1, 8); // 26.3 RELAXED
                        localRangeProjection = cl::NDRange(8, 32, 1); // 25.82 RELAXED
                        localRangeProjection = cl::NDRange(1, 32, 8); // 36.59 RELAXED
                        localRangeProjection = cl::NDRange(1, 8, 32); // 47.2 RELAXED
                        localRangeProjection = cl::NDRange(8, 1, 32); // 25.1 RELAXED
                        localRangeProjection = cl::NDRange(16, 16, 1); // 31.18 RELAXED
                        localRangeProjection = cl::NDRange(16, 1, 16); // 26.12 RELAXED
                        localRangeProjection = cl::NDRange(1, 16, 16); // 42.28 RELAXED
                        localRangeProjection = cl::NDRange(16, 8, 2); // 40.83 RELAXED
                        localRangeProjection = cl::NDRange(16, 2, 8); // 27 RELAXED
                        localRangeProjection = cl::NDRange(8, 16, 2); // 29.07 RELAXED
                        localRangeProjection = cl::NDRange(2, 16, 8); // 29.98 RELAXED
                        localRangeProjection = cl::NDRange(8, 2, 16); // 26.1 RELAXED
                        localRangeProjection = cl::NDRange(2, 8, 16); // 33.99 RELAXED
                        localRangeProjection = cl::NDRange(16, 4, 4); // 30.98 RELAXED
                        localRangeProjection = cl::NDRange(4, 16, 4); // 25.22 RELAXED
                        localRangeProjection = cl::NDRange(4, 4, 16); // 28.43 RELAXED
                        localRangeProjection = cl::NDRange(8, 8, 4); // 26.01 RELAXED
                        localRangeProjection = cl::NDRange(8, 4, 8); // 25.94 RELAXED
                        localRangeProjection = cl::NDRange(8, 8, 4); // 25.99 RELAXED
                        localRangeProjection = cl::NDRange(128, 1, 1); // 31.41 RELAXED
                        localRangeProjection = cl::NDRange(1, 128, 1); // 29.77 RELAXED
                        localRangeProjection = cl::NDRange(1, 1, 128); // 32.65 RELAXED
                        localRangeProjection = cl::NDRange(64, 2, 1); // 36.02 RELAXED
                        localRangeProjection = cl::NDRange(64, 1, 2); // 48.74 RELAXED
                        localRangeProjection = cl::NDRange(1, 64, 2); // 31.02 RELAXED
                        localRangeProjection = cl::NDRange(2, 64, 1); // 25.5 RELAXED
                        localRangeProjection = cl::NDRange(2, 1, 64); // 28.37 RELAXED
                        localRangeProjection = cl::NDRange(1, 2, 64); // 37.47 RELAXED
                        localRangeProjection = cl::NDRange(32, 4, 1); // 70.6 RELAXED
                        localRangeProjection = cl::NDRange(32, 1, 4); // 36.1 RELAXED
                        localRangeProjection = cl::NDRange(1, 32, 4); // 33.84 RELAXED
                        localRangeProjection = cl::NDRange(4, 32, 1); // 27.14 RELAXED
                        localRangeProjection = cl::NDRange(4, 1, 32); // 26.39 RELAXED
                        localRangeProjection = cl::NDRange(1, 4, 32); // 42.66 RELAXED
                        localRangeProjection = cl::NDRange(32, 2, 2); // 52.77 RELAXED
                        localRangeProjection = cl::NDRange(2, 32, 2); // 25.63 RELAXED
                        localRangeProjection = cl::NDRange(2, 2, 32); // 31.02 RELAXED
                        localRangeProjection = cl::NDRange(16, 8, 1); // 62.91 RELAXED
                        localRangeProjection = cl::NDRange(16, 1, 8); // 30.92 RELAXED
                        localRangeProjection = cl::NDRange(8, 16, 1); // 35.56 RELAXED
                        localRangeProjection = cl::NDRange(1, 16, 8); // 41.01 RELAXED
                        localRangeProjection = cl::NDRange(1, 8, 16); // 44.49 RELAXED
                        localRangeProjection = cl::NDRange(8, 1, 16); // 27.51 RELAXED
                        localRangeProjection = cl::NDRange(16, 4, 2); // 52.41 RELAXED
                        localRangeProjection = cl::NDRange(16, 2, 4); // 35.46 RELAXED
                        localRangeProjection = cl::NDRange(2, 16, 4); // 28.74 RELAXED
                        localRangeProjection = cl::NDRange(4, 16, 2); // 38.59 RELAXED
                        localRangeProjection = cl::NDRange(2, 4, 16); // 33.57 RELAXED
                        localRangeProjection = cl::NDRange(4, 2, 16); // 28.13 RELAXED
                        localRangeProjection = cl::NDRange(8, 8, 2); // 42.77 RELAXED
                        localRangeProjection = cl::NDRange(8, 2, 8); // 29.21 RELAXED
                        localRangeProjection = cl::NDRange(2, 8, 8); // 32.44 RELAXED
                        localRangeProjection = cl::NDRange(8, 4, 4); // 33.73 RELAXED
                        localRangeProjection = cl::NDRange(4, 8, 4); // 28.29 RELAXED
                        localRangeProjection = cl::NDRange(4, 4, 8); // 28.96 RELAXED
                        localRangeProjection = cl::NDRange(64, 1, 1); // 73.33 RELAXED
                        localRangeProjection = cl::NDRange(1, 64, 1); // 30.72 RELAXED
                        localRangeProjection = cl::NDRange(1, 1, 64); // 34.66 RELAXED
                        localRangeProjection = cl::NDRange(32, 2, 1); // 83.75 RELAXED
                        localRangeProjection = cl::NDRange(32, 1, 2); // 60.47 RELAXED
                        localRangeProjection = cl::NDRange(1, 32, 2); // 33.54 RELAXED
                        localRangeProjection = cl::NDRange(2, 32, 1); // 28.32 RELAXED
                        localRangeProjection = cl::NDRange(1, 2, 32); // 37.56 RELAXED
                        localRangeProjection = cl::NDRange(2, 1, 32); // 29.73 RELAXED
                        localRangeProjection = cl::NDRange(16, 4, 1); // 88.21 RELAXED
                        localRangeProjection = cl::NDRange(16, 1, 4); // 43.81 RELAXED
                        localRangeProjection = cl::NDRange(4, 16, 1); // 45.73 RELAXED
                        localRangeProjection = cl::NDRange(1, 16, 4); // 38.55 RELAXED
                        localRangeProjection = cl::NDRange(4, 1, 16); // 30.69 RELAXED
                        localRangeProjection = cl::NDRange(1, 4, 16); // 41.65 RELAXED
                        localRangeProjection = cl::NDRange(16, 2, 2); // 61.03 RELAXED
                        localRangeProjection = cl::NDRange(2, 16, 2); // 40.26 RELAXED
                        localRangeProjection = cl::NDRange(2, 2, 16); // 32.01 RELAXED
                        localRangeProjection = cl::NDRange(8, 8, 1); // 84.33 RELAXED
                        localRangeProjection = cl::NDRange(8, 1, 8); // 35.4 RELAXED
                        localRangeProjection = cl::NDRange(1, 8, 8); // 42.66 RELAXED
                        localRangeProjection = cl::NDRange(8, 4, 2); // 57.54 RELAXED
                        localRangeProjection = cl::NDRange(8, 2, 4); // 40.99 RELAXED
                        localRangeProjection = cl::NDRange(4, 8, 2); // 41.55 RELAXED
                        localRangeProjection = cl::NDRange(2, 8, 4); // 34.14 RELAXED
                        localRangeProjection = cl::NDRange(4, 2, 8); // 33.25 RELAXED
                        localRangeProjection = cl::NDRange(2, 4, 8); // 33.94 RELAXED
                        localRangeProjection = cl::NDRange(4, 4, 4); // 37.98 RELAXED
                        localRangeProjection = cl::NDRange(32, 1, 1); // 115.24 RELAXED
                        localRangeProjection = cl::NDRange(1, 32, 1); // 33.92 RELAXED
                        localRangeProjection = cl::NDRange(1, 1, 32); // 39.21 RELAXED
                        localRangeProjection = cl::NDRange(16, 2, 1); // 114.62 RELAXED
                        localRangeProjection = cl::NDRange(16, 1, 2); // 77.93 RELAXED
                        localRangeProjection = cl::NDRange(2, 16, 1); // 75.46 RELAXED
                        localRangeProjection = cl::NDRange(1, 16, 2); // 45.32 RELAXED
                        localRangeProjection = cl::NDRange(2, 1, 16); // 43.31 RELAXED
                        localRangeProjection = cl::NDRange(1, 2, 16); // 41.63 RELAXED
                        localRangeProjection = cl::NDRange(8, 4, 1); // 110.89 RELAXED
                        localRangeProjection = cl::NDRange(8, 1, 4); // 62.35 RELAXED
                        localRangeProjection = cl::NDRange(4, 8, 1); // 81.61 RELAXED
                        localRangeProjection = cl::NDRange(1, 8, 4); // 42.95 RELAXED
                        localRangeProjection = cl::NDRange(4, 1, 8); // 51.45 RELAXED
                        localRangeProjection = cl::NDRange(1, 4, 8); // 43.01 RELAXED
                        localRangeProjection = cl::NDRange(8, 2, 2); // 73.64 RELAXED
                        localRangeProjection = cl::NDRange(2, 8, 2); // 51.65 RELAXED
                        localRangeProjection = cl::NDRange(2, 2, 8); // 45.86 RELAXED
                        localRangeProjection = cl::NDRange(4, 4, 2); // 68.42 RELAXED
                        localRangeProjection = cl::NDRange(4, 2, 4); // 56.71 RELAXED
                        localRangeProjection = cl::NDRange(2, 4, 4); // 50.84 RELAXED
                        localRangeProjection = cl::NDRange(16, 1, 1); // 143.31 RELAXED
                        localRangeProjection = cl::NDRange(1, 16, 1); // 82.41 RELAXED
                        localRangeProjection = cl::NDRange(1, 1, 16); // 63.43 RELAXED
                        localRangeProjection = cl::NDRange(8, 2, 1); // 137.76 RELAXED
                        localRangeProjection = cl::NDRange(8, 1, 2); // 114.3 RELAXED
                        localRangeProjection = cl::NDRange(2, 8, 1); // 90.78 RELAXED
                        localRangeProjection = cl::NDRange(1, 8, 2); // 71.05 RELAXED
                        localRangeProjection = cl::NDRange(2, 1, 8); // 77.34 RELAXED
                        localRangeProjection = cl::NDRange(1, 2, 8); // 67.66 RELAXED
                        localRangeProjection = cl::NDRange(4, 4, 1); // 132.09 RELAXED
                        localRangeProjection = cl::NDRange(4, 1, 4); // 93.69 RELAXED
                        localRangeProjection = cl::NDRange(1, 4, 4); // 71.06 RELAXED
                        localRangeProjection = cl::NDRange(4, 2, 2); // 105.82 RELAXED
                        localRangeProjection = cl::NDRange(2, 4, 2); // 94.46 RELAXED
                        localRangeProjection = cl::NDRange(2, 2, 4); // 82.65 RELAXED
                        localRangeProjection = cl::NDRange(8, 1, 1); // 214.05 RELAXED
                        localRangeProjection = cl::NDRange(1, 8, 1); // 127.61 RELAXED
                        localRangeProjection = cl::NDRange(1, 1, 8); // 115.48 RELAXED
                        localRangeProjection = cl::NDRange(4, 2, 1); // 196.41 RELAXED
                        localRangeProjection = cl::NDRange(4, 1, 2); // 171.96 RELAXED
                        localRangeProjection = cl::NDRange(2, 4, 1); // 171.29 RELAXED
                        localRangeProjection = cl::NDRange(1, 4, 2); // 126.08 RELAXED
                        localRangeProjection = cl::NDRange(2, 1, 4); // 142.78 RELAXED
                        localRangeProjection = cl::NDRange(1, 2, 4); // 121.82 RELAXED
                        localRangeProjection = cl::NDRange(2, 2, 2); // 152.92 RELAXED
                        localRangeProjection = cl::NDRange(4, 1, 1); // 313.17 RELAXED
                        localRangeProjection = cl::NDRange(1, 4, 1); // 224.94 RELAXED
                        localRangeProjection = cl::NDRange(1, 1, 4); // 211.77 RELAXED
                        localRangeProjection = cl::NDRange(2, 2, 1); // 272.84 RELAXED
                        localRangeProjection = cl::NDRange(2, 1, 2); // 260.53 RELAXED
                        localRangeProjection = cl::NDRange(1, 2, 2); // 219.3 RELAXED
                        localRangeProjection = cl::NDRange(2, 1, 1); // 476.79 RELAXED
                        localRangeProjection = cl::NDRange(1, 2, 1); // 385.66 RELAXED
                        localRangeProjection = cl::NDRange(1, 1, 2); // 381.26 RELAXED
                        localRangeProjection = cl::NDRange(1, 1, 1); // 697.49 RELAXED
            */
        }
        return projectorLocalNDRange;
    }

    cl::NDRange guessBackprojectorLocalNDRange()
    {
        cl::NDRange backprojectorLocalNDRange;
        if(vdimx % 4 == 0 && vdimy % 16 == 0 && workGroupSize >= 64)
        {
            backprojectorLocalNDRange = cl::NDRange(4, 16, 1); // 4.05 RELAXED
        } else
        {
            backprojectorLocalNDRange = cl::NDRange();
        }
        return backprojectorLocalNDRange;
        /*
                    // ZYX
                    localRangeBackprojection = cl::NDRange(1, 8, 8);
                    // Natural XYZ backprojection ordering
                    localRangeBackprojection = cl::NDRange(); // 5.55 RELAXED
                    localRangeBackprojection = cl::NDRange(256, 1, 1); // 5.55 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 256, 1); // 5.51 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 1, 256); // 5.55 RELAXED
                    localRangeBackprojection = cl::NDRange(128, 2, 1); // 5.16 RELAXED
                    localRangeBackprojection = cl::NDRange(128, 1, 2); // 5.18 RELAXED
                    localRangeBackprojection = cl::NDRange(2, 128, 1); // 5.40 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 128, 2); // 7.18 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 2, 128); // 12.88 RELAXED
                    localRangeBackprojection = cl::NDRange(2, 1, 128); // 13.99 RELAXED
                    localRangeBackprojection = cl::NDRange(64, 4, 1); // 4.62 RELAXED
                    localRangeBackprojection = cl::NDRange(64, 1, 4); // 4.78 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 64, 4); // 6.83 RELAXED
                    localRangeBackprojection = cl::NDRange(4, 64, 1); // 4.29 RELAXED
                    localRangeBackprojection = cl::NDRange(4, 1, 64); // 7.73 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 4, 64); // 8.71 RELAXED
                    localRangeBackprojection = cl::NDRange(64, 2, 2); // 4.69 RELAXED
                    localRangeBackprojection = cl::NDRange(2, 64, 2); // 5.02 RELAXED
                    localRangeBackprojection = cl::NDRange(2, 2, 64); // 7.54 RELAXED
                    localRangeBackprojection = cl::NDRange(8, 8, 4); // 4.30 RELAXED
                    localRangeBackprojection = cl::NDRange(16, 16, 1); // 4.28 RELAXED
                    localRangeBackprojection = cl::NDRange(32, 4, 2); // 4.46 RELAXED
                    localRangeBackprojection = cl::NDRange(32, 2, 4); // 4.50 RELAXED
                    localRangeBackprojection = cl::NDRange(4, 32, 2); // 4.26 RELAXED
                    localRangeBackprojection = cl::NDRange(2, 32, 4); // 5.03 RELAXED
                    localRangeBackprojection = cl::NDRange(2, 4, 32); // 6.28 RELAXED
                    localRangeBackprojection = cl::NDRange(4, 2, 32); // 5.83 RELAXED
                    localRangeBackprojection = cl::NDRange(32, 8, 1); // 4.39 RELAXED
                    localRangeBackprojection = cl::NDRange(32, 1, 8); // 4.81 RELAXED
                    localRangeBackprojection = cl::NDRange(8, 32, 1); // 4.17 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 32, 8); // 6.78 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 8, 32); // 7.46 RELAXED
                    localRangeBackprojection = cl::NDRange(8, 1, 32); // 6.15 RELAXED
                    localRangeBackprojection = cl::NDRange(16, 16, 1); // 4.28 RELAXED
                    localRangeBackprojection = cl::NDRange(16, 1, 16); // 5.28 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 16, 16); // 6.92 RELAXED
                    localRangeBackprojection = cl::NDRange(16, 8, 2); // 4.31 RELAXED
                    localRangeBackprojection = cl::NDRange(16, 2, 8); // 4.65 RELAXED
                    localRangeBackprojection = cl::NDRange(8, 16, 2); // 4.13 RELAXED
                    localRangeBackprojection = cl::NDRange(2, 16, 8); // 5.08 RELAXED
                    localRangeBackprojection = cl::NDRange(8, 2, 16); // 4.93 RELAXED
                    localRangeBackprojection = cl::NDRange(2, 8, 16); // 5.49 RELAXED
                    localRangeBackprojection = cl::NDRange(16, 4, 4); // 4.39 RELAXED
                    localRangeBackprojection = cl::NDRange(4, 16, 4); // 4.29 RELAXED
                    localRangeBackprojection = cl::NDRange(4, 4, 16); // 4.92 RELAXED
                    localRangeBackprojection = cl::NDRange(8, 8, 4); // 4.2 RELAXED
                    localRangeBackprojection = cl::NDRange(8, 4, 8); // 4.39 RELAXED
                    localRangeBackprojection = cl::NDRange(8, 8, 4); // 4.23 RELAXED
                    localRangeBackprojection = cl::NDRange(128, 1, 1); // 5.16 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 128, 1); // 7.12 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 1, 128); // 15.73 RELAXED
                    localRangeBackprojection = cl::NDRange(64, 2, 1); // 4.72 RELAXED
                    localRangeBackprojection = cl::NDRange(64, 1, 2); // 4.69 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 64, 2); // 5.97 RELAXED
                    localRangeBackprojection = cl::NDRange(2, 64, 1); // 4.48 RELAXED
                    localRangeBackprojection = cl::NDRange(2, 1, 64); // 8.29 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 2, 64); // 8.08 RELAXED
                    localRangeBackprojection = cl::NDRange(32, 4, 1); // 4.46 RELAXED
                    localRangeBackprojection = cl::NDRange(32, 1, 4); // 4.60 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 32, 4); // 5.78 RELAXED
                    localRangeBackprojection = cl::NDRange(4, 32, 1); // 4.11 RELAXED
                    localRangeBackprojection = cl::NDRange(4, 1, 32); // 5.80 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 4, 32); // 6.38 RELAXED
                    localRangeBackprojection = cl::NDRange(32, 2, 2); // 4.49 RELAXED
                    localRangeBackprojection = cl::NDRange(2, 32, 2); // 4.42 RELAXED
                    localRangeBackprojection = cl::NDRange(2, 2, 32); // 5.64 RELAXED
                    localRangeBackprojection = cl::NDRange(16, 8, 1); // 4.26 RELAXED
                    localRangeBackprojection = cl::NDRange(16, 1, 8); // 4.72 RELAXED
                    localRangeBackprojection = cl::NDRange(8, 16, 1); // 4.08 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 16, 8); // 5.80 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 8, 16); // 5.93 RELAXED
                    localRangeBackprojection = cl::NDRange(8, 1, 16); // 4.85 RELAXED
                    localRangeBackprojection = cl::NDRange(16, 4, 2); // 4.31 RELAXED
                    localRangeBackprojection = cl::NDRange(16, 2, 4); // 4.39 RELAXED
                    localRangeBackprojection = cl::NDRange(2, 16, 4); // 4.37 RELAXED
                    localRangeBackprojection = cl::NDRange(4, 16, 2); // 4.08 RELAXED
                    localRangeBackprojection = cl::NDRange(2, 4, 16); // 4.79 RELAXED
                    localRangeBackprojection = cl::NDRange(4, 2, 16); // 4.58 RELAXED
                    localRangeBackprojection = cl::NDRange(8, 8, 2); // 4.08 RELAXED
                    localRangeBackprojection = cl::NDRange(8, 2, 8); // 4.35 RELAXED
                    localRangeBackprojection = cl::NDRange(2, 8, 8); // 4.55 RELAXED
                    localRangeBackprojection = cl::NDRange(8, 4, 4); // 4.20 RELAXED
                    localRangeBackprojection = cl::NDRange(4, 8, 4); // 4.16 RELAXED
                    localRangeBackprojection = cl::NDRange(4, 4, 8); // 4.27 RELAXED
                    localRangeBackprojection = cl::NDRange(64, 1, 1); // 4.77 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 64, 1); // 5.93 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 1, 64); // 9.23 RELAXED
                    localRangeBackprojection = cl::NDRange(32, 2, 1); // 4.49 RELAXED
                    localRangeBackprojection = cl::NDRange(32, 1, 2); // 4.59 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 32, 2); // 5.20 RELAXED
                    localRangeBackprojection = cl::NDRange(2, 32, 1); // 4.34 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 2, 32); // 6.28 RELAXED
                    localRangeBackprojection = cl::NDRange(2, 1, 32); // 6.35 RELAXED
                    localRangeBackprojection = cl::NDRange(16, 4, 1); // 4.31 RELAXED
                    localRangeBackprojection = cl::NDRange(16, 1, 4); // 4.65 RELAXED
                    localRangeBackprojection = cl::NDRange(4, 16, 1); // 4.05 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 16, 4); // 5.13 RELAXED
                    localRangeBackprojection = cl::NDRange(4, 1, 16); // 4.97 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 4, 16); // 5.39 RELAXED
                    localRangeBackprojection = cl::NDRange(16, 2, 2); // 4.42 RELAXED
                    localRangeBackprojection = cl::NDRange(2, 16, 2); // 4.30 RELAXED
                    localRangeBackprojection = cl::NDRange(2, 2, 16); // 4.82 RELAXED
                    localRangeBackprojection = cl::NDRange(8, 8, 1); // 4.06 RELAXED
                    localRangeBackprojection = cl::NDRange(8, 1, 8); // 4.59 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 8, 8); // 5.13 RELAXED
                    localRangeBackprojection = cl::NDRange(8, 4, 2); // 4.17 RELAXED
                    localRangeBackprojection = cl::NDRange(8, 2, 4); // 4.34 RELAXED
                    localRangeBackprojection = cl::NDRange(4, 8, 2); // 4.09 RELAXED
                    localRangeBackprojection = cl::NDRange(2, 8, 4); // 4.37 RELAXED
                    localRangeBackprojection = cl::NDRange(4, 2, 8); // 4.36 RELAXED
                    localRangeBackprojection = cl::NDRange(2, 4, 8); // 4.42 RELAXED
                    localRangeBackprojection = cl::NDRange(4, 4, 4); // 4.18 RELAXED
                    localRangeBackprojection = cl::NDRange(32, 1, 1); // 8.62 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 32, 1); // 8.30 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 1, 32); // 11.38 RELAXED
                    localRangeBackprojection = cl::NDRange(16, 2, 1); // 8.33 RELAXED
                    localRangeBackprojection = cl::NDRange(16, 1, 2); // 8.49 RELAXED
                    localRangeBackprojection = cl::NDRange(2, 16, 1); // 7.87 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 16, 2); // 8.21 RELAXED
                    localRangeBackprojection = cl::NDRange(2, 1, 16); // 8.94 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 2, 16); // 8.77 RELAXED
                    localRangeBackprojection = cl::NDRange(8, 4, 1); // 7.88 RELAXED
                    localRangeBackprojection = cl::NDRange(8, 1, 4); // 8.21 RELAXED
                    localRangeBackprojection = cl::NDRange(4, 8, 1); // 7.75 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 8, 4); // 8.16 RELAXED
                    localRangeBackprojection = cl::NDRange(4, 1, 8); // 8.31 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 4, 8); // 8.29 RELAXED
                    localRangeBackprojection = cl::NDRange(8, 2, 2); // 8.03 RELAXED
                    localRangeBackprojection = cl::NDRange(2, 8, 2); // 7.9 RELAXED
                    localRangeBackprojection = cl::NDRange(2, 2, 8); // 8.14 RELAXED
                    localRangeBackprojection = cl::NDRange(4, 4, 2); // 7.82 RELAXED
                    localRangeBackprojection = cl::NDRange(4, 2, 4); // 7.90 RELAXED
                    localRangeBackprojection = cl::NDRange(2, 4, 4); // 7.90 RELAXED
                    localRangeBackprojection = cl::NDRange(16, 1, 1); // 16.07 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 16, 1); // 15.14 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 1, 16); // 16.06 RELAXED
                    localRangeBackprojection = cl::NDRange(8, 2, 1); // 15.27 RELAXED
                    localRangeBackprojection = cl::NDRange(8, 1, 2); // 15.35 RELAXED
                    localRangeBackprojection = cl::NDRange(2, 8, 1); // 14.96 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 8, 2); // 14.96 RELAXED
                    localRangeBackprojection = cl::NDRange(2, 1, 8); // 15.04 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 2, 8); // 14.84 RELAXED
                    localRangeBackprojection = cl::NDRange(4, 4, 1); // 14.89 RELAXED
                    localRangeBackprojection = cl::NDRange(4, 1, 4); // 15 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 4, 4); // 14.83 RELAXED
                    localRangeBackprojection = cl::NDRange(4, 2, 2); // 14.89 RELAXED
                    localRangeBackprojection = cl::NDRange(2, 4, 2); // 14.83 RELAXED
                    localRangeBackprojection = cl::NDRange(2, 2, 4); // 14.80 RELAXED
                    localRangeBackprojection = cl::NDRange(8, 1, 1); // 29.62 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 8, 1); // 28.72 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 1, 8); // 27.29 RELAXED
                    localRangeBackprojection = cl::NDRange(4, 2, 1); // 28.79 RELAXED
                    localRangeBackprojection = cl::NDRange(4, 1, 2); // 28.49 RELAXED
                    localRangeBackprojection = cl::NDRange(2, 4, 1); // 28.60 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 4, 2); // 28.13 RELAXED
                    localRangeBackprojection = cl::NDRange(2, 1, 4); // 27.58 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 2, 4); // 27.35 RELAXED
                    localRangeBackprojection = cl::NDRange(2, 2, 2); // 28.18 RELAXED
                    localRangeBackprojection = cl::NDRange(4, 1, 1); // 55.16 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 4, 1); // 54.45 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 1, 4); // 50.40 RELAXED
                    localRangeBackprojection = cl::NDRange(2, 2, 1); // 54.57 RELAXED
                    localRangeBackprojection = cl::NDRange(2, 1, 2); // 52.74 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 2, 2); // 52.48 RELAXED
                    localRangeBackprojection = cl::NDRange(2, 1, 1); // 102.21 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 2, 1); // 101.91 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 1, 2); // 96.86 RELAXED
                    localRangeBackprojection = cl::NDRange(1, 1, 1); // 187.64 RELAXED
        */
    }

    BaseReconstructor(uint32_t pdimx,
                      uint32_t pdimy,
                      uint32_t pdimz,
                      uint32_t vdimx,
                      uint32_t vdimy,
                      uint32_t vdimz,
                      uint32_t workGroupSize = 256,
                      cl::NDRange projectorLocalNDRange = cl::NDRange(),
                      cl::NDRange backprojectorLocalNDRange = cl::NDRange())
        : AlgorithmsBarrierBuffers(pdimx, pdimy, pdimz, vdimx, vdimy, vdimz, workGroupSize)
    {
        pdims = cl_int2({ int(pdimx), int(pdimy) });
        pdims_uint = cl_uint2({ pdimx, pdimy });
        vdims = cl_int3({ int(vdimx), int(vdimy), int(vdimz) });
        timestamp = std::chrono::steady_clock::now();
        std::size_t projectorLocalNDRangeDim = projectorLocalNDRange.dimensions();
        std::size_t backprojectorLocalNDRangeDim = backprojectorLocalNDRange.dimensions();
        if(projectorLocalNDRangeDim == 3)
        {
            if(projectorLocalNDRange[0] == 0 && projectorLocalNDRange[1] == 0
               && projectorLocalNDRange[2] == 0)
            {
                this->projectorLocalNDRange = cl::NDRange();
                this->projectorLocalNDRangeBarrier = cl::NDRange();
            } else if(projectorLocalNDRange[0] == 0 || projectorLocalNDRange[1] == 0
                      || projectorLocalNDRange[2] == 0)
            {
                this->projectorLocalNDRange = guessProjectionLocalNDRange(false);
                this->projectorLocalNDRangeBarrier = guessProjectionLocalNDRange(true);
            } else
            {
                this->projectorLocalNDRange = projectorLocalNDRange;
                this->projectorLocalNDRangeBarrier = projectorLocalNDRange;
            }
        } else
        {
            if(projectorLocalNDRangeDim != 0)
            {
                LOGE << io::xprintf(
                    "Wrong specification of projectorLocalNDRange, trying guessing!");
            }
            this->projectorLocalNDRange = guessProjectionLocalNDRange(false);
            this->projectorLocalNDRangeBarrier = guessProjectionLocalNDRange(true);
        }
        if(backprojectorLocalNDRangeDim == 3)
        {
            if(backprojectorLocalNDRange[0] == 0 && backprojectorLocalNDRange[1] == 0
               && backprojectorLocalNDRange[2] == 0)
            {
                this->backprojectorLocalNDRange = cl::NDRange();
            } else if(backprojectorLocalNDRange[0] == 0 || backprojectorLocalNDRange[1] == 0
                      || backprojectorLocalNDRange[2] == 0)
            {
                this->backprojectorLocalNDRange = guessBackprojectorLocalNDRange();
            } else
            {
                this->backprojectorLocalNDRange = backprojectorLocalNDRange;
            }
        } else
        {
            if(backprojectorLocalNDRangeDim != 0)
            {
                LOGE << io::xprintf(
                    "Wrong specification of backprojectorLocalNDRange, trying guessing!");
            }
            this->backprojectorLocalNDRange = guessBackprojectorLocalNDRange();
        }
        projectorLocalNDRangeDim = this->projectorLocalNDRange.dimensions();
        backprojectorLocalNDRangeDim = this->backprojectorLocalNDRange.dimensions();
        if(projectorLocalNDRangeDim == 0)
        {
            LOGD << io::xprintf("projectorLocalNDRange = cl::NDRange()");
        } else
        {
            LOGD << io::xprintf("projectorLocalNDRange = cl::NDRange(%d, %d, %d)",
                                this->projectorLocalNDRange[0], this->projectorLocalNDRange[1],
                                this->projectorLocalNDRange[2]);
        }
        if(backprojectorLocalNDRangeDim == 0)
        {
            LOGD << io::xprintf("backprojectorLocalNDRangeDim = cl::NDRange()");
        } else
        {
            LOGD << io::xprintf("backprojectorLocalNDRange = cl::NDRange(%d, %d, %d)",
                                this->backprojectorLocalNDRange[0],
                                this->backprojectorLocalNDRange[1],
                                this->backprojectorLocalNDRange[2]);
        }
    }

    void initializeCVPProjector(bool useExactScaling,
                                bool barrierVariant,
                                uint32_t LOCALARRAYSIZE = 7680);
    void initializeSidonProjector(uint32_t probesPerEdgeX, uint32_t probesPerEdgeY);
    void initializeTTProjector();
    void initializeVolumeConvolution();

    void useJacobiVectorCLCode();

    int problemSetup(float* projection,
                     float* volume,
                     bool volumeContainsX0,
                     std::vector<std::shared_ptr<matrix::CameraI>> camera,
                     double voxelSpacingX,
                     double voxelSpacingY,
                     double voxelSpacingZ,
                     double volumeCenterX = 0.0,
                     double volumeCenterY = 0.0,
                     double volumeCenterZ = 0.0);

    int allocateXBuffers(uint32_t xBufferCount);
    int allocateBBuffers(uint32_t bBufferCount);
    int allocateTmpXBuffers(uint32_t xBufferCount);
    int allocateTmpBBuffers(uint32_t bBufferCount);
    std::shared_ptr<cl::Buffer> getBBuffer(uint32_t i);
    std::shared_ptr<cl::Buffer> getXBuffer(uint32_t i);
    std::shared_ptr<cl::Buffer> getTmpBBuffer(uint32_t i);
    std::shared_ptr<cl::Buffer> getTmpXBuffer(uint32_t i);

    virtual int reconstruct(uint32_t maxItterations, float minDiscrepancyError) = 0;
    double adjointProductTest();
    int vectorIntoBuffer(cl::Buffer X, float* v, std::size_t size);

    static std::vector<std::shared_ptr<CameraI>>
    encodeProjectionMatrices(std::shared_ptr<io::DenProjectionMatrixReader> pm);

    void setReportingParameters(bool verbose,
                                uint32_t reportKthIteration = 0,
                                std::string progressPrefixPath = "");

protected:
    const cl_float FLOATZERO = 0.0;
    const cl_double DOUBLEZERO = 0.0;
    float FLOATONE = 1.0f;
    // Constructor defined variables
    cl_int2 pdims;
    cl_uint2 pdims_uint;
    cl_int3 vdims;

    // Problem setup variables
    double voxelSpacingX, voxelSpacingY, voxelSpacingZ;
    cl_double3 voxelSizes;
    cl_double3 volumeCenter;
    std::vector<std::shared_ptr<CameraI>> cameraVector;
    std::vector<cl_double16> PM12Vector;
    std::vector<cl_double16> ICM16Vector;
    std::vector<float> scalingFactorVector;

    // Variables for projectors and openCL initialization
    bool useCVPProjector = true;
    bool exactProjectionScaling = true;
    bool CVPBarrierImplementation = false;
    uint32_t LOCALARRAYSIZE = 0;
    bool useSidonProjector = false;
    cl_uint2 pixelGranularity = { 1, 1 };
    bool useTTProjector = false;
    bool useVolumeAsInitialX0 = false;

    uint32_t xBufferCount, bBufferCount, tmpXBufferCount, tmpBBufferCount;

    // Class functions
    int initializeVectors(float* projection, float* volume, bool volumeContainsX0);
    void writeVolume(cl::Buffer& X, std::string path);
    void writeProjections(cl::Buffer& B, std::string path);
    std::vector<cl_double16> inverseProjectionMatrices();

    // Printing and reporting
    void setTimestamp(bool finishCommandQueue);
    std::chrono::milliseconds millisecondsFromTimestamp(bool setNewTimestamp);
    std::string printTime(std::string msg, bool finishCommandQueue, bool setNewTimestamp);
    void reportTime(std::string msg, bool finishCommandQueue, bool setNewTimestamp);

    // Functions to manipulate with buffers
    int multiplyVectorsIntoFirstVector(cl::Buffer& A, cl::Buffer& B, uint64_t size);
    int vectorA_multiple_B_equals_C(cl::Buffer& A, cl::Buffer& B, cl::Buffer& C, uint64_t size);
    int copyFloatVector(cl::Buffer& from, cl::Buffer& to, unsigned int size);
    int scaleFloatVector(cl::Buffer& v, float f, unsigned int size);
    int B_equals_A_plus_B_offsets(
        cl::Buffer& A, unsigned int ao, cl::Buffer& B, unsigned int bo, unsigned int size);
    int
    addIntoFirstVectorSecondVectorScaled(cl::Buffer& a, cl::Buffer& b, float f, unsigned int size);
    int
    addIntoFirstVectorScaledSecondVector(cl::Buffer& a, cl::Buffer& b, float f, unsigned int size);
    int invertFloatVector(cl::Buffer& X, unsigned int size);
    std::vector<float> computeScalingFactors();

    cl::NDRange projectorLocalNDRange;
    cl::NDRange projectorLocalNDRangeBarrier;
    cl::NDRange backprojectorLocalNDRange;
    /**
     * Backprojection X = AT(B)
     *
     * @param B Buffer of all projections from all reconstructed angles of the minimal size
     * BDIM*sizeof(float).
     * @param X Buffer of the size at least XDIM*sizeof(float) to be backprojected to.
     * @param initialProjectionIndex For OS SART 0 by default
     * @param projectionIncrement For OS SART 1 by default
     *
     * @return 0 on success
     */
    int backproject(cl::Buffer& B,
                    cl::Buffer& X,
                    uint32_t initialProjectionIndex = 0,
                    uint32_t projectionIncrement = 1);
    int backproject_minmax(cl::Buffer& B,
                           cl::Buffer& X,
                           uint32_t initialProjectionIndex = 0,
                           uint32_t projectionIncrement = 1);

    /**
     * Projection B = A (X)
     *
     * @param X Buffer of the size at least XDIM*sizeof(float) to be projected.
     * @param B Buffer to write all projections from all reconstructed angles of the minimal size
     * BDIM*sizeof(float).
     * @param initialProjectionIndex For OS SART 0 by default
     * @param projectionIncrement For OS SART 1 by default
     *
     * @return
     */
    int project(cl::Buffer& X,
                cl::Buffer& B,
                uint32_t initialProjectionIndex = 0,
                uint32_t projectionIncrement = 1);

    float* x = nullptr; // Volume data
    float* b = nullptr; // Projection data

    // OpenCL buffers
    std::shared_ptr<cl::Buffer> b_buf = nullptr;
    std::shared_ptr<cl::Buffer> x_buf = nullptr;
    // tmp_b_buf for rescaling, tmp_x_buf for LSQR
    std::shared_ptr<cl::Buffer> tmp_x_buf = nullptr, tmp_b_buf = nullptr;
    std::vector<std::shared_ptr<cl::Buffer>> x_buffers, tmp_x_buffers;
    std::vector<std::shared_ptr<cl::Buffer>> b_buffers, tmp_b_buffers;
    std::chrono::time_point<std::chrono::steady_clock> timestamp;

    bool verbose = false;
    std::string progressPrefixPath = "";
    uint32_t reportKthIteration = 0;
};

} // namespace KCT
