//==============================backprojector_minmax.cl=====================================
#ifndef zeroPrecisionTolerance
#define zeroPrecisionTolerance 1e-10
#endif

/// backprojectEdgeValues(INDEXfactor, V, P, projection, pdims);
float inline backprojectMinMaxEdgeValues(global const float* projection,
                                         private REAL16 CM,
                                         private REAL3 v,
                                         private int PX,
                                         private REAL scaledCutArea,
                                         private REAL3 voxelSizes,
                                         private int2 pdims)
{
    const REAL3 distanceToEdge = (REAL3)(ZERO, ZERO, HALF * voxelSizes.s2);
    const REAL3 v_up = v + distanceToEdge;
    const REAL3 v_down = v - distanceToEdge;
    // const REAL3 v_diff = v_down - v_up;
    const REAL negativeEdgeLength = -voxelSizes.s2;
    const REAL PY_up = PROJECTY0(CM, v_up);
    const REAL PY_down = PROJECTY0(CM, v_down);
    const int PJ_up = convert_int_rtn(PY_up + HALF);
    const int PJ_down = convert_int_rtn(PY_down + HALF);
    REAL lambda;
    REAL lastLambda = ZERO;
    REAL leastLambda;
    REAL3 Fvector;
    int PJ_max;
    float ADD = INFINITY;
    float projectionValue, factor;
    if(PJ_down < PJ_up)
    {
        if(PJ_up >= 0 && PJ_down < pdims.y)
        {
            int J;
            if(PJ_down < 0)
            {
                J = 0;
                Fvector = CM.s456 + HALF * CM.s89a;
                // lastLambda = (dot(v_down, Fvector) + CM.s7 + HALF * CM.sb) / (dot(v_diff,
                // Fvector));
                lastLambda = (dot(v_down, Fvector) + CM.s7 + HALF * CM.sb)
                    / (negativeEdgeLength * Fvector.s2);
            } else
            {
                J = PJ_down;
                Fvector = CM.s456 - (J - HALF) * CM.s89a;
            }
            if(PJ_up >= pdims.y)
            {
                PJ_max = pdims.y - 1;
                REAL3 Qvector = CM.s456 - (PJ_max + HALF) * CM.s89a;
                // leastLambda = (dot(v_down, Qvector) + CM.s7 - ((REAL)PJ_max + HALF) * CM.sb)
                //    / (dot(v_diff, Qvector));
                leastLambda = (dot(v_down, Qvector) + CM.s7 - ((REAL)PJ_max + HALF) * CM.sb)
                    / (negativeEdgeLength * Qvector.s2);
            } else
            {
                PJ_max = PJ_up;
                leastLambda = ONE;
            }
            for(; J < PJ_max; J++)
            {
                projectionValue = projection[PX * pdims.y + J];
                Fvector -= CM.s89a; // Fvector = CM.s456 - (J + HALF) * CM.s89a;
                lambda = (dot(v_down, Fvector) + CM.s7 - ((REAL)J + HALF) * CM.sb)
                    / (negativeEdgeLength * Fvector.s2);
                factor = (lambda - lastLambda) * scaledCutArea;
                if(factor > ZERO)
                {
                    ADD = min(ADD, projectionValue / factor);
                }
                lastLambda = lambda;
            }
            projectionValue = projection[PX * pdims.y + PJ_max];
            factor = (leastLambda - lastLambda) * scaledCutArea;
            if(factor > ZERO)
            {
                ADD = min(ADD, projectionValue / factor);
            }
        }
    } else if(PJ_down > PJ_up)
    {
        if(PJ_down >= 0 && PJ_up < pdims.y)
        {
            // We will count with negative scaledCutArea of lambda by dividing by dot(v_diff,
            // Fvector) instead of dot(-v_diff, Fvector)  Because scaledCutAreaPerUnit is negative
            // the scaledCutArea (lambda
            // - lastLambda)*scaledCutAreaPerUnit will be positive
            // lambda here measures negative distance from v_up to a given intersection point
            int J;
            if(PJ_up < 0)
            {
                J = 0;
                Fvector = CM.s456 + HALF * CM.s89a;
                lastLambda = (dot(v_up, Fvector) + CM.s7 + HALF * CM.sb)
                    / (negativeEdgeLength * Fvector.s2);
            } else
            {
                J = PJ_up;
                Fvector = CM.s456 - (J - HALF) * CM.s89a;
            }
            if(PJ_down >= pdims.y)
            {
                PJ_max = pdims.y - 1;
                REAL3 Qvector = CM.s456 - (PJ_max + HALF) * CM.s89a;
                leastLambda = (dot(v_up, Qvector) + CM.s7 - ((REAL)PJ_max + HALF) * CM.sb)
                    / (negativeEdgeLength * Qvector.s2);
            } else
            {
                PJ_max = PJ_down;
                leastLambda = -ONE;
            }
            for(; J < PJ_max; J++)
            {
                projectionValue = projection[PX * pdims.y + J];
                Fvector -= CM.s89a; // Fvector = CM.s456 - (J + HALF) * CM.s89a;
                lambda = (dot(v_up, Fvector) + CM.s7 - ((REAL)J + HALF) * CM.sb)
                    / (negativeEdgeLength * Fvector.s2);
                factor = (lastLambda - lambda) * scaledCutArea;
                if(factor > ZERO)
                {
                    ADD = min(ADD, projectionValue / factor);
                }
                lastLambda = lambda;
            }
            // PJ_max
            projectionValue = projection[PX * pdims.y + PJ_max];
            factor = (lastLambda - leastLambda) * scaledCutArea;
            if(factor > ZERO)
            {
                ADD = min(ADD, projectionValue / factor);
            }
        }
    } else if(PJ_down == PJ_up && PJ_down >= 0 && PJ_down < pdims.y)
    {
        projectionValue = projection[PX * pdims.y + PJ_down];
        factor = scaledCutArea;
        if(factor > ZERO)
        {
            ADD = min(ADD, projectionValue / factor);
        }
    }
    return ADD;
}

/** Project given volume using cutting voxel projector.
 *
 *
 * @param volume Volume to project.
 * @param projection Projection to construct.
 * @param CM Projection matrix. This projection matrix is constructed in the way that (i,j,k) =
 * (0,0,0,1) is projected to the center of the voxel with given coordinates.
 * @param sourcePosition Source position in the xyz space.
 * @param normalToDetector Normal to detector in the (i,j,k) space.
 * @param vdims Dimensions of the volume.
 * @param voxelSizes Lengths of the voxel edges.
 * @param pdims Dimensions of the projection.
 * @param scalingFactor Scale the results by this factor.
 * @param dummy Parameter not to SEGFAULT Intel.
 *
 * @return
 */
void kernel FLOATcutting_voxel_minmaxbackproject(global float* restrict volume,
                                                 global const float* restrict projection,
                                                 private uint projectionOffset,
                                                 private double16 _CM,
                                                 private double3 _sourcePosition,
                                                 private double3 _normalToDetector,
                                                 private int3 vdims,
                                                 private double3 _voxelSizes,
                                                 private double3 _volumeCenter,
                                                 private int2 pdims,
                                                 private float globalScalingMultiplier,
                                                 private int2 dummy)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);
#ifdef RELAXED
    const float16 CM = convert_float16(_CM);
    const float3 sourcePosition = convert_float3(_sourcePosition);
    const float3 voxelSizes = convert_float3(_voxelSizes);
    const float3 volumeCenter = convert_float3(_volumeCenter);
#else
#define CM _CM
#define sourcePosition _sourcePosition
#define voxelSizes _voxelSizes
#define volumeCenter _volumeCenter
#endif
    const REAL3 halfVoxelSizes = HALF * voxelSizes;
    const REAL3 volumeCenter_voxelcenter_offset
        = (REAL3)(2 * i + 1 - vdims.x, 2 * j + 1 - vdims.y, 2 * k + 1 - vdims.z) * halfVoxelSizes;
    const REAL3 voxelcenter_xyz = volumeCenter + volumeCenter_voxelcenter_offset - sourcePosition;
    const uint IND = voxelIndex(i, j, k, vdims);
    float ADD = INFINITY;
    const REAL voxelVolume = voxelSizes.x * voxelSizes.y * voxelSizes.z;
    REAL sourceToVoxel_xyz_norm2 = dot(voxelcenter_xyz, voxelcenter_xyz);
#ifdef RELAXED
    float value = (voxelVolume * globalScalingMultiplier / sourceToVoxel_xyz_norm2);
#else
    float value = (float)(voxelVolume * globalScalingMultiplier / sourceToVoxel_xyz_norm2);
#endif

    REAL px00, px10, px01, px11;
    REAL3 vx00, vx10, vx01, vx11;
    vx00 = voxelcenter_xyz + halfVoxelSizes * (REAL3)(-ONE, -ONE, ZERO);
    vx10 = voxelcenter_xyz + halfVoxelSizes * (REAL3)(ONE, -ONE, ZERO);
    vx01 = voxelcenter_xyz + halfVoxelSizes * (REAL3)(-ONE, ONE, ZERO);
    vx11 = voxelcenter_xyz + halfVoxelSizes * (REAL3)(ONE, ONE, ZERO);
    {
        REAL nx = dot(voxelcenter_xyz, CM.s012);
        REAL dv = dot(voxelcenter_xyz, CM.s89a);
        REAL nhx = halfVoxelSizes.x * CM.s0;
        REAL nhy = halfVoxelSizes.y * CM.s1;
        REAL dhx = halfVoxelSizes.x * CM.s8;
        REAL dhy = halfVoxelSizes.y * CM.s9;
        px00 = (nx - nhx - nhy) / (dv - dhx - dhy);
        px01 = (nx - nhx + nhy) / (dv - dhx + dhy);
        px10 = (nx + nhx - nhy) / (dv + dhx - dhy);
        px11 = (nx + nhx + nhy) / (dv + dhx + dhy);
    }
   REAL pxx_min, pxx_max; // Minimum and maximum values of projector x coordinate
    int max_PX,
        min_PX; // Pixel to which are the voxels with minimum and maximum values are
                // projected
    // pxx_min = fmin(fmin(px00, px10), fmin(px01, px11));
    // pxx_max = fmax(fmax(px00, px10), fmax(px01, px11));
    REAL3* V_ccw[4]; // Point in which minimum is achieved and counter clock wise
                     // points
    // from the minimum voxel
    REAL* PX_ccw[4]; // Point in which minimum is achieved and counter clock wise
                     // points

    if(px00 < px10)
    {
        if(px00 < px01)
        {
            pxx_min = px00;
            V_ccw[0] = &vx00;
            V_ccw[1] = &vx10;
            V_ccw[2] = &vx11;
            V_ccw[3] = &vx01;
            PX_ccw[0] = &px00;
            PX_ccw[1] = &px10;
            PX_ccw[2] = &px11;
            PX_ccw[3] = &px01;
            if(px01 > px11)
            {
                pxx_max = px01;
            } else if(px10 > px11)
            {
                pxx_max = px10;
            } else
            {
                pxx_max = px11;
            }
        } else if(px01 < px11)
        {
            pxx_min = px01;
            V_ccw[0] = &vx01;
            V_ccw[1] = &vx00;
            V_ccw[2] = &vx10;
            V_ccw[3] = &vx11;
            PX_ccw[0] = &px01;
            PX_ccw[1] = &px00;
            PX_ccw[2] = &px10;
            PX_ccw[3] = &px11;
            if(px10 > px11)
            {
                pxx_max = px10;
            } else
            {
                pxx_max = px11;
            }
        } else
        {
            pxx_min = px11;
            pxx_max = px10;
            V_ccw[0] = &vx11;
            V_ccw[1] = &vx01;
            V_ccw[2] = &vx00;
            V_ccw[3] = &vx10;
            PX_ccw[0] = &px11;
            PX_ccw[1] = &px01;
            PX_ccw[2] = &px00;
            PX_ccw[3] = &px10;
        }
    } else if(px10 < px11)
    {
        pxx_min = px10;
        V_ccw[0] = &vx10;
        V_ccw[1] = &vx11;
        V_ccw[2] = &vx01;
        V_ccw[3] = &vx00;
        PX_ccw[0] = &px10;
        PX_ccw[1] = &px11;
        PX_ccw[2] = &px01;
        PX_ccw[3] = &px00;
        if(px00 > px01)
        {
            pxx_max = px00;
        } else if(px11 > px01)
        {
            pxx_max = px11;
        } else
        {
            pxx_max = px01;
        }
    } else if(px11 < px01)
    {
        pxx_min = px11;
        V_ccw[0] = &vx11;
        V_ccw[1] = &vx01;
        V_ccw[2] = &vx00;
        V_ccw[3] = &vx10;
        PX_ccw[0] = &px11;
        PX_ccw[1] = &px01;
        PX_ccw[2] = &px00;
        PX_ccw[3] = &px10;
        if(px00 > px01)
        {
            pxx_max = px00;
        } else
        {
            pxx_max = px01;
        }
    } else
    {
        pxx_min = px01;
        pxx_max = px00;
        V_ccw[0] = &vx01;
        V_ccw[1] = &vx00;
        V_ccw[2] = &vx10;
        V_ccw[3] = &vx11;
        PX_ccw[0] = &px01;
        PX_ccw[1] = &px00;
        PX_ccw[2] = &px10;
        PX_ccw[3] = &px11;
    }

    min_PX = convert_int_rtn(pxx_min + zeroPrecisionTolerance + HALF);
    max_PX = convert_int_rtn(pxx_max - zeroPrecisionTolerance + HALF);

    if(max_PX >= 0 && min_PX < pdims.x)
    {
        REAL3 vd1 = (*V_ccw[1]) - (*V_ccw[0]);
        REAL3 vd3 = (*V_ccw[3]) - (*V_ccw[0]);
        float intermediateValue;
        if(max_PX <= min_PX) // These indices are in the admissible range
        {
            min_PX = convert_int_rtn(HALF * (pxx_min + pxx_max) + HALF);
            ADD = backprojectMinMaxEdgeValues(&projection[projectionOffset], CM,
                                              HALF * (vx10 + vx01), min_PX, value,
                                              voxelSizes, pdims);
            volume[IND] = min(ADD, volume[IND]);
        } else
        {
            REAL lastRectangleSectionRelativeArea, nextRectangleSectionRelativeArea,
                relativeCutArea;
            REAL3 lastInt, nextInt, Int;
            int I = max(-1, min_PX);
            int I_STOP = min(max_PX, pdims.x);
            int numberOfEdges;
            REAL scaledCutArea;
            // Section of the square that corresponds to the indices < i
            // CCW and CW coordinates of the last intersection on the lines specified by the points
            // in V_ccw lastRectangleSectionRelativeArea
            //    = findIntersectionPoints(((REAL)I) + HALF, V_ccw[0], V_ccw[1], V_ccw[2],
            //    V_ccw[3],
            //                             PX_ccw[0], PX_ccw[1], PX_ccw[2], PX_ccw[3], &lastInt);
            lastRectangleSectionRelativeArea = exactIntersectionPoints0_extended(
                ((REAL)I) + HALF, V_ccw[0], V_ccw[1], V_ccw[2], V_ccw[3], vd1, vd3, PX_ccw[0],
                PX_ccw[1], PX_ccw[2], PX_ccw[3], CM, &lastInt);
            if(I >= 0)
            {
                scaledCutArea = value * lastRectangleSectionRelativeArea;
                intermediateValue
                    = backprojectMinMaxEdgeValues(&projection[projectionOffset], CM, lastInt, I,
                                                  scaledCutArea, voxelSizes, pdims);
                ADD = min(intermediateValue, ADD);
            }
            for(I = I + 1; I < I_STOP; I++)
            {
                // nextRectangleSectionRelativeArea
                //    = findIntersectionPoints(((REAL)I) + HALF, V_ccw[0], V_ccw[1], V_ccw[2],
                //    V_ccw[3],
                //                             PX_ccw[0], PX_ccw[1], PX_ccw[2], PX_ccw[3],
                //                             &nextInt);
                nextRectangleSectionRelativeArea = exactIntersectionPoints0_extended(
                    ((REAL)I) + HALF, V_ccw[0], V_ccw[1], V_ccw[2], V_ccw[3], vd1, vd3, PX_ccw[0],
                    PX_ccw[1], PX_ccw[2], PX_ccw[3], CM, &nextInt);
                relativeCutArea
                    = nextRectangleSectionRelativeArea - lastRectangleSectionRelativeArea;
                Int = (nextRectangleSectionRelativeArea * nextInt
                       - lastRectangleSectionRelativeArea * lastInt)
                    / relativeCutArea;
                scaledCutArea = value * relativeCutArea;
                intermediateValue = backprojectMinMaxEdgeValues(
                    &projection[projectionOffset], CM, Int, I, scaledCutArea, voxelSizes, pdims);
                ADD = min(intermediateValue, ADD);
                lastRectangleSectionRelativeArea = nextRectangleSectionRelativeArea;
                lastInt = nextInt;
            }
            if(I_STOP < pdims.x)
            {
                relativeCutArea = ONE - lastRectangleSectionRelativeArea;
                Int = ((*V_ccw[0] + *V_ccw[2]) * HALF - lastRectangleSectionRelativeArea * lastInt)
                    / relativeCutArea;
                scaledCutArea = value * relativeCutArea;
                intermediateValue = backprojectMinMaxEdgeValues(
                    &projection[projectionOffset], CM, Int, I, scaledCutArea, voxelSizes, pdims);
                ADD = min(intermediateValue, ADD);
            }
            volume[IND] = min(ADD, volume[IND]);
        }
    }
}
//==============================END File/backprojector_minmax.cl==================================
