//==============================backprojector_minmax.cl=====================================
#ifndef zeroPrecisionTolerance
#define zeroPrecisionTolerance 1e-10
#endif

/// backprojectEdgeValues(INDEXfactor, V, P, projection, pdims);
float inline backprojectMinMaxEdgeValues(global const float* projection,
                                         private double16 CM,
                                         private double3 v,
                                         private int PX,
                                         private double scaledCutArea,
                                         private double3 voxelSizes,
                                         private int2 pdims)
{
    const double3 distanceToEdge = (double3)(0.0, 0.0, 0.5 * voxelSizes.s2);
    const double3 v_up = v + distanceToEdge;
    const double3 v_down = v - distanceToEdge;
    // const double3 v_diff = v_down - v_up;
    const double negativeEdgeLength = -voxelSizes.s2;
    const double PY_up = projectY(CM, v_up);
    const double PY_down = projectY(CM, v_down);
    const int PJ_up = convert_int_rtn(PY_up + 0.5);
    const int PJ_down = convert_int_rtn(PY_down + 0.5);
    double lambda;
    double lastLambda = 0.0;
    double leastLambda;
    double3 Fvector;
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
                Fvector = CM.s456 + 0.5 * CM.s89a;
                // lastLambda = (dot(v_down, Fvector) + CM.s7 + 0.5 * CM.sb) / (dot(v_diff,
                // Fvector));
                lastLambda = (dot(v_down, Fvector) + CM.s7 + 0.5 * CM.sb)
                    / (negativeEdgeLength * Fvector.s2);
            } else
            {
                J = PJ_down;
                Fvector = CM.s456 - (J - 0.5) * CM.s89a;
            }
            if(PJ_up >= pdims.y)
            {
                PJ_max = pdims.y - 1;
                double3 Qvector = CM.s456 - (PJ_max + 0.5) * CM.s89a;
                // leastLambda = (dot(v_down, Qvector) + CM.s7 - ((double)PJ_max + 0.5) * CM.sb)
                //    / (dot(v_diff, Qvector));
                leastLambda = (dot(v_down, Qvector) + CM.s7 - ((double)PJ_max + 0.5) * CM.sb)
                    / (negativeEdgeLength * Qvector.s2);
            } else
            {
                PJ_max = PJ_up;
                leastLambda = 1.0;
            }
            for(; J < PJ_max; J++)
            {
                projectionValue = projection[PX + pdims.x * J];
                Fvector -= CM.s89a; // Fvector = CM.s456 - (J + 0.5) * CM.s89a;
                lambda = (dot(v_down, Fvector) + CM.s7 - ((double)J + 0.5) * CM.sb)
                    / (negativeEdgeLength * Fvector.s2);
                factor = (lambda - lastLambda) * scaledCutArea;
                if(factor > 0.0)
                {
                    ADD = min(ADD, projectionValue / factor);
                }
                lastLambda = lambda;
            }
            projectionValue = projection[PX + pdims.x * PJ_max];
            factor = (leastLambda - lastLambda) * scaledCutArea;
            if(factor > 0.0)
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
                Fvector = CM.s456 + 0.5 * CM.s89a;
                lastLambda = (dot(v_up, Fvector) + CM.s7 + 0.5 * CM.sb)
                    / (negativeEdgeLength * Fvector.s2);
            } else
            {
                J = PJ_up;
                Fvector = CM.s456 - (J - 0.5) * CM.s89a;
            }
            if(PJ_down >= pdims.y)
            {
                PJ_max = pdims.y - 1;
                double3 Qvector = CM.s456 - (PJ_max + 0.5) * CM.s89a;
                leastLambda = (dot(v_up, Qvector) + CM.s7 - ((double)PJ_max + 0.5) * CM.sb)
                    / (negativeEdgeLength * Qvector.s2);
            } else
            {
                PJ_max = PJ_down;
                leastLambda = -1.0;
            }
            for(; J < PJ_max; J++)
            {
                projectionValue = projection[PX + pdims.x * J];
                Fvector -= CM.s89a; // Fvector = CM.s456 - (J + 0.5) * CM.s89a;
                lambda = (dot(v_up, Fvector) + CM.s7 - ((double)J + 0.5) * CM.sb)
                    / (negativeEdgeLength * Fvector.s2);
                factor = (lastLambda - lambda) * scaledCutArea;
                if(factor > 0.0)
                {
                    ADD = min(ADD, projectionValue / factor);
                }
                lastLambda = lambda;
            }
            // PJ_max
            projectionValue = projection[PX + pdims.x * PJ_max];
            factor = (lastLambda - leastLambda) * scaledCutArea;
            if(factor > 0.0)
            {
                ADD = min(ADD, projectionValue / factor);
            }
        }
    } else if(PJ_down == PJ_up && PJ_down >= 0 && PJ_down < pdims.y)
    {
        projectionValue = projection[PX + pdims.x * PJ_down];
        factor = scaledCutArea;
        if(factor > 0.0)
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
                                                 private double16 CM,
                                                 private double3 sourcePosition,
                                                 private double3 normalToDetector,
                                                 private int3 vdims,
                                                 private double3 voxelSizes,
                                                 private double3 volumeCenter,
                                                 private int2 pdims,
                                                 private float globalScalingMultiplier,
                                                 private int2 dummy)
{
    int i = get_global_id(2);
    int j = get_global_id(1);
    int k = get_global_id(0); // This is more effective from the perspective of atomic colisions
    float ADD = INFINITY;
    const double3 IND_ijk = { (double)(i), (double)(j), (double)(k) };
    const double3 zerocorner_xyz
        = volumeCenter - sourcePosition - 0.5 * convert_double3(vdims) * voxelSizes;
    const double3 voxelcorner_xyz = zerocorner_xyz + (IND_ijk * voxelSizes);
    const double3 voxelcenter_xyz = voxelcorner_xyz + 0.5 * voxelSizes;
    const uint IND = voxelIndex(i, j, k, vdims);
    const float voxelVolume = voxelSizes.x * voxelSizes.y * voxelSizes.z;
    int cornerProjectionIndex = projectionIndex0(CM, voxelcorner_xyz, pdims);
    float scalingFactor;
    if(cornerProjectionIndex
           == projectionIndex0(CM, voxelcorner_xyz + voxelSizes * (double3)(1.0, 1.0, 1.0), pdims)
       && cornerProjectionIndex
           == projectionIndex0(CM, voxelcorner_xyz + voxelSizes * (double3)(1.0, 1.0, 0.0), pdims)
       && cornerProjectionIndex
           == projectionIndex0(CM, voxelcorner_xyz + voxelSizes * (double3)(1.0, 0.0, 1.0), pdims)
       && cornerProjectionIndex
           == projectionIndex0(CM, voxelcorner_xyz + voxelSizes * (double3)(0.0, 1.0, 1.0), pdims)
       && cornerProjectionIndex
           == projectionIndex0(CM, voxelcorner_xyz + voxelSizes * (double3)(1.0, 0.0, 0.0), pdims)
       && cornerProjectionIndex
           == projectionIndex0(CM, voxelcorner_xyz + voxelSizes * (double3)(0.0, 1.0, 0.0), pdims)
       && cornerProjectionIndex
           == projectionIndex0(CM, voxelcorner_xyz + voxelSizes * (double3)(0.0, 0.0, 1.0),
                               pdims)) // When all projections are the same
    {
        if(cornerProjectionIndex != -1)
        {
            double sourceToVoxel_xyz_norm2 = dot(voxelcenter_xyz, voxelcenter_xyz);
            scalingFactor = globalScalingMultiplier * voxelVolume / sourceToVoxel_xyz_norm2;
            ADD = projection[projectionOffset + cornerProjectionIndex] / scalingFactor;
            volume[IND] = min(ADD, volume[IND]);
        }
        return;
    }
    double sourceToVoxel_xyz_norm2 = dot(voxelcenter_xyz, voxelcenter_xyz);
    scalingFactor = globalScalingMultiplier * voxelVolume / sourceToVoxel_xyz_norm2;
    // I assume that the volume point (x,y,z_1) projects to the same px as (x,y,z_2) for any z_1,
    // z_2  This assumption is restricted to the voxel edges, where it holds very accurately  We
    // project the rectangle that lies on the z midline of the voxel on the projector
    double px00, px01, px10, px11;
    double3 vx00, vx01, vx10, vx11;
    vx00 = voxelcorner_xyz + voxelSizes * (double3)(0.0, 0.0, 0.5);
    vx01 = voxelcorner_xyz + voxelSizes * (double3)(1.0, 0.0, 0.5);
    vx10 = voxelcorner_xyz + voxelSizes * (double3)(0.0, 1.0, 0.5);
    vx11 = voxelcorner_xyz + voxelSizes * (double3)(1.0, 1.0, 0.5);
    px00 = projectX0(CM, vx00);
    px01 = projectX0(CM, vx01);
    px10 = projectX0(CM, vx10);
    px11 = projectX0(CM, vx11);
    // We now figure out the vertex that projects to minimum and maximum px
    double pxx_min, pxx_max; // Minimum and maximum values of projector x coordinate
    int max_PX,
        min_PX; // Pixel to which are the voxels with minimum and maximum values are
                // projected
    // pxx_min = fmin(fmin(px00, px01), fmin(px10, px11));
    // pxx_max = fmax(fmax(px00, px01), fmax(px10, px11));
    double3* V_ccw[4]; // Point in which maximum is achieved and counter clock wise points
    // from the minimum voxel
    double* PX_ccw[4]; // Point in which maximum is achieved and counter clock wise  points
    // from the minimum voxel
    if(px00 < px01)
    {
        if(px00 < px10)
        {
            pxx_min = px00;
            V_ccw[0] = &vx00;
            V_ccw[1] = &vx01;
            V_ccw[2] = &vx11;
            V_ccw[3] = &vx10;
            PX_ccw[0] = &px00;
            PX_ccw[1] = &px01;
            PX_ccw[2] = &px11;
            PX_ccw[3] = &px10;
            if(px10 > px11)
            {
                pxx_max = px10;
            } else if(px01 > px11)
            {
                pxx_max = px01;
            } else
            {
                pxx_max = px11;
            }
        } else if(px10 < px11)
        {
            pxx_min = px10;
            V_ccw[0] = &vx10;
            V_ccw[1] = &vx00;
            V_ccw[2] = &vx01;
            V_ccw[3] = &vx11;
            PX_ccw[0] = &px10;
            PX_ccw[1] = &px00;
            PX_ccw[2] = &px01;
            PX_ccw[3] = &px11;
            if(px01 > px11)
            {
                pxx_max = px01;
            } else
            {
                pxx_max = px11;
            }
        } else
        {
            pxx_min = px11;
            pxx_max = px01;
            V_ccw[0] = &vx11;
            V_ccw[1] = &vx10;
            V_ccw[2] = &vx00;
            V_ccw[3] = &vx01;
            PX_ccw[0] = &px11;
            PX_ccw[1] = &px10;
            PX_ccw[2] = &px00;
            PX_ccw[3] = &px01;
        }

    } else if(px01 < px11)
    {
        pxx_min = px01;
        V_ccw[0] = &vx01;
        V_ccw[1] = &vx11;
        V_ccw[2] = &vx10;
        V_ccw[3] = &vx00;
        PX_ccw[0] = &px01;
        PX_ccw[1] = &px11;
        PX_ccw[2] = &px10;
        PX_ccw[3] = &px00;
        if(px00 > px10)
        {
            pxx_max = px00;
        } else if(px11 > px10)
        {
            pxx_max = px11;
        } else
        {
            pxx_max = px10;
        }
    } else if(px11 < px10)
    {
        pxx_min = px11;
        V_ccw[0] = &vx11;
        V_ccw[1] = &vx10;
        V_ccw[2] = &vx00;
        V_ccw[3] = &vx01;
        PX_ccw[0] = &px11;
        PX_ccw[1] = &px10;
        PX_ccw[2] = &px00;
        PX_ccw[3] = &px01;
        if(px00 > px10)
        {
            pxx_max = px00;
        } else
        {
            pxx_max = px10;
        }
    } else
    {
        pxx_min = px10;
        pxx_max = px00;
        V_ccw[0] = &vx10;
        V_ccw[1] = &vx00;
        V_ccw[2] = &vx01;
        V_ccw[3] = &vx11;
        PX_ccw[0] = &px10;
        PX_ccw[1] = &px00;
        PX_ccw[2] = &px01;
        PX_ccw[3] = &px11;
    }

    min_PX = convert_int_rtn(pxx_min + zeroPrecisionTolerance + 0.5);
    max_PX = convert_int_rtn(pxx_max - zeroPrecisionTolerance + 0.5);
    if(max_PX >= 0 && min_PX < pdims.x)
    {
        double3 vd1 = (*V_ccw[1]) - (*V_ccw[0]);
        double3 vd3 = (*V_ccw[3]) - (*V_ccw[0]);
        float intermediateValue;
        if(max_PX <= min_PX) // These indices are in the admissible range
        {
            min_PX = convert_int_rtn(0.5 * (pxx_min + pxx_max) + 0.5);
            ADD = backprojectMinMaxEdgeValues(&projection[projectionOffset], CM,
                                              (vx10 + vx01) / 2.0, min_PX, scalingFactor,
                                              voxelSizes, pdims);
            volume[IND] = min(ADD, volume[IND]);
        } else
        {
            double lastRectangleSectionRelativeArea, nextRectangleSectionRelativeArea,
                relativeCutArea;
            double3 lastInt, nextInt, Int;
            int I = max(-1, min_PX);
            int I_STOP = min(max_PX, pdims.x);
            int numberOfEdges;
            double scaledCutArea;
            // Section of the square that corresponds to the indices < i
            // CCW and CW coordinates of the last intersection on the lines specified by the points
            // in V_ccw lastRectangleSectionRelativeArea
            //    = findIntersectionPoints(((double)I) + 0.5, V_ccw[0], V_ccw[1], V_ccw[2],
            //    V_ccw[3],
            //                             PX_ccw[0], PX_ccw[1], PX_ccw[2], PX_ccw[3], &lastInt);
            lastRectangleSectionRelativeArea = exactIntersectionPoints0_extended(
                ((double)I) + 0.5, V_ccw[0], V_ccw[1], V_ccw[2], V_ccw[3], vd1, vd3, PX_ccw[0],
                PX_ccw[1], PX_ccw[2], PX_ccw[3], CM, &lastInt);
            if(I >= 0)
            {
                scaledCutArea = scalingFactor * lastRectangleSectionRelativeArea;
                intermediateValue
                    = backprojectMinMaxEdgeValues(&projection[projectionOffset], CM, lastInt, I,
                                                  scaledCutArea, voxelSizes, pdims);
                ADD = min(intermediateValue, ADD);
            }
            for(I = I + 1; I < I_STOP; I++)
            {
                // nextRectangleSectionRelativeArea
                //    = findIntersectionPoints(((double)I) + 0.5, V_ccw[0], V_ccw[1], V_ccw[2],
                //    V_ccw[3],
                //                             PX_ccw[0], PX_ccw[1], PX_ccw[2], PX_ccw[3],
                //                             &nextInt);
                nextRectangleSectionRelativeArea = exactIntersectionPoints0_extended(
                    ((double)I) + 0.5, V_ccw[0], V_ccw[1], V_ccw[2], V_ccw[3], vd1, vd3, PX_ccw[0],
                    PX_ccw[1], PX_ccw[2], PX_ccw[3], CM, &nextInt);
                relativeCutArea
                    = nextRectangleSectionRelativeArea - lastRectangleSectionRelativeArea;
                Int = (nextRectangleSectionRelativeArea * nextInt
                       - lastRectangleSectionRelativeArea * lastInt)
                    / relativeCutArea;
                scaledCutArea = scalingFactor * relativeCutArea;
                intermediateValue = backprojectMinMaxEdgeValues(
                    &projection[projectionOffset], CM, Int, I, scaledCutArea, voxelSizes, pdims);
                ADD = min(intermediateValue, ADD);
                lastRectangleSectionRelativeArea = nextRectangleSectionRelativeArea;
                lastInt = nextInt;
            }
            if(I_STOP < pdims.x)
            {
                relativeCutArea = 1 - lastRectangleSectionRelativeArea;
                Int = ((*V_ccw[0] + *V_ccw[2]) * 0.5 - lastRectangleSectionRelativeArea * lastInt)
                    / relativeCutArea;
                scaledCutArea = scalingFactor * relativeCutArea;
                intermediateValue = backprojectMinMaxEdgeValues(
                    &projection[projectionOffset], CM, Int, I, scaledCutArea, voxelSizes, pdims);
                ADD = min(intermediateValue, ADD);
            }
            volume[IND] = min(ADD, volume[IND]);
        }
    }
}
//==============================END File/backprojector_minmax.cl==================================
