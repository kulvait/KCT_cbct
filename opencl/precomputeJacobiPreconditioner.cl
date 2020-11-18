/// backprojectEdgeValues(INDEXfactor, V, P, projection, pdims);
float inline exactJacobiValues(private double16 CM,
                               private double3 v,
                               private int PX,
                               private double value,
                               private double3 voxelSizes,
                               private int2 pdims)
{
    float ADD = 0.0f;
    float ad;
    const double3 distanceToEdge = (double3)(0.0, 0.0, 0.5 * voxelSizes.s2);
    double3 v_up = v + distanceToEdge;
    double3 v_down = v - distanceToEdge;
    double negativeEdgeLength = -voxelSizes.s2;
    double PY_up = projectY(CM, v_up);
    double PY_down = projectY(CM, v_down);
    int PJ_up = convert_int_rtn(PY_up + 0.5);
    int PJ_down = convert_int_rtn(PY_down + 0.5);
    double lambda;
    double lastLambda = 0.0;
    double leastLambda;
    double3 Fvector;
    int PJ_max;

    if(PY_down > PY_up)
    {
        int tmp_i;
        double tmp_d;
        double3 tmp_d3;
        tmp_i = PJ_down;
        PJ_down = PJ_up;
        PJ_up = tmp_i;
        tmp_d = PY_down;
        PY_down = PY_up;
        PY_up = tmp_d;
        tmp_d3 = v_down;
        v_down = v_up;
        v_up = v_down;
    }

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
                Fvector -= CM.s89a; // Fvector = CM.s456 - (J + 0.5) * CM.s89a;
                lambda = (dot(v_down, Fvector) + CM.s7 - ((double)J + 0.5) * CM.sb)
                    / (negativeEdgeLength * Fvector.s2);
                ad = (lambda - lastLambda) * value;
                ADD += ad * ad;
                // Atomic version of projection[ind] += value;
                lastLambda = lambda;
            }
            // PJ_max
            ad = (leastLambda - lastLambda) * value;
            ADD += ad * ad;
        }
    } else if(PJ_down == PJ_up && PJ_down >= 0 && PJ_down < pdims.y)
    {
        ADD += value * value;
    }
    return ADD;
}

/** This functions precomputes diag(A^T A) to use as Jacobi preconditioner.
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
 *
 * @return
 */
void kernel FLOATcutting_voxel_jacobiPreconditionerVector(global float* volumePreconditionerVector,
                                                          private double16 CM,
                                                          private double3 sourcePosition,
                                                          private double3 normalToDetector,
                                                          private int3 vdims,
                                                          private double3 voxelSizes,
                                                          private int2 pdims,
                                                          private float scalingFactor)
{
    // I can not rescale projections here so I have to use the logic with proper scaling factor
    uint i = get_global_id(2);
    uint j = get_global_id(1);
    uint k = get_global_id(0); // This is more effective from the perspective of atomic colisions
    float ADD = 0.0;
    const double3 IND_ijk = { (double)(i), (double)(j), (double)(k) };
    const double3 zerocorner_xyz
        = { -0.5 * (double)vdims.x * voxelSizes.x, -0.5 * (double)vdims.y * voxelSizes.y,
            -0.5 * (double)vdims.z * voxelSizes.z }; // -convert_double3(vdims) / 2.0;
    const double3 voxelcorner_xyz = zerocorner_xyz
        + (IND_ijk * voxelSizes); // Using widening and vector multiplication operations
    const uint IND = voxelIndex(i, j, k, vdims);
    const float voxelVolume = voxelSizes.x * voxelSizes.y * voxelSizes.z;
    const double3 voxelcenter_xyz
        = voxelcorner_xyz + voxelSizes * 0.5; // Using widening and vector multiplication operations
    int cornerProjectionIndex = projectionIndex(CM, voxelcorner_xyz, pdims);
    if(cornerProjectionIndex
           == projectionIndex(CM, voxelcorner_xyz + voxelSizes * (double3)(1.0, 1.0, 1.0), pdims)
       && cornerProjectionIndex
           == projectionIndex(CM, voxelcorner_xyz + voxelSizes * (double3)(1.0, 1.0, 0.0), pdims)
       && cornerProjectionIndex
           == projectionIndex(CM, voxelcorner_xyz + voxelSizes * (double3)(1.0, 0.0, 1.0), pdims)
       && cornerProjectionIndex
           == projectionIndex(CM, voxelcorner_xyz + voxelSizes * (double3)(0.0, 1.0, 1.0), pdims)
       && cornerProjectionIndex
           == projectionIndex(CM, voxelcorner_xyz + voxelSizes * (double3)(1.0, 0.0, 0.0), pdims)
       && cornerProjectionIndex
           == projectionIndex(CM, voxelcorner_xyz + voxelSizes * (double3)(0.0, 1.0, 0.0), pdims)
       && cornerProjectionIndex
           == projectionIndex(CM, voxelcorner_xyz + voxelSizes * (double3)(0.0, 0.0, 1.0),
                              pdims)) // When all projections are the same
    {
        if(cornerProjectionIndex != -1)
        {
            double3 sourceToVoxel_xyz = voxelcenter_xyz - sourcePosition;
            double sourceToVoxel_xyz_norm = length(sourceToVoxel_xyz);
            double cosine = dot(normalToDetector, sourceToVoxel_xyz) / sourceToVoxel_xyz_norm;
            double cosPowThree = cosine * cosine * cosine;
            float value
                = scalingFactor / (sourceToVoxel_xyz_norm * sourceToVoxel_xyz_norm * cosPowThree);
            ADD = value * voxelSizes.x * voxelSizes.y * voxelSizes.z;
            ADD = ADD * ADD;
            volumePreconditionerVector[IND] += ADD;
        }
        return;
    }
    double3 sourceToVoxel_xyz = voxelcenter_xyz - sourcePosition;
    double sourceToVoxel_xyz_norm = length(sourceToVoxel_xyz);
    double cosine = dot(normalToDetector, sourceToVoxel_xyz) / sourceToVoxel_xyz_norm;
    double cosPowThree = cosine * cosine * cosine;
    float value = scalingFactor / (sourceToVoxel_xyz_norm * sourceToVoxel_xyz_norm * cosPowThree);

    double px00, px01, px10, px11;
    double3 vx00, vx01, vx10, vx11;
    vx00 = voxelcorner_xyz + voxelSizes * (double3)(0.0, 0.0, 0.5);
    vx01 = voxelcorner_xyz + voxelSizes * (double3)(1.0, 0.0, 0.5);
    vx10 = voxelcorner_xyz + voxelSizes * (double3)(0.0, 1.0, 0.5);
    vx11 = voxelcorner_xyz + voxelSizes * (double3)(1.0, 1.0, 0.5);
    px00 = projectX(CM, vx00);
    px01 = projectX(CM, vx01);
    px10 = projectX(CM, vx10);
    px11 = projectX(CM, vx11);
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
        if(max_PX <= min_PX) // These indices are in the admissible range
        {
            min_PX = convert_int_rtn(0.5 * (pxx_min + pxx_max) + 0.5);
            ADD += exactJacobiValues(CM, (vx00 + vx11) / 2, min_PX, value, voxelSizes, pdims);
        } else
        {

            double lastSectionSize, nextSectionSize, polygonSize;
            double3 lastInt, nextInt, Int;
            int I = max(-1, min_PX);
            int I_STOP = min(max_PX, pdims.x);
            int numberOfEdges;
            double factor;
            // Section of the square that corresponds to the indices < i
            // CCW and CW coordinates of the last intersection on the lines specified by the
            // points in V_ccw
            lastSectionSize
                = exactIntersectionPoints(((double)I) + 0.5, V_ccw[0], V_ccw[1], V_ccw[2], V_ccw[3],
                                          PX_ccw[0], PX_ccw[1], PX_ccw[2], PX_ccw[3], CM, &lastInt);

            if(I >= 0)
            {
                factor = value * lastSectionSize;
                ADD += exactJacobiValues(CM, lastInt, I, factor, voxelSizes, pdims);
            }
            for(I = I + 1; I < I_STOP; I++)
            {
                nextSectionSize = exactIntersectionPoints(((double)I) + 0.5, V_ccw[0], V_ccw[1],
                                                          V_ccw[2], V_ccw[3], PX_ccw[0], PX_ccw[1],
                                                          PX_ccw[2], PX_ccw[3], CM, &nextInt);
                polygonSize = nextSectionSize - lastSectionSize;
                Int = (nextSectionSize * nextInt - lastSectionSize * lastInt) / polygonSize;
                factor = value * polygonSize;
                ADD += exactJacobiValues(CM, Int, I, factor, voxelSizes, pdims);
                lastSectionSize = nextSectionSize;
                lastInt = nextInt;
            }
            if(I_STOP < pdims.x)
            {
                polygonSize = 1.0 - lastSectionSize;
                Int = ((*V_ccw[0] + *V_ccw[2]) * 0.5 - lastSectionSize * lastInt) / polygonSize;
                factor = value * polygonSize;
                ADD += exactJacobiValues(CM, Int, I, factor, voxelSizes, pdims);
            }
        }
    }
    volumePreconditionerVector[IND] += ADD;
}
