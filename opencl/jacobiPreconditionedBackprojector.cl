//==============================jacobiPreconditionedBackprojector.cl=====================================
/// backprojectEdgeValues(INDEXfactor, V, P, projection, pdims);
float inline backprojectEdgeValues(global float* projection,
                                   private double16 CM,
                                   private double3 v,
                                   private int PX,
                                   private double value,
                                   private double3 voxelSizes,
                                   private int2 pdims)
{
    float ADD = 0.0;
    double3 v_down, v_up;
    double PY_down, PY_up;
    int PJ_down, PJ_up;
    v_down = v + voxelSizes * (double3)(0.0, 0.0, -0.5);
    v_up = v + voxelSizes * (double3)(0.0, 0.0, 0.5);
    PY_down = projectY(CM, v_down);
    PY_up = projectY(CM, v_up);
    PJ_down = convert_int_rtn(PY_down + 0.5);
    PJ_up = convert_int_rtn(PY_up + 0.5);
    if(PJ_down > PJ_up)
    {
        int tmp_i;
        double tmp_d;
        tmp_i = PJ_down;
        PJ_down = PJ_up;
        PJ_up = tmp_i;
        tmp_d = PY_down;
        PY_down = PY_up;
        PY_up = tmp_d;
    }
    if(PJ_up < 0 || PJ_down >= pdims.y)
    {
        return 0.0;
    }
    float factor;
    if(PJ_down == PJ_up)
    {
        factor = value * voxelSizes.z;
        ADD = projection[PX + pdims.x * PJ_down] * factor;
        return ADD;
    }
    float stepSize = voxelSizes.z * value
        / (PY_up - PY_down); // Lenght of z in volume to increase y in projection by 1
    // Add part that maps to PJ_down
    int j, j_STOP;
    if(PJ_down >= 0)
    {
        double nextGridY;
        nextGridY = (double)PJ_down + 0.5;
        factor = (nextGridY - PY_down) * stepSize;
        ADD = projection[PX + pdims.x * PJ_down] * factor;
        j = PJ_down + 1;
    } else
    {
        j = 0;
    }

    if(PJ_up < pdims.y)
    {
        double prevGridY;
        prevGridY = (double)PJ_up - 0.5;
        factor = (PY_up - prevGridY) * stepSize;
        ADD += projection[PX + pdims.x * PJ_up] * factor;
        j_STOP = PJ_up;
    } else
    {
        j_STOP = pdims.y;
    }
    for(; j < j_STOP; j++)
    {
        ADD += projection[PX + pdims.x * j] * stepSize;
    }
    // Add part that maps to PJ_up
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
 *
 * @return
 */
void kernel FLOATjacobiPreconditionedCutting_voxel_backproject(global const float* restrict volume,
                                                               global const float* restrict preconditioner,
                                                               global float* projection,
                                                               private uint projectionOffset,
                                                               private double16 CM,
                                                               private double3 sourcePosition,
                                                               private double3 normalToDetector,
                                                               private int3 vdims,
                                                               private double3 voxelSizes,
                                                               private double3 volumeCenter,
                                                               private int2 pdims,
                                                               private float scalingFactor)
{
    int i = get_global_id(2);
    int j = get_global_id(1);
    int k = get_global_id(0); // This is more effective from the perspective of atomic colisions
    float ADD = 0.0;
    const double3 IND_ijk = { (double)(i), (double)(j), (double)(k) };
    const double3 zerocorner_xyz = { volumeCenter.x - 0.5 * (double)vdims.x * voxelSizes.x,
                                     volumeCenter.y - 0.5 * (double)vdims.y * voxelSizes.y,
                                     volumeCenter.z - 0.5 * (double)vdims.z * voxelSizes.z };
    const double3 voxelcorner_xyz = zerocorner_xyz
        + (IND_ijk * voxelSizes); // Using widening and vector multiplication operations
    // EXPERIMENTAL ... reconstruct inner circle
    /*   const double3 pixcoords = zerocorner_xyz + voxelSizes * (IND_ijk + (double3)(0.5, 0.5,
       0.5)); if(sqrt(pixcoords.x * pixcoords.x + pixcoords.y * pixcoords.y) > 110.0)
       {
           return;
       }*/
    // EXPERIMENTAL ... reconstruct inner circle
    // If all the corners of given voxel points to a common coordinate, then compute the value based
    // on the center
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
            const uint IND = voxelIndex(i, j, k, vdims);
            const double3 voxelcenter_xyz = voxelcorner_xyz
                + voxelSizes * 0.5; // Using widening and vector multiplication operations
            double3 sourceToVoxel_xyz = voxelcenter_xyz - sourcePosition;
            double sourceToVoxel_xyz_norm = length(sourceToVoxel_xyz);
            double cosine = dot(normalToDetector, sourceToVoxel_xyz) / sourceToVoxel_xyz_norm;
            double cosPowThree = cosine * cosine * cosine;
            float value = preconditioner[IND] * scalingFactor
                / (sourceToVoxel_xyz_norm * sourceToVoxel_xyz_norm * cosPowThree);
            ADD = projection[projectionOffset + cornerProjectionIndex] * value * voxelSizes.x
                * voxelSizes.y * voxelSizes.z;
            volume[IND] += ADD;
        }
        return;
    }
    // EXPERIMENTAL ... reconstruct inner circle
    /*   const double3 pixcoords = zerocorner_xyz + voxelSizes * (IND_ijk + (double3)(0.5, 0.5,
       0.5)); if(sqrt(pixcoords.x * pixcoords.x + pixcoords.y * pixcoords.y) > 110.0)
       {
           return;
       }*/
    // EXPERIMENTAL ... reconstruct inner circle
    // If all the corners of given voxel points to a common coordinate, then compute the value based
    // on the center
    const uint IND = voxelIndex(i, j, k, vdims);
    const double3 voxelcenter_xyz
        = voxelcorner_xyz + voxelSizes * 0.5; // Using widening and vector multiplication operations
    double3 sourceToVoxel_xyz = voxelcenter_xyz - sourcePosition;
    double sourceToVoxel_xyz_norm = length(sourceToVoxel_xyz);
    double cosine = dot(normalToDetector, sourceToVoxel_xyz) / sourceToVoxel_xyz_norm;
    double cosPowThree = cosine * cosine * cosine;
    float value = preconditioner[IND] * scalingFactor
        / (sourceToVoxel_xyz_norm * sourceToVoxel_xyz_norm * cosPowThree);
    // I assume that the volume point (x,y,z_1) projects to the same px as (x,y,z_2) for any z_1,
    // z_2  This assumption is restricted to the voxel edges, where it holds very accurately  We
    // project the rectangle that lies on the z midline of the voxel on the projector
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
        min_PX; // Pixel to which are the voxels with minimum and maximum values are projected
    pxx_min = fmin(fmin(px00, px01), fmin(px10, px11));
    pxx_max = fmax(fmax(px00, px01), fmax(px10, px11));
    max_PX = convert_int_rtn(pxx_max + 0.5);
    min_PX = convert_int_rtn(pxx_min + 0.5);
    if(max_PX < 0 || min_PX >= pdims.x)
    {
        return;
    }
    if(max_PX == min_PX) // Due to the previous statement I know that these indices are inside the
                         // admissible range
    {
        ADD = backprojectEdgeValues(&projection[projectionOffset], CM, (vx10 + vx01) / 2.0, min_PX,
                                    value * voxelSizes.x * voxelSizes.y, voxelSizes, pdims);
        volume[IND] += ADD;
        return;
    }

    double3 *V_max, *V_ccw[4]; // Point in which maximum is achieved and counter clock wise points
    // from the minimum voxel
    double *PX_max,
        *PX_ccw[4]; // Point in which maximum is achieved and counter clock wise  points
    // from the minimum voxel
    if(px00 == pxx_min)
    {
        V_ccw[0] = &vx00;
        V_ccw[1] = &vx01;
        V_ccw[2] = &vx11;
        V_ccw[3] = &vx10;
        PX_ccw[0] = &px00;
        PX_ccw[1] = &px01;
        PX_ccw[2] = &px11;
        PX_ccw[3] = &px10;
    } else if(px01 == pxx_min)
    {
        V_ccw[0] = &vx01;
        V_ccw[1] = &vx11;
        V_ccw[2] = &vx10;
        V_ccw[3] = &vx00;
        PX_ccw[0] = &px01;
        PX_ccw[1] = &px11;
        PX_ccw[2] = &px10;
        PX_ccw[3] = &px00;
    } else if(px10 == pxx_min)
    {
        V_ccw[0] = &vx10;
        V_ccw[1] = &vx00;
        V_ccw[2] = &vx01;
        V_ccw[3] = &vx11;
        PX_ccw[0] = &px10;
        PX_ccw[1] = &px00;
        PX_ccw[2] = &px01;
        PX_ccw[3] = &px11;
    } else // its px11
    {
        V_ccw[0] = &vx11;
        V_ccw[1] = &vx10;
        V_ccw[2] = &vx00;
        V_ccw[3] = &vx01;
        PX_ccw[0] = &px11;
        PX_ccw[1] = &px10;
        PX_ccw[2] = &px00;
        PX_ccw[3] = &px01;
    }
    if(px10 == pxx_max)
    {
        V_max = &vx10;
        PX_max = &px10;
    } else if(px11 == pxx_max)
    {
        V_max = &vx11;
        PX_max = &px11;
    } else if(px00 == pxx_max)
    {
        V_max = &vx00;
        PX_max = &px00;
    } else // its px01
    {
        V_max = &vx01;
        PX_max = &px01;
    }
    double lastSectionSize, nextSectionSize, polygonSize;
    double3 lastInt, nextInt, Int;
    int I = max(-1, min_PX);
    int I_STOP = min(max_PX, pdims.x);
    int numberOfEdges;
    double factor;
    // Section of the square that corresponds to the indices < i
    // CCW and CW coordinates of the last intersection on the lines specified by the points in
    // V_ccw
    lastSectionSize
        = findIntersectionPoints(((double)I) + 0.5, V_ccw[0], V_ccw[1], V_ccw[2], V_ccw[3],
                                 PX_ccw[0], PX_ccw[1], PX_ccw[2], PX_ccw[3], &lastInt);
    if(I >= 0)
    {
        factor = value * lastSectionSize * voxelSizes.x * voxelSizes.y;
        ADD += backprojectEdgeValues(&projection[projectionOffset], CM, lastInt, I, factor,
                                     voxelSizes, pdims);
    }
    for(I = I + 1; I < I_STOP; I++)
    {
        nextSectionSize
            = findIntersectionPoints(((double)I) + 0.5, V_ccw[0], V_ccw[1], V_ccw[2], V_ccw[3],
                                     PX_ccw[0], PX_ccw[1], PX_ccw[2], PX_ccw[3], &nextInt);
        polygonSize = nextSectionSize - lastSectionSize;
        Int = (nextSectionSize * nextInt - lastSectionSize * lastInt) / polygonSize;
        factor = value * polygonSize * voxelSizes.x * voxelSizes.y;
        ADD += backprojectEdgeValues(&projection[projectionOffset], CM, Int, I, factor, voxelSizes,
                                     pdims);
        lastSectionSize = nextSectionSize;
        lastInt = nextInt;
    }
    if(I_STOP < pdims.x)
    {
        polygonSize = 1 - lastSectionSize;
        Int = ((*V_ccw[0] + *V_ccw[2]) * 0.5 - lastSectionSize * lastInt) / polygonSize;
        factor = value * polygonSize * voxelSizes.x * voxelSizes.y;
        ADD += backprojectEdgeValues(&projection[projectionOffset], CM, Int, I, factor, voxelSizes,
                                     pdims);
    }
    volume[IND] += ADD;
}
//==============================END jacobiPreconditionedBackprojector.cl=====================================
