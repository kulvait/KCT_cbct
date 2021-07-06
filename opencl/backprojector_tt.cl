//==============================backprojector_tt.cl=====================================
/// backprojectEdgeValues(INDEXfactor, V, P, projection, pdims);
float inline backprojectVericalFootprints(global const float* projection,
                                          private double footprintX,
                                          private int PX,
                                          private double PY_inc0,
                                          private double PY_inc1,
                                          private double PY_inc2,
                                          private double PY_inc3,
                                          private int2 pdims)
{
    double footprint = 0.0f;
    double intervalLength;
    int min_PY, max_PY;
    min_PY = convert_int_rtn(PY_inc0 + 0.5);
    max_PY = convert_int_rtn(PY_inc1 + 0.5);
    intervalLength = PY_inc1 - PY_inc0;
    if(max_PY >= 0 && min_PY < pdims.y && intervalLength > 0.0)
    {
        int J = max(0, min_PY);
        int J_STOP = min(max_PY + 1, pdims.y);
        for(; J < J_STOP; J++)
        {
            footprint += projection[PX * pdims.y + J] * footprintX
                * gamma1(fmax(PY_inc0, J - 0.5) - PY_inc0, fmin(PY_inc1, J + 0.5) - PY_inc0,
                         intervalLength);
        }
    }
    min_PY = convert_int_rtn(PY_inc1 + 0.5);
    max_PY = convert_int_rtn(PY_inc2 + 0.5);
    if(max_PY >= 0 && min_PY < pdims.y)
    {
        int J = max(0, min_PY);
        int J_STOP = min(max_PY + 1, pdims.y);
        for(; J < J_STOP; J++)
        {
            footprint += projection[PX * pdims.y + J] * footprintX
                * gamma2(fmax(PY_inc1, J - 0.5) - PY_inc1, fmin(PY_inc2, J + 0.5) - PY_inc1);
        }
    }
    min_PY = convert_int_rtn(PY_inc2 + 0.5);
    max_PY = convert_int_rtn(PY_inc3 + 0.5);
    intervalLength = PY_inc3 - PY_inc2;
    if(max_PY >= 0 && min_PY < pdims.y && intervalLength > 0.0)
    {
        int J = max(0, min_PY);
        int J_STOP = min(max_PY + 1, pdims.y);
        for(; J < J_STOP; J++)
        {
            footprint += projection[PX * pdims.y + J] * footprintX
                * gamma1(PY_inc3 - fmin(PY_inc3, J + 0.5), PY_inc3 - fmax(PY_inc2, J - 0.5),
                         intervalLength);
        }
    }
    return footprint;
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
void kernel FLOATta3_backproject(global float* restrict volume,
                                 global const float* restrict projection,
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
    uint i = get_global_id(0);
    uint j = get_global_id(1);
    uint k = get_global_id(2);
    const double3 IND_ijk = { (double)(i), (double)(j), (double)(k) };
    const double3 zerocorner_xyz = { volumeCenter.x - 0.5 * (double)vdims.x * voxelSizes.x,
                                     volumeCenter.y - 0.5 * (double)vdims.y * voxelSizes.y,
                                     volumeCenter.z - 0.5 * (double)vdims.z * voxelSizes.z };
    const double3 voxelcorner_xyz = zerocorner_xyz
        + (IND_ijk * voxelSizes); // Using widening and vector multiplication operations
    // If all the corners of given voxel points to a common coordinate, then
    // compute the value based on the center
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
        if(cornerProjectionIndex == -1)
        {
            return;
        }
    }
    const uint IND = voxelIndex(i, j, k, vdims);
    const float voxelVolume = voxelSizes.x * voxelSizes.y * voxelSizes.z;
    const double3 voxelcenter_xyz
        = voxelcorner_xyz + voxelSizes * 0.5; // Using widening and vector multiplication operations
    double3 sourceToVoxel_xyz = voxelcenter_xyz - sourcePosition;
    double3 sourceToVoxel_xyz_unit
        = normalize(sourceToVoxel_xyz); // This vector could be rephrased
                                        // as (cos \varphi sqrt(x^2+y^2), sin \varphi sqrt(x^2+y^2),
                                        // sin \theta)
    double xxplusyy = sqrt(sourceToVoxel_xyz_unit.x * sourceToVoxel_xyz_unit.x
                           + sourceToVoxel_xyz_unit.y * sourceToVoxel_xyz_unit.y);
    sourceToVoxel_xyz_unit = fabs(sourceToVoxel_xyz_unit);
    // To avoid nan induced by dividing by zero
    double xpath = xxplusyy * voxelSizes.x * sourceToVoxel_xyz_unit.y;
    double ypath = xxplusyy * voxelSizes.y * sourceToVoxel_xyz_unit.x;
    double path;
    if(xpath < ypath)
    {
        path = xxplusyy * voxelSizes.x / sourceToVoxel_xyz_unit.x;
    } else
    {
        path = xxplusyy * voxelSizes.y / sourceToVoxel_xyz_unit.y;
    }
    const double A3 = path / sqrt(1 - sourceToVoxel_xyz_unit.z * sourceToVoxel_xyz_unit.z);
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
    // pxx_min = fmin(fmin(px00, px01), fmin(px10, px11));
    // pxx_max = fmax(fmax(px00, px01), fmax(px10, px11));
    double3* V_inc[4]; // Point in which minimum is achieved and sorted points up to maximum
    double* PX_inc[4]; // Point in which minimum is achieved and sorted points up to maximum
    double PY_inc[4]; // Point in which minimum is achieved and sorted points up to maximum
    if(px00 < px01)
    {
        if(px00 < px10)
        {
            pxx_min = px00;
            V_inc[0] = &vx00;
            PX_inc[0] = &px00;
            if(px11 >= px01)
            {
                if(px11 >= px10)
                {
                    pxx_max = px11;
                    V_inc[3] = &vx11;
                    PX_inc[3] = &px11;
                    if(px01 < px10)
                    {
                        V_inc[1] = &vx01;
                        V_inc[2] = &vx10;
                        PX_inc[1] = &px01;
                        PX_inc[2] = &px10;
                    } else
                    {
                        V_inc[1] = &vx10;
                        V_inc[2] = &vx01;
                        PX_inc[1] = &px10;
                        PX_inc[2] = &px01;
                    }
                } else
                {
                    V_inc[1] = &vx01;
                    V_inc[2] = &vx11;
                    PX_inc[1] = &px01;
                    PX_inc[2] = &px11;
                    V_inc[3] = &vx10;
                    PX_inc[3] = &px10;
                    pxx_max = px10;
                }
            } else
            {
                pxx_max = px01;
                V_inc[1] = &vx10;
                V_inc[2] = &vx11;
                PX_inc[1] = &px10;
                PX_inc[2] = &px11;
                V_inc[3] = &vx01;
                PX_inc[3] = &px01;
            }
        } else
        {
            if(px10 < px11)
            {
                pxx_min = px10;
                V_inc[0] = &vx10;
                PX_inc[0] = &px10;
                if(px01 < px11)
                {
                    V_inc[1] = &vx00;
                    V_inc[2] = &vx01;
                    PX_inc[1] = &px00;
                    PX_inc[2] = &px01;
                    V_inc[3] = &vx11;
                    PX_inc[3] = &px11;
                    pxx_max = px11;
                } else
                {
                    V_inc[3] = &vx01;
                    PX_inc[3] = &px01;
                    pxx_max = px01;
                    if(px00 < px11)
                    {
                        V_inc[1] = &vx00;
                        V_inc[2] = &vx11;
                        PX_inc[1] = &px00;
                        PX_inc[2] = &px11;
                    } else
                    {
                        V_inc[1] = &vx11;
                        V_inc[2] = &vx00;
                        PX_inc[1] = &px11;
                        PX_inc[2] = &px00;
                    }
                }
            } else
            {
                pxx_min = px11;
                pxx_max = px01;
                V_inc[0] = &vx11;
                V_inc[1] = &vx10;
                V_inc[2] = &vx00;
                V_inc[3] = &vx01;
                PX_inc[0] = &px11;
                PX_inc[1] = &px10;
                PX_inc[2] = &px00;
                PX_inc[3] = &px01;
            }
        }
    } else
    {
        if(px01 < px11)
        {
            pxx_min = px01;
            V_inc[0] = &vx01;
            PX_inc[0] = &px01;
            if(px10 >= px11)
            {
                if(px10 >= px00)
                {
                    V_inc[3] = &vx10;
                    PX_inc[3] = &px10;
                    if(px00 < px11)
                    {
                        V_inc[1] = &vx00;
                        V_inc[2] = &vx11;
                        PX_inc[1] = &px00;
                        PX_inc[2] = &px11;
                    } else
                    {
                        V_inc[1] = &vx11;
                        V_inc[2] = &vx00;
                        PX_inc[1] = &px11;
                        PX_inc[2] = &px00;
                    }
                    pxx_max = px10;
                } else
                {
                    PX_inc[1] = &px11;
                    PX_inc[2] = &px10;
                    V_inc[1] = &vx11;
                    V_inc[2] = &vx10;
                    V_inc[3] = &vx00;
                    PX_inc[3] = &px00;
                    pxx_max = px00;
                }
            } else
            {
                V_inc[1] = &vx00;
                V_inc[2] = &vx10;
                V_inc[3] = &vx11;
                PX_inc[1] = &px00;
                PX_inc[2] = &px10;
                PX_inc[3] = &px11;
                pxx_max = px11;
            }
        } else
        {
            if(px11 < px10)
            {
                pxx_min = px11;
                V_inc[0] = &vx11;
                PX_inc[0] = &px11;
                if(px00 >= px10)
                {
                    V_inc[3] = &vx00;
                    PX_inc[3] = &px00;
                    pxx_max = px00;
                    if(px01 < px10)
                    {
                        V_inc[1] = &vx01;
                        V_inc[2] = &vx10;
                        PX_inc[1] = &px01;
                        PX_inc[2] = &px10;
                    } else
                    {
                        V_inc[1] = &vx10;
                        V_inc[2] = &vx01;
                        PX_inc[1] = &px10;
                        PX_inc[2] = &px01;
                    }
                } else
                {
                    V_inc[1] = &vx01;
                    V_inc[2] = &vx00;
                    V_inc[3] = &vx10;
                    PX_inc[1] = &px01;
                    PX_inc[2] = &px00;
                    PX_inc[3] = &px10;
                    pxx_max = px10;
                }
            } else
            {
                pxx_min = px10;
                pxx_max = px00;
                V_inc[0] = &vx10;
                V_inc[1] = &vx11;
                V_inc[2] = &vx01;
                V_inc[3] = &vx00;
                PX_inc[0] = &px10;
                PX_inc[1] = &px11;
                PX_inc[2] = &px01;
                PX_inc[3] = &px00;
            }
        }
    }
    double3 vb00, vb01, vb10, vb11;
    double3 vt00, vt01, vt10, vt11;
    double pb00, pb01, pb10, pb11, pt00, pt01, pt10, pt11;
    vb00 = voxelcorner_xyz + voxelSizes * (double3)(0.0, 0.0, 0.0);
    vb01 = voxelcorner_xyz + voxelSizes * (double3)(1.0, 0.0, 0.0);
    vb10 = voxelcorner_xyz + voxelSizes * (double3)(0.0, 1.0, 0.0);
    vb11 = voxelcorner_xyz + voxelSizes * (double3)(1.0, 1.0, 0.0);
    vt00 = voxelcorner_xyz + voxelSizes * (double3)(0.0, 0.0, 1.0);
    vt01 = voxelcorner_xyz + voxelSizes * (double3)(1.0, 0.0, 1.0);
    vt10 = voxelcorner_xyz + voxelSizes * (double3)(0.0, 1.0, 1.0);
    vt11 = voxelcorner_xyz + voxelSizes * (double3)(1.0, 1.0, 1.0);
    pb00 = projectY(CM, vb00);
    pt00 = projectY(CM, vt00);
    if(pb00 < pt00)
    {
        pb01 = projectY(CM, vb01);
        pb10 = projectY(CM, vb10);
        pb11 = projectY(CM, vb11);
        pt01 = projectY(CM, vt01);
        pt10 = projectY(CM, vt10);
        pt11 = projectY(CM, vt11);
    } else
    {
        pt00 = projectY(CM, vb00);
        pt01 = projectY(CM, vb01);
        pt10 = projectY(CM, vb10);
        pt11 = projectY(CM, vb11);
        pb00 = projectY(CM, vt00);
        pb01 = projectY(CM, vt01);
        pb10 = projectY(CM, vt10);
        pb11 = projectY(CM, vt11);
    }
    PY_inc[0] = fmin(fmin(pb00, pb01), fmin(pb10, pb11));
    PY_inc[1] = fmax(fmax(pb00, pb01), fmax(pb10, pb11));
    PY_inc[2] = fmin(fmin(pt00, pt01), fmin(pt10, pt11));
    PY_inc[3] = fmax(fmax(pt00, pt01), fmax(pt10, pt11));
    // First part of trapezoid in transversal direction
    float ADD = 0.0f;
    double footprintX, intervalLength;
    min_PX = convert_int_rtn(pxx_min + 0.5);
    max_PX = convert_int_rtn(*PX_inc[1] + 0.5);
    intervalLength = *PX_inc[1] - *PX_inc[0];
    if(max_PX >= 0 && min_PX < pdims.x && intervalLength > 0.0)
    {
        int I = max(0, min_PX);
        int I_STOP = min(max_PX + 1, pdims.x);
        for(; I < I_STOP; I++)
        {
            footprintX = gamma1(fmax(I - 0.5, *PX_inc[0]) - *PX_inc[0],
                                fmin(I + 0.5, *PX_inc[1]) - *PX_inc[0], intervalLength);
            ADD += backprojectVericalFootprints(&projection[projectionOffset], footprintX, I,
                                                PY_inc[0], PY_inc[1], PY_inc[2], PY_inc[3], pdims);
        }
    }
    // Second part of trapezoid in transversal direction
    min_PX = convert_int_rtn(*PX_inc[1] + 0.5);
    max_PX = convert_int_rtn(*PX_inc[2] + 0.5);
    if(max_PX >= 0 && min_PX < pdims.x)
    {
        int I = max(0, min_PX);
        int I_STOP = min(max_PX + 1, pdims.x);
        for(; I < I_STOP; I++)
        {
            footprintX = gamma2(fmax(I - 0.5, *PX_inc[1]), fmin(I + 0.5, *PX_inc[2]));
            ADD += backprojectVericalFootprints(&projection[projectionOffset], footprintX, I,
                                                PY_inc[0], PY_inc[1], PY_inc[2], PY_inc[3], pdims);
        }
    }
    // Third part of trapezoid in transversal direction
    min_PX = convert_int_rtn(*PX_inc[2] + 0.5);
    max_PX = convert_int_rtn(*PX_inc[3] + 0.5);
    intervalLength = *PX_inc[3] - *PX_inc[2];
    if(max_PX >= 0 && min_PX < pdims.x && intervalLength > 0.0)
    {
        int I = max(0, min_PX);
        int I_STOP = min(max_PX + 1, pdims.x);
        for(; I < I_STOP; I++)
        {
            footprintX = gamma1(*PX_inc[3] - fmin(I + 0.5, *PX_inc[3]),
                                *PX_inc[3] - fmax(I - 0.5, *PX_inc[2]), intervalLength);
            ADD += backprojectVericalFootprints(&projection[projectionOffset], footprintX, I,
                                                PY_inc[0], PY_inc[1], PY_inc[2], PY_inc[3], pdims);
        }
    }
    ADD = ADD * scalingFactor * A3;
    volume[IND] += ADD;
}
//==============================END backprojector_tt.cl=====================================
