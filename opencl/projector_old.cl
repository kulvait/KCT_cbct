//==============================projector_old.cl=====================================
#ifndef zeroPrecisionTolerance
#define zeroPrecisionTolerance 1e-10
#endif

/// insertEdgeValues(factor, V, P, projection, pdims);
void inline insertEdgeValues(global float* projection,
                             private double16 CM, 
                             private double3 v,
                             private int PX, 
                             private double value,
                             private double3 voxelSizes,
                             private int2 pdims)
{
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
        return;
    }   
    if(PJ_down == PJ_up)
    {   
        AtomicAdd_g_f(&projection[PX * pdims.y + PJ_down],
                      value * voxelSizes.z); // Atomic version of projection[ind] += value;
        return;
    }   
    double stepSize = voxelSizes.z * value
        / (PY_up
           - PY_down); // Length of z in volume to increase y in projection by 1 multiplied by value
    // int j = max(-1, PJ_down);
    // int j_STOP = min(PJ_up, pdims.y);
    int j, j_STOP;
    // Add part that maps to PJ_down
    if(PJ_down >= 0)
    {   
        // double nextGridY;
        // nextGridY = (double)PJ_down + 0.5;
        // factor = (nextGridY - PY_down) * stepSize * value;
        AtomicAdd_g_f(&projection[PX * pdims.y + PJ_down],
                      ((double)PJ_down + 0.5 - PY_down)
                          * stepSize); // Atomic version of projection[ind] += value;
        j = PJ_down + 1;
    } else
    {
        j = 0;
    }
    // Add part that maps to PJ_up
    if(PJ_up < pdims.y)
    {
        // double prevGridY;
        // prevGridY = (double)PJ_up - 0.5;
        // factor = (PY_up - prevGridY) * stepSize * value;
        AtomicAdd_g_f(&projection[PX * pdims.y + PJ_up],
                      (PY_up - ((double)PJ_up - 0.5))
                          * stepSize); // Atomic version of projection[ind] += value;
        j_STOP = PJ_up;
    } else
    {
        j_STOP = pdims.y;
    }
    for(; j < j_STOP; j++)
    {
        AtomicAdd_g_f(&projection[PX * pdims.y + j],
                      stepSize); // Atomic version of projection[ind] += value;
    }
}


/**
 * We parametrize the line segment from A to B by parameter t such that t=0 for A and t=1 for B.
 * Then we will find the t corresponding to the point v = t*B+(1-t)A  that maps to the coordinate PX
 * on the detector. We assume that the mapping is linear and that A maps to PX_A and B maps to PX_B.
 * If A and B maps to the same PX, t=MAXFLOAT
 *
 * @param PX
 * @param PX_A PX index related to A
 * @param PX_B PX index related to B
 *
 * @return Parametrization of the line that maps to PX.
 */
inline double intersectionXTime(double PX, double PX_A, double PX_B)
{
    if(PX_A == PX_B)
    {
        return DBL_MAX;
    } else
    {
        return (PX - PX_A) / (PX_B - PX_A);
    }
}

/**
 * Let v0,v1,v2,v3 and v0,v3,v2,v1 be a piecewise lines that maps on detector on values PX_ccw0,
 * PX_ccw1, PX_ccw2, PX_ccw3. Find a two parametrization factors that maps to a PX on these
 * piecewise lines. We expect that the mappings are linear and nondecreasing up to the certain point
 * from both sides. Parametrization is returned in nextIntersections variable first from
 * v0,v1,v2,v3,v0 lines and next from v0,v3,v2,v1,v0. It expects that PX is between min(*PX_ccw0,
 * *PX_ccw1, *PX_ccw2, *PX_ccw3) and max(*PX_ccw0, *PX_ccw1, *PX_ccw2, *PX_ccw3).
 *
 * @param CM Projection camera matrix.
 * @param PX Mapping of projector
 * @param v0 Point v0
 * @param v1 Point v1
 * @param v2 Point v2
 * @param v3 Point v3
 * @param PX_ccw0 Mapping of v0
 * @param PX_ccw1 Mapping of v1
 * @param PX_ccw2 Mapping of v2
 * @param PX_ccw3 Mapping of v3
 * @param nextIntersections Output tuple of parametrizations that maps to PX.
 * @param v_ccw Output first coordinate that maps to PX on the parametrized piecewise line
 * v0,v1,v2,v3.
 * @param v_cw Output first coordinate that maps to PX on the parametrized piecewise line
 * v0,v3,v2,v1.
 */
inline double findIntersectionPoints(const double PX,
                                     const double3* v0,
                                     const double3* v1,
                                     const double3* v2,
                                     const double3* v3,
                                     const double* PX_ccw0,
                                     const double* PX_ccw1,
                                     const double* PX_ccw2,
                                     const double* PX_ccw3,
                                     double3* centroid)
{
    double p, q, tmp, totalweight;
    double3 v_cw, v_ccw, shift;
    if(PX <= (*PX_ccw1))
    {
        p = intersectionXTime(PX, *PX_ccw0, *PX_ccw1);
        v_ccw = (*v0) * (1.0 - p) + (*v1) * p;
        if(PX <= (*PX_ccw3))
        {
            q = intersectionXTime(PX, *PX_ccw0, *PX_ccw3);
            v_cw = (*v0) * (1.0 - q) + (*v3) * q;
            (*centroid) = (v_ccw + v_cw + (*v0)) / 3.0;
            return p * q * 0.5;
        } else if(PX <= (*PX_ccw2))
        {
            q = intersectionXTime(PX, *PX_ccw3, *PX_ccw2);
            v_cw = (*v3) * (1.0 - p) + (*v2) * p;
            tmp = (q - p) * 0.5;
            totalweight = p + tmp;
            (*centroid)
                = (p * ((v_ccw + (*v3)) / 2.0) + tmp * (v_ccw + v_cw + v_ccw + (*v3) - (*v0)) / 3.0)
                / totalweight;
            return totalweight;
        } else
        {
            q = intersectionXTime(PX, *PX_ccw2, *PX_ccw1);
            v_cw = (*v2) * (1.0 - p) + (*v1) * p;
            tmp = (1.0 - p) * (1.0 - q) * 0.5;
            totalweight = 1 - tmp;
            (*centroid) = (((*v0) + (*v2)) / 2 - tmp * (v_ccw + v_cw + (*v1)) / 3.0) / totalweight;
            return totalweight;
        }
    } else if(PX <= (*PX_ccw2))
    {
        p = intersectionXTime(PX, *PX_ccw1, *PX_ccw2);
        v_ccw = (*v1) * (1.0 - p) + (*v2) * p;
        if(PX <= (*PX_ccw3))
        {
            q = intersectionXTime(PX, *PX_ccw0, *PX_ccw3);
            v_cw = (*v0) * (1.0 - p) + (*v3) * p;
            tmp = (q - p) * 0.5;
            totalweight = p + tmp;
            (*centroid)
                = (p * ((v_ccw + (*v0)) / 2.0) + tmp * (v_ccw + v_cw + v_ccw + (*v0) - (*v1)) / 3.0)
                / totalweight;
            return totalweight;
        } else
        {
            q = intersectionXTime(PX, *PX_ccw3, *PX_ccw2);
            v_cw = (*v3) * (1.0 - p) + (*v2) * p;
            tmp = (1.0 - p) * (1.0 - q) * 0.5;
            totalweight = 1.0 - tmp;
            (*centroid) = (((*v0) + (*v2)) / 2 - tmp * (v_ccw + v_cw + (*v2)) / 3.0) / totalweight;
            return totalweight;
        }
    } else
    {
        p = intersectionXTime(PX, *PX_ccw2, *PX_ccw3);
        v_ccw = (*v2) * (1.0 - p) + (*v3) * p;
        q = intersectionXTime(PX, *PX_ccw0, *PX_ccw3);
        v_cw = (*v0) * (1.0 - p) + (*v3) * p;
        tmp = (1.0 - p) * (1.0 - q) * 0.5;
        totalweight = 1.0 - tmp;
        (*centroid) = (((*v1) + (*v3)) / 2 - tmp * (v_ccw + v_cw + (*v3)) / 3.0) / totalweight;
        return totalweight;
    }
}

/**
 * Compute point parametrized by p on piecewise line segments V_ccw0 ... V_ccw1 ... V_ccw2 ...
 * V_ccw3
 *
 * @param p
 * @param V_ccw0
 * @param V_ccw1
 * @param V_ccw2
 * @param V_ccw3
 */
double3
intersectionPoint(double p, double3* V_ccw0, double3* V_ccw1, double3* V_ccw2, double3* V_ccw3)
{
    double3 v;
    if(p <= 1.0)
    {
        v = (*V_ccw0) * (1.0 - p) + p * (*V_ccw1);
    } else if(p <= 2.0)
    {
        p -= 1.0;
        v = (*V_ccw1) * (1.0 - p) + p * (*V_ccw2);
    } else if(p <= 3.0)
    {
        p -= 2.0;
        v = (*V_ccw2) * (1.0 - p) + p * (*V_ccw3);
    }
    return v;
}

/**
 * From parameters dox.x and dox.y compute the size of the square that is parametrized by these
 * parameters starting from single vertex.
 *
 * @param abc
 *
 * @return
 */
double computeSquareSize(double2 abc)
{
    if(abc.x > abc.y)
    {
        double tmp = abc.x;
        abc.x = abc.y;
        abc.y = tmp;
    }
    // abc.x<=abc.y
    if(abc.x <= 1.0)
    {
        if(abc.y <= 1.0)
        {
            return abc.x * abc.y / 2.0; // Triangle that is bounded by corners (0, a, b)
        } else if(abc.y <= 2.0)
        {
            abc.y = abc.y - 1.0; // Upper edge length
            return abc.x + (abc.y - abc.x) / 2.0;
        } else if(abc.y <= 3.0)
        {
            abc.x = 1.0 - abc.x;
            abc.y = 3.0 - abc.y;
            return 1.0 - (abc.x * abc.y) / 2.0; // Whole square but the area of
                                                // the triangle bounded by // corners (a, 1, b)
        }
    } else if(abc.x <= 2.0)
    {
        // abc.y<=2
        abc.x = 2.0 - abc.x;
        abc.y = 2.0 - abc.y;
        return 1.0 - (abc.x * abc.y) / 2.0; // Whole square but the area of the
                                            // triangle bounded by (a, 2, b)
    }
    return 0; // This is not gonna happen
}

inline uint voxelIndex(uint i, uint j, uint k, int4 vdims)
{
    return i + j * vdims.x + k * vdims.x * vdims.y;
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
void kernel OLD_FLOATcutting_voxel_project(global const float* restrict volume,
                                           global float* restrict projection,
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
    uint i = get_global_id(2);
    uint j = get_global_id(1);
    uint k = get_global_id(0); // This is more effective from the perspective of atomic colisions
    const double3 IND_ijk = { (double)(i), (double)(j), (double)(k) };
    const double3 zerocorner_xyz = { volumeCenter.x - 0.5 * (double)vdims.x * voxelSizes.x,
                                     volumeCenter.y - 0.5 * (double)vdims.y * voxelSizes.y,
                                     volumeCenter.z - 0.5 * (double)vdims.z * voxelSizes.z };
    // If all the corners of given voxel points to a common coordinate, then compute the value based
    // on the center
    int8 cube_abi
        = { projectionIndex(CM, zerocorner_xyz + voxelSizes * IND_ijk, pdims),
            projectionIndex(CM, zerocorner_xyz + voxelSizes * (IND_ijk + (double3)(1.0, 0.0, 0.0)),
                            pdims),
            projectionIndex(CM, zerocorner_xyz + voxelSizes * (IND_ijk + (double3)(0.0, 1.0, 0.0)),
                            pdims),
            projectionIndex(CM, zerocorner_xyz + voxelSizes * (IND_ijk + (double3)(1.0, 1.0, 0.0)),
                            pdims),
            projectionIndex(CM, zerocorner_xyz + voxelSizes * (IND_ijk + (double3)(0.0, 0.0, 1.0)),
                            pdims),
            projectionIndex(CM, zerocorner_xyz + voxelSizes * (IND_ijk + (double3)(1.0, 0.0, 1.0)),
                            pdims),
            projectionIndex(CM, zerocorner_xyz + voxelSizes * (IND_ijk + (double3)(0.0, 1.0, 1.0)),
                            pdims),
            projectionIndex(CM, zerocorner_xyz + voxelSizes * (IND_ijk + (double3)(1.0, 1.0, 1.0)),
                            pdims) };
    if(all(cube_abi
           == -1)) // When all projections of the voxel corners points outside projector area
    {
        return;
    }
    const uint IND = voxelIndex(i, j, k, vdims);
    float voxelValue = volume[IND];
    const double3 voxelcenter_xyz = zerocorner_xyz
        + ((IND_ijk + 0.5) * voxelSizes); // Using widening and vector multiplication operations

    double3 sourceToVoxel_xyz = voxelcenter_xyz - sourcePosition;
    double sourceToVoxel_xyz_norm = length(sourceToVoxel_xyz);
    double cosine = dot(normalToDetector, sourceToVoxel_xyz) / sourceToVoxel_xyz_norm;
    double cosPowThree = cosine * cosine * cosine;
    float value = voxelValue * scalingFactor
        / (sourceToVoxel_xyz_norm * sourceToVoxel_xyz_norm * cosPowThree);
    if(all(cube_abi == cube_abi.x)) // When all projections are the same
    {
        AtomicAdd_g_f(&projection[projectionOffset + cube_abi.x],
                      value); // Atomic version of projection[ind] += value;
        return;
    }
    // I assume that the volume point (x,y,z_1) projects to the same px as (x,y,z_2) for any z_1,
    // z_2  This assumption is restricted to the voxel edges, where it holds very accurately  We
    // project the rectangle that lies on the z midline of the voxel on the projector
    double px00, px01, px10, px11;
    double3 vx00, vx01, vx10, vx11;
    vx00 = zerocorner_xyz + voxelSizes * (IND_ijk + (double3)(0.0, 0.0, 0.5));
    vx01 = zerocorner_xyz + voxelSizes * (IND_ijk + (double3)(1.0, 0.0, 0.5));
    vx10 = zerocorner_xyz + voxelSizes * (IND_ijk + (double3)(0.0, 1.0, 0.5));
    vx11 = zerocorner_xyz + voxelSizes * (IND_ijk + (double3)(1.0, 1.0, 0.5));
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
        insertEdgeValues(&projection[projectionOffset], CM, (vx00 + vx11) / 2, min_PX, value,
                         voxelSizes, pdims);
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
        factor = value * lastSectionSize;
        insertEdgeValues(&projection[projectionOffset], CM, lastInt, I, factor, voxelSizes, pdims);
    }
    for(I = I + 1; I < I_STOP; I++)
    {
        nextSectionSize
            = findIntersectionPoints(((double)I) + 0.5, V_ccw[0], V_ccw[1], V_ccw[2], V_ccw[3],
                                     PX_ccw[0], PX_ccw[1], PX_ccw[2], PX_ccw[3], &nextInt);
        polygonSize = nextSectionSize - lastSectionSize;
        Int = (nextSectionSize * nextInt - lastSectionSize * lastInt) / polygonSize;
        double factor = value * polygonSize;
        insertEdgeValues(&projection[projectionOffset], CM, Int, I, factor, voxelSizes, pdims);
        lastSectionSize = nextSectionSize;
        lastInt = nextInt;
    }
    if(I_STOP < pdims.x)
    {
        polygonSize = 1 - lastSectionSize;
        Int = ((*V_ccw[0] + *V_ccw[2]) * 0.5 - lastSectionSize * lastInt) / polygonSize;
        factor = value * polygonSize;
        insertEdgeValues(&projection[projectionOffset], CM, Int, I, factor, voxelSizes, pdims);
    }
}
//==============================END projector_old.cl=====================================
