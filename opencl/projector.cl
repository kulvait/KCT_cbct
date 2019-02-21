/** Atomic float addition.
 *
 * Function from
 * https://streamhpc.com/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved/.
 *
 *
 * @param source Pointer to the memory to perform atomic operation on.
 * @param operand Float to add.
 */
inline void AtomicAdd_g_f(volatile __global float* source, const float operand)
{
    union
    {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union
    {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do
    {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while(atomic_cmpxchg((volatile __global unsigned int*)source, prevVal.intVal, newVal.intVal)
            != prevVal.intVal);
}

/** Projection of a volume point v onto X coordinate on projector.
 * No checks for boundaries.
 *
 * @param CM Projection camera matrix
 * @param v Volume point
 * @param PX_out Output
 */
void projectX(private double16* CM, private double4* v, private double* PX_out)
{
    (*PX_out) = ((*v).x * (*CM)[0] + (*v).y * (*CM)[1] + (*v).z * (*CM)[2] + (*CM)[3])
        / ((*v).x * (*CM)[8] + (*v).y * (*CM)[9] + (*v).z * (*CM)[10] + (*CM)[11]);
}

/** Projection of a volume point v onto Y coordinate on projector.
 * No checks for boundaries.
 *
 * @param CM Projection camera matrix
 * @param v Volume point
 * @param PY_out Output
 */
void projectY(private double16* CM, private double4* v, private double* PY_out)
{
    (*PY_out) = ((*v).x * (*CM)[4] + (*v).y * (*CM)[5] + (*v).z * (*CM)[6] + (*CM)[7])
        / ((*v).x * (*CM)[8] + (*v).y * (*CM)[9] + (*v).z * (*CM)[10] + (*CM)[11]);
}

/** Projection of a volume point v onto P coordinate on projector.
 * No checks for boundaries.
 *
 * @param CM Projection camera matrix
 * @param v Volume point
 * @param P_out Output
 */
void project(private double16* CM, private double4* v, private double2* P_out)
{

    double4 coord;
    coord.x = v->x * (*CM)[0] + v->y * (*CM)[1] + v->z * (*CM)[2] + (*CM)[3];
    coord.y = v->x * (*CM)[4] + v->y * (*CM)[5] + v->z * (*CM)[6] + (*CM)[7];
    coord.z = v->x * (*CM)[8] + v->y * (*CM)[9] + v->z * (*CM)[10] + (*CM)[11];
    P_out->x = coord.x / coord.z;
    P_out->y = coord.y / coord.z;
}

int2 projectionIndices(private double16 P, private double4 vdim, int2 pdims)
{
    double4 coord;
    coord.x = vdim.x * P[0] + vdim.y * P[1] + vdim.z * P[2] + P[3];
    coord.y = vdim.x * P[4] + vdim.y * P[5] + vdim.z * P[6] + P[7];
    coord.z = vdim.x * P[8] + vdim.y * P[9] + vdim.z * P[10] + P[11];
    coord.x /= coord.z;
    coord.y /= coord.z;
    int2 ind;
    ind.x = convert_int_rtn(coord.x + 0.5);
    ind.y = convert_int_rtn(coord.y + 0.5);
    if(ind.x >= 0 && ind.y >= 0 && ind.x < pdims.x && ind.y < pdims.y)
    {
        return ind;
    } else
    {
        return pdims;
    }
}

int projectionIndex(private double16* CM, private double4 v, int2 pdims)
{
    double4 coord;
    coord.x = v.x * (*CM)[0] + v.y * (*CM)[1] + v.z * (*CM)[2] + (*CM)[3];
    coord.y = v.x * (*CM)[4] + v.y * (*CM)[5] + v.z * (*CM)[6] + (*CM)[7];
    coord.z = v.x * (*CM)[8] + v.y * (*CM)[9] + v.z * (*CM)[10] + (*CM)[11];
    coord.x /= coord.z;
    coord.y /= coord.z;
    int2 ind;
    ind.x = convert_int_rtn(coord.x + 0.5);
    ind.y = convert_int_rtn(coord.y + 0.5);
    if(ind.x >= 0 && ind.y >= 0 && ind.x < pdims.x && ind.y < pdims.y)
    {
        return ind.x + pdims.x * ind.y;
    } else
    {
        return pdims.x * pdims.y;
    }
}

/// insertEdgeValues(factor, V, P, projection, pdims);
void insertEdgeValues(int PX,
                      double value,
                      double4 v,
                      private double16 CM,
                      global float* projection,
                      private double4 voxelSizes,
                      private int2 pdims)
{

    double4 v_down, v_up;
    double PY_down, PY_up;
    int PJ_down, PJ_up;
    v_down = v + voxelSizes * (double4)(0, 0, -0.5, 0);
    v_up = v + voxelSizes * (double4)(0, 0, +0.5, 0);
    projectY(&CM, &v_down, &PY_down);
    projectY(&CM, &v_up, &PY_up);
    PJ_down = convert_int_rtn(PY_down + 0.5);
    PJ_up = convert_int_rtn(PY_up + 0.5);
    int increment = 1;
    if(PJ_down == PJ_up)
    {
        if(PJ_down >= 0 && PJ_down < pdims.y)
        {
            AtomicAdd_g_f(&projection[PX + pdims.x * PJ_down],
                          value * voxelSizes.z); // Atomic version of projection[ind] += value;
        }
        return;
    }
    if(PJ_down > PJ_up)
    {
        increment = -1;
    }
    double stepSize = voxelSizes.z
        / (PY_up - PY_down); // Lenght of z in volume to increase y in projection by 1
    double factor;
    if(PJ_up >= 0 && PJ_up < pdims.y && PJ_down >= 0 && PJ_down < pdims.y)
    { // Do not check for every insert

        for(int j = PJ_down + increment; j != PJ_up; j += increment)
        {
            AtomicAdd_g_f(&projection[PX + pdims.x * j],
                          value * stepSize * increment); // Atomic version of projection[ind] += value;
        }

        // Add part that maps to PJ_down
        double nextGridY;
        if(increment > 0)
        {
            nextGridY = (double)PJ_down + 0.5;
        } else
        {
            nextGridY = (double)PJ_down - 0.5;
        }
        factor = (nextGridY - PY_down) * stepSize;
        AtomicAdd_g_f(&projection[PX + pdims.x * PJ_down],
                      value * factor); // Atomic version of projection[ind] += value;
        // Add part that maps to PJ_up
        double prevGridY;
        if(increment > 0)
        {
            prevGridY = (double)PJ_up - 0.5;
        } else
        {
            prevGridY = (double)PJ_up + 0.5;
        }
        factor = (PY_up - prevGridY) * stepSize;
        AtomicAdd_g_f(&projection[PX + pdims.x * PJ_up],
                      value * factor); // Atomic version of projection[ind] += value;

    } else
    {

        for(int j = PJ_down + increment; j != PJ_up; j += increment)
        {

            if(j >= 0 && j < pdims.y)
            {
                AtomicAdd_g_f(&projection[PX + pdims.x * j],
                              value * stepSize * increment); // Atomic version of projection[ind] += value;
            }
        }

        // Add part that maps to PJ_down
        if(PJ_down >= 0 && PJ_down < pdims.y)
        {
            double nextGridY;
            if(increment > 0)
            {
                nextGridY = (double)PJ_down + 0.5;
            } else
            {
                nextGridY = (double)PJ_down - 0.5;
            }
            factor = (nextGridY - PY_down) * stepSize;
            AtomicAdd_g_f(&projection[PX + pdims.x * PJ_down],
                          value * factor); // Atomic version of projection[ind] += value;
        } // Add part that maps to PJ_up
        if(PJ_up >= 0 && PJ_up < pdims.y)
        {
            double prevGridY;
            if(increment > 0)
            {
                prevGridY = (double)PJ_up - 0.5;
            } else
            {
                prevGridY = (double)PJ_up + 0.5;
            }
            factor = (PY_up - prevGridY) * stepSize;
            AtomicAdd_g_f(&projection[PX + pdims.x * PJ_up],
                          value * factor); // Atomic version of projection[ind] += value;
        }
    }
}

/**
 * We parametrize the line segment from A to B by parameter t such that t=0 for A and t=1 for B.
 * Then we will find the t corresponding to the point v = t*B+(1-t)A  that maps to the coordinate PX
 * on the detector. We assume that the mapping is linear. If A and B maps to the same PX, t=MAXFLOAT
 *
 * @param CM
 * @param PX
 * @param A
 * @param B
 *
 * @return Parametrization of the line that maps to PX.
 */
inline double intersectionXTime(private double16* CM, double* PX, double4* A, double4* B)
{
    double PX_A, PX_B;
    projectX(CM, A, &PX_A);
    projectX(CM, B, &PX_B);
    if(PX_A == PX_B)
    {
        return DBL_MAX;
    }
    return ((*PX) - PX_A) / (PX_B - PX_A);
}

/** Find the position parametrization as double2 in the range [0,4) and [0,4) of the piecewise lines
 * V_ccw0 ... V_ccw1 ... V_ccw2 ... V_ccw3 and V_ccw0 ... V_ccw3 ... V_ccw2 ... V_ccw1 that maps to
 * a start of PI coordinate, that means to PI-0.5.
 *
 *
 * @param CM Projection camera matrix
 * @param PX PX coordinate to be mapped on projector.
 * @param lastIntersections Start search from these points
 * @param V_ccw0
 * @param V_ccw1
 * @param V_ccw2
 * @param V_ccw3
 * @param V_max Point where maximum X is achieved
 *
 * @return position parametrization as double2
 */
double2 findIntersectionPoints(private double16* CM,
                               int PI,
                               double2 lastIntersections,
                               double4* V_ccw0,
                               double4* V_ccw1,
                               double4* V_ccw2,
                               double4* V_ccw3,
                               double4* V_max)
{
    double PX = (double)PI - 0.5;
    double intersectionTime;
    double2 nextIntersections = ((double2)(-1.0, -1.0));
    while(nextIntersections.x < 0)
    {
        if(lastIntersections.x < 1.0)
        {
            intersectionTime = intersectionXTime(CM, &PX, V_ccw0, V_ccw1);
            if(intersectionTime <= 1.0 && intersectionTime >= 0)
            {
                nextIntersections.x = intersectionTime;
            } else if(V_ccw1 == V_max)
            {
                nextIntersections.x = 1.0;
            } else
            {
                lastIntersections.x = 1.0;
            }
        } else if(lastIntersections.x < 2.0)
        {
            intersectionTime = intersectionXTime(CM, &PX, V_ccw1, V_ccw2);
            if(intersectionTime <= 1.0 && intersectionTime >= 0)
            {
                nextIntersections.x = 1.0 + intersectionTime;
            } else if(V_ccw2 == V_max)
            {
                nextIntersections.x = 2.0;
            } else
            {
                lastIntersections.x = 2.0;
            }

        } else if(lastIntersections.x < 3.0)
        {
            intersectionTime = intersectionXTime(CM, &PX, V_ccw2, V_ccw3);
            if(intersectionTime <= 1.0 && intersectionTime >= 0)
            {
                nextIntersections.x = 2.0 + intersectionTime;
            } else if(V_ccw3 == V_max)
            {
                nextIntersections.x = 3.0;
            } else
            {
                // I should not get here
            }
        }
    }
    while(nextIntersections.y < 0)
    {

        if(lastIntersections.y < 1.0)
        {
            intersectionTime = intersectionXTime(CM, &PX, V_ccw0, V_ccw3);
            if(intersectionTime <= 1.0 && intersectionTime >= 0)
            {
                nextIntersections.y = intersectionTime;
            } else if(V_ccw3 == V_max)
            {
                nextIntersections.y = 1.0;
            } else
            {
                lastIntersections.y = 1.0;
            }
        } else if(lastIntersections.y < 2.0)
        {
            intersectionTime = intersectionXTime(CM, &PX, V_ccw3, V_ccw2);
            if(intersectionTime <= 1.0 && intersectionTime >= 0)
            {
                nextIntersections.y = 1.0 + intersectionTime;
            } else if(V_ccw2 == V_max)
            {
                nextIntersections.y = 2.0;
            } else
            {
                lastIntersections.y = 2.0;
            }

        } else if(lastIntersections.y < 3.0)
        {
            intersectionTime = intersectionXTime(CM, &PX, V_ccw2, V_ccw1);
            if(intersectionTime <= 1.0 && intersectionTime >= 0)
            {
                nextIntersections.y = 2.0 + intersectionTime;
            } else if(V_ccw1 == V_max)
            {
                nextIntersections.y = 3.0;
            } else
            {
                // I should not get here
            }
        }
    }
    return nextIntersections;
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
double4
intersectionPoint(double p, double4* V_ccw0, double4* V_ccw1, double4* V_ccw2, double4* V_ccw3)
{
    double4 v;
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

inline int volIndex(int* i, int* j, int* k, int4* vdims)
{
    return (*i) + (*j) * vdims->x + (*k) * (vdims->x * vdims->y);
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
void kernel FLOATcutting_voxel_project(global float* volume,
                                       global float* projection,
                                       private double16 CM,
                                       private double4 sourcePosition,
                                       private double4 normalToDetector,
                                       private int4 vdims,
                                       private double4 voxelSizes,
                                       private int2 pdims,
                                       private float scalingFactor)
{
    int i = get_global_id(2);
    int j = get_global_id(1);
    int k = get_global_id(0); // This is more effective from the perspective of atomic colisions
    const double4 IND_ijk = { (double)(i), (double)(j), (double)(k), 0.0 };
    const double4 zerocorner_xyz = -convert_double4(vdims) / 2.0;
    // If all the corners of given voxel points to a common coordinate, then compute the value based
    // on the center
    int pdimMax = pdims.x * pdims.y;
    int8 cube_abi
        = { projectionIndex(&CM, zerocorner_xyz + voxelSizes * IND_ijk, pdims),
            projectionIndex(&CM, zerocorner_xyz + voxelSizes * (IND_ijk + (double4)(1, 0, 0, 0)),
                            pdims),
            projectionIndex(&CM, zerocorner_xyz + voxelSizes * (IND_ijk + (double4)(0, 1, 0, 0)),
                            pdims),
            projectionIndex(&CM, zerocorner_xyz + voxelSizes * (IND_ijk + (double4)(1, 1, 0, 0)),
                            pdims),
            projectionIndex(&CM, zerocorner_xyz + voxelSizes * (IND_ijk + (double4)(0, 0, 1, 0)),
                            pdims),
            projectionIndex(&CM, zerocorner_xyz + voxelSizes * (IND_ijk + (double4)(1, 0, 1, 0)),
                            pdims),
            projectionIndex(&CM, zerocorner_xyz + voxelSizes * (IND_ijk + (double4)(0, 1, 1, 0)),
                            pdims),
            projectionIndex(&CM, zerocorner_xyz + voxelSizes * (IND_ijk + (double4)(1, 1, 1, 0)),
                            pdims) };
    if(all(cube_abi
           == pdimMax)) // When all projections of the voxel corners points outside projector area
    {
        return;
    }
    const double4 voxelcenter_xyz = zerocorner_xyz
        + ((IND_ijk + 0.5) * voxelSizes); // Using widening and vector multiplication operations
    float voxelValue = volume[volIndex(&i, &j, &k, &vdims)];

    double4 sourceToVoxel_xyz = voxelcenter_xyz - sourcePosition;
    double sourceToVoxel_xyz_norm = length(sourceToVoxel_xyz);
    double cosine = dot(normalToDetector, sourceToVoxel_xyz) / sourceToVoxel_xyz_norm;
    double cosPowThree = cosine * cosine * cosine;
    float value = voxelValue * scalingFactor
        / (sourceToVoxel_xyz_norm * sourceToVoxel_xyz_norm * cosPowThree);
    if(all(cube_abi == cube_abi.x)) // When all projections are the same
    {
        AtomicAdd_g_f(&projection[cube_abi.x],
                      value); // Atomic version of projection[ind] += value;
        return;
    }
    // I assume that the volume point (x,y,z_1) projects to the same px as (x,y,z_2) for any z_1,
    // z_2  This assumption is restricted to the voxel edges, where it holds very accurately  We
    // project the rectangle that lies on the z midline of the voxel on the projector
    double2 px00, px01, px10, px11;
    double4 vx00, vx01, vx10, vx11;
    vx00 = zerocorner_xyz + voxelSizes * (IND_ijk + (double4)(0, 0, 0.5, 0));
    vx01 = zerocorner_xyz + voxelSizes * (IND_ijk + (double4)(1, 0, 0.5, 0));
    vx10 = zerocorner_xyz + voxelSizes * (IND_ijk + (double4)(0, 1, 0.5, 0));
    vx11 = zerocorner_xyz + voxelSizes * (IND_ijk + (double4)(1, 1, 0.5, 0));
    project(&CM, &vx00, &px00);
    project(&CM, &vx01, &px01);
    project(&CM, &vx10, &px10);
    project(&CM, &vx11, &px11);
    // We now figure out the vertex that projects to minimum and maximum px
    double pxx_min, pxx_max; // Minimum and maximum values of projector x coordinate
    int max_PX, min_PX; // Pixel to which are the voxels with minimum and maximum values projected
    pxx_min = min(min(min(px00.x, px01.x), px10.x), px11.x);
    pxx_max = max(max(max(px00.x, px01.x), px10.x), px11.x);
    max_PX = convert_int_rtn(pxx_max + 0.5);
    min_PX = convert_int_rtn(pxx_min + 0.5);
    double4 *V_max, *V_ccw[4]; // Point in which maximum is achieved and counter clock wise points
                               // from the minimum voxel
    if(px00.x == pxx_min)
    {
        V_ccw[0] = &vx00;
        V_ccw[1] = &vx01;
        V_ccw[2] = &vx11;
        V_ccw[3] = &vx10;
    } else if(px01.x == pxx_min)
    {
        V_ccw[0] = &vx01;
        V_ccw[1] = &vx11;
        V_ccw[2] = &vx10;
        V_ccw[3] = &vx00;
    } else if(px10.x == pxx_min)
    {
        V_ccw[0] = &vx10;
        V_ccw[1] = &vx00;
        V_ccw[2] = &vx01;
        V_ccw[3] = &vx11;
    } else // its px11
    {
        V_ccw[0] = &vx11;
        V_ccw[1] = &vx10;
        V_ccw[2] = &vx00;
        V_ccw[3] = &vx01;
    }
    if(px00.x == pxx_max)
    {
        V_max = &vx00;
    } else if(px01.x == pxx_max)
    {
        V_max = &vx01;
    } else if(px10.x == pxx_max)
    {
        V_max = &vx10;
    } else // its px11
    {
        V_max = &vx11;
    }
    // Section of the square that corresponds to the indices < i
    double previousSectionsSize = 0.0;
    // CCW and CW coordinates of the last intersection on the lines specified by the points in V_ccw
    double2 lastIntersections = { 0.0, 0.0 };
    for(int I = min_PX; I < max_PX; I++)
    {
        if(I >= -1 && I < pdims.x)
        {
            double2 nextIntersections = findIntersectionPoints(
                &CM, I + 1, lastIntersections, V_ccw[0], V_ccw[1], V_ccw[2], V_ccw[3], V_max);
            double newSectionsSize = computeSquareSize(nextIntersections);
            double cutSize = newSectionsSize - previousSectionsSize;
            if(I >= 0)
            {
                // Number of edges is a number of vertical edges that we cut according to PY
                // coordinate
                int numberOfEdges = 0; // NextIntersections
                if(lastIntersections.x == 0 && lastIntersections.y == 0)
                {
                    numberOfEdges += 1;
                } else
                {
                    numberOfEdges += 2;
                }
                if(nextIntersections.x + nextIntersections.y == 4)
                {
                    numberOfEdges += 1;
                } else
                {
                    numberOfEdges += 2;
                }
                double4 V;
                double factor = value * cutSize / numberOfEdges;
                if(lastIntersections.x == 0 && lastIntersections.y == 0)
                {
                    V = *V_ccw[0];
                    insertEdgeValues(I, factor, V, CM, projection, voxelSizes, pdims);
                } else
                {
                    V = intersectionPoint(lastIntersections.x, V_ccw[0], V_ccw[1], V_ccw[2],
                                          V_ccw[3]);
                    insertEdgeValues(I, factor, V, CM, projection, voxelSizes, pdims);
                    V = intersectionPoint(lastIntersections.y, V_ccw[0], V_ccw[3], V_ccw[2],
                                          V_ccw[1]);
                    insertEdgeValues(I, factor, V, CM, projection, voxelSizes, pdims);
                }
                if(nextIntersections.x + nextIntersections.y == 4)
                {
                    V = *V_max;
                    insertEdgeValues(I, factor, V, CM, projection, voxelSizes, pdims);
                } else
                {
                    V = intersectionPoint(nextIntersections.x, V_ccw[0], V_ccw[1], V_ccw[2],
                                          V_ccw[3]);
                    insertEdgeValues(I, factor, V, CM, projection, voxelSizes, pdims);
                    V = intersectionPoint(nextIntersections.y, V_ccw[0], V_ccw[3], V_ccw[2],
                                          V_ccw[1]);
                    insertEdgeValues(I, factor, V, CM, projection, voxelSizes, pdims);
                }
            }
            previousSectionsSize = newSectionsSize;
            lastIntersections = nextIntersections;
        }
    }
}
