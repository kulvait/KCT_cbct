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

void projectX(private double16* P, private double4* voxelInd, private double* out)
{

    double2 coord;
    coord.x = (*voxelInd).x * (*P)[0] + (*voxelInd).y * (*P)[1] + (*voxelInd).z * (*P)[2] + (*P)[3];
    coord.y
        = (*voxelInd).x * (*P)[8] + (*voxelInd).y * (*P)[9] + (*voxelInd).z * (*P)[10] + (*P)[11];
    (*out) = coord.x / coord.y;
}

typedef struct pointWeightElm
{
    double4 V_xyz;
    float weight;
} PointWeightElm;

// How far from A is the next boundary point on the line |AB|
float nextBoundaryPoint(private double16* P, double4* A, double4* B)
{
    int A_pi, B_pi;
    projectX(P, &A, &A_p);
    projectX(P, &B, &B_p);
    // Find the pixels to which corresponds points A and B
    A_pi = convert_int_rtn(A_p);
    B_pi = convert_int_rtn(B_p);
    int edgeCount = B_pi - A_pi;
    float edgeLength = length(A - B);
    float stepLength = edgeLength / (B_p - A_p);
    float boundary = float(A_pi) + float(0.5);
    float Aptoedge = boundary - A_p;
    float k = stepLength * Aptoedge;
    if(k == 0)
    {
        return stepLength;
    } else
    {
        return k;
    }
}

void project(private double16* P, private double4 voxelInd, private double2* out)
{

    double4 coord;
    coord.x = voxelInd.x * (*P)[0] + voxelInd.y * (*P)[1] + voxelInd.z * (*P)[2] + (*P)[3];
    coord.y = voxelInd.x * (*P)[4] + voxelInd.y * (*P)[5] + voxelInd.z * (*P)[6] + (*P)[7];
    coord.z = voxelInd.x * (*P)[8] + voxelInd.y * (*P)[9] + voxelInd.z * (*P)[10] + (*P)[11];
    (*out).x = coord.x / coord.z;
    (*out).y = coord.y / coord.z;
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
    ind.x = convert_int_rtn(coord.x);
    ind.y = convert_int_rtn(coord.y);
    if(ind.x >= 0 && ind.y >= 0 && ind.x < pdims.x && ind.y < pdims.y)
    {
        return ind;
    } else
    {
        return pdims;
    }
}

int projectionIndex(private double16 P, private double4 vdim, int2 pdims)
{
    double4 coord;
    coord.x = vdim.x * P[0] + vdim.y * P[1] + vdim.z * P[2] + P[3];
    coord.y = vdim.x * P[4] + vdim.y * P[5] + vdim.z * P[6] + P[7];
    coord.z = vdim.x * P[8] + vdim.y * P[9] + vdim.z * P[10] + P[11];
    coord.x /= coord.z;
    coord.y /= coord.z;
    int2 ind;
    ind.x = convert_int_rtn(coord.x);
    ind.y = convert_int_rtn(coord.y);
    if(ind.x >= 0 && ind.y >= 0 && ind.x < pdims.x && ind.y < pdims.y)
    {
        return ind.x + pdims.x * ind.y;
    } else
    {
        return pdims.x * pdims.y;
    }
}

/// insertEdgeValues(factor, V, P, projection, pdims);
void insert_edge_values(int PX,
                        double value,
                        double4 v,
                        private double16 PM,
                        global float* projection,
                        private double4 voxelSizes,
                        private int2 pdims)
{
    double2 p_down, p_up;
    project(&PM, V + voxelSizes * (double4)(0, 0, -0.5, 0), &p_down);
    project(&PM, V + voxelSizes * (double4)(0, 0, +0.5, 0), &p_up);
    int py_down, py_up;
    py_down = convert_int_rtn(p_down.y);
    py_up = convert_int_rtn(p_up.y);
    int increment = 1;
    if(py_down == py_up)
    {
        if(py_down.y >= 0 && py_down.y < pdims.y)
        {
            AtomicAdd_g_f(&projection[PX + pdims.x * p_down.y],
                          value*voxelSizes.Z); // Atomic version of projection[ind] += value;
        }
        return;
    }
    if(py_down > py_up)
    {
        increment = -1;
    }
    double stepSize = voxelSizes.Z
        / (p_up.y - p_down.y); // Lenght of z in volume to increase y in projection by 1
    double previousDistance = 0.0;
    double nextDistance;
    for(int j = py_down; j != py_up; j+=increment)
    {
        if(j >= 0 && j < pdims.y)
        {
            double factor = 1.0;
            if(j == py_down)
            {
                double nextGridY;
                if(increment > 0)
                {
                    nextGridY = double(j) + 0.5;
                } else
                {
                    nextGridY = double(j) - 0.5;
                }
                factor = (nextGridY - p_down.y) * stepSize;
            } else if(j < max_PY - 1)
            {
                factor = 1.0 / totalY;
            } else
            {
                double prevGridY;
                if(increment > 0)
                {
                    prevGridY = double(j) - 0.5;
                } else
                {
                    prevGridY = double(j) + 0.5;
                }
                factor = (p_up.y - prevGridY) * stepSize;
            }
            AtomicAdd_g_f(&projection[PX + pdims.x * j],
                          value*voxelSizes.Z); // Atomic version of projection[ind] += value;
        }
    }
}

/** Project given volume using cutting voxel projector.
 *
 *
 * @param volume Volume to project.
 * @param projection Projection to construct.
 * @param PM Projection matrix. This projection matrix is constructed in the way that (i,j,k) =
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
void kernel FLOATcutting_voxel_project(read_only const image3d_t volume,
                                       global float* projection,
                                       private double16 PM,
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
        = { projectIndex(P, zerocorner_xyz + voxelSizes * IND_ijk, pdims),
            projectIndex(P, zerocorner_xyz + voxelSizes * (IND_ijk + (double4)(1, 0, 0, 0)), pdims),
            projectIndex(P, zerocorner_xyz + voxelSizes * (IND_ijk + (double4)(0, 1, 0, 0)), pdims),
            projectIndex(P, zerocorner_xyz + voxelSizes * (IND_ijk + (double4)(1, 1, 0, 0)), pdims),
            projectIndex(P, zerocorner_xyz + voxelSizes * (IND_ijk + (double4)(0, 0, 1, 0)), pdims),
            projectIndex(P, zerocorner_xyz + voxelSizes * (IND_ijk + (double4)(1, 0, 1, 0)), pdims),
            projectIndex(P, zerocorner_xyz + voxelSizes * (IND_ijk + (double4)(0, 1, 1, 0)), pdims),
            projectIndex(P, zerocorner_xyz + voxelSizes * (IND_ijk + (double4)(1, 1, 1, 0)),
                         pdims) };
    if(cube_abi == cube_abi.x
       && cube_abi.x
           == pdimMax) // When all projections of the voxel corners points outside projector area
    {
        return;
    }
    const double4 voxelcenter_xyz = zerocorner_xyz
        + ((IND_ijk + 0.5) * voxelSizes); // Using widening and vector multiplication operations
    float4 voxelValue = read_imagef(volume, (int4)(i, j, k, 0));

    double4 sourceToVoxel_xyz = voxelcenter_xyz - sourcePosition;
    double sourceToVoxel_xyz_norm = length(sourceToVoxel_xyz);
    double cosine = dot(normalToDetector, sourceToVoxel_xyz) / sourceToVoxel_xyz_norm;
    double cosPowThree = cosine * cosine * cosine;
    float value = voxelValue.x * scalingFactor
        / (sourceToVoxel_xyz_norm * sourceToVoxel_xyz_norm * cosPowThree);
    if(cube_abi == cube_abi.x) // When all projections are the same
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
    project(&P, vx00, &px00);
    project(&P, vx01, &px01);
    project(&P, vx10, &px10);
    project(&P, vx11, &px11);
    // We now figure out the vertex that projects to minimum and maximum px
    double pxx_min, pxx_max; // Minimum and maximum values of projector x coordinate
    int max_PX, min_PX; // Pixel to which are the voxels with minimum and maximum values projected
    pxx_min = min(min(min(px00.x, px01.x), px10.x), px11.x);
    pxx_max = max(max(max(px00.x, px01.x), px10.x), px11.x);
    max_PX = convert_int_rtn(pxx_max);
    min_PX = convert_int_rtn(pxx_min);
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
    for(int i = min_PX; i < max_PX; i++)
    {
        if(i >= 0 && i < pdims.x)
        {
            double2 nextIntersections = findIntersectionPoints(i + 1, lastIntersections, V_ccw[0],
                                                               V_ccw[1], V_ccw[2], V_ccw[3], V_max);
            double newSectionsSize = computeSquareSize(nextIntersections);
            double cutSize = newSectionsSize - previousSectionsSize;
            // Number of edges is a number of vertical edges that we cut according to PY coordinate
            int numberOfEdges = 0; // NextIntersections
            if(lastIntersections.x == 0 && lastIntersections.y == 0)
            {
                numberOfEdges += 1;
            } else
            {
                numberOfEdges += 2;
            }
            if(nextIntersections.x == nextIntersections.y)
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
                insertEdgeValues(i, factor, V, P, projection, voxelSizes, pdims);
            } else
            {
                V = intersectionPoint(lastIntersections.x, V_ccw[0], V_ccw[1], V_ccw[2], V_ccw[3]);
                insertEdgeValues(i, factor, V, P, projection, voxelSizes, pdims);
                V = intersectionPoint(lastIntersections.y, V_ccw[0], V_ccw[3], V_ccw[2], V_ccw[1]);
                insertEdgeValues(i, factor, V, P, projection, voxelSizes, pdims);
            }
            if(nextIntersections.x == nextIntersections.y)
            {
                V = *V_max;
                insertEdgeValues(i, factor, V, P, projection, voxelSizes, pdims);
            } else
            {
                V = intersectionPoint(nextIntersections.x, V_ccw[0], V_ccw[1], V_ccw[2], V_ccw[3]);
                insertEdgeValues(i, factor, V, P, projection, voxelSizes, pdims);
                V = intersectionPoint(nextIntersections.y, V_ccw[0], V_ccw[3], V_ccw[2], V_ccw[1]);
                insertEdgeValues(i, factor, V, P, projection, voxelSizes, pdims);
            }

            // numberOfEdges
            //    += convert_int_rtn(floor(nextIntersections.x) - floor(lastIntersections.x));
            // numberOfEdges
            //    += convert_int_rtn(floor(nextIntersections.y) - floor(lastIntersections.y));
            // Now climb over edges CCW

            double intersectionk = lastIntersections.x;
            double4 V;
            double2 P_MIN;
            double2 P_MAX;
            intersectionk = lastIntersections.y;
            while(intersectionk <= nextIntersections.y)
            {
                int arrayIndex = convert_int_rtn(floor(intersectionk));
                double k = intersectionk - floor(intersectionk);
                V = (*V_ccw[(-arrayIndex) % 4]) * (1 - k) + k * (*V_ccw[(-arrayIndex - 1) % 4]);
                project(&P, V, &P_MIN);
                project(&P, V + voxelSizes * (double4)(0, 0, 1, 0), &P_MAX);
                // Now assign values to the sections of the line that projects between P_MIN and
                // P_MAX

                int max_PY, min_PY;
                min_PY = convert_int_rtn(P_MIN.y);
                max_PY = convert_int_rtn(P_MAX.y);
                double totalY = P_MAX.y - P_MIN.y;
                double intersection = (nextGridY - P_MIN.y) / totalY;

                for(int j = min_PY; j != max_PY; j++)
                {
                    if(j >= 0 && j < pdims.y)
                    {
                        double factor = 1.0;
                        if(j == min_PY)
                        {
                            double nextGridY;
                            if(totalY > 0)
                            {
                                nextGridY = ceil(P_MIN.y) + 0.5;
                            } else
                            {
                                nextGridY = ceil(P_MIN.y) - 0.5;
                            }
                            factor = (nextGridY - P_MIN.y) / totalY;
                        } else if(j < max_PY - 1)
                        {
                            factor = 1.0 / totalY;
                        } else
                        {
                            double nextGridY;
                            if(totalY > 0)
                            {
                                nextGridY = ceil(P_MAX.y) - 0.5;
                            } else
                            {
                                nextGridY = ceil(P_MAX.y) + 0.5;
                            }
                            factor = (P_MAX.y - nextGridY) / totalY;
                        }

                        factor = factor * cutSize / numberOfEdges;

                        AtomicAdd_g_f(&projection[i + pdims.x * j],
                                      value
                                          * factor); // Atomic version of projection[ind] += value;
                    }
                }
                if(floor(intersectionk) < floor(nextIntersections.y))
                {
                    intersectionk = floor(intersectionk) + 1.0;
                } else if(intersectionk == nextIntersections.y)
                {
                    intersectionk += 1.0;
                } else
                {
                    intersectionk = nextIntersections.y;
                }
            }
        }
    }
}
