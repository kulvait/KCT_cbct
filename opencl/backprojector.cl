//==============================backprojector.cl=====================================
#ifndef zeroPrecisionTolerance
#define zeroPrecisionTolerance 1e-10
#endif

//#ifndef RELAXED
//#define RELAXED
//#endif

#define PINDEX(PX, MAX) (PX >= 0 || PX < MAX ? PX : -1)

#define BACKPROJECTMINMAX(PJ_min, PJ_max, v_min, v_min_minus_v_max_y)                              \
    int J;                                                                                         \
    REAL3 Fvector;                                                                                 \
    REAL lambda;                                                                                   \
    REAL lastLambda = ZERO;                                                                        \
    REAL leastLambda;                                                                              \
    if(PJ_max >= pdims.y)                                                                          \
    {                                                                                              \
        PJ_max = pdims.y - 1;                                                                      \
        Fvector = CM.s456 - (PJ_max + HALF) * CM.s89a;                                             \
        leastLambda = dot(v_min, Fvector) / (v_min_minus_v_max_y * Fvector.s2);                    \
    } else                                                                                         \
    {                                                                                              \
        leastLambda = ONE;                                                                         \
    }                                                                                              \
    if(PJ_min < 0)                                                                                 \
    {                                                                                              \
        J = 0;                                                                                     \
        Fvector = CM.s456 + HALF * CM.s89a;                                                        \
        lastLambda = dot(v_min, Fvector) / (v_min_minus_v_max_y * Fvector.s2);                     \
    } else                                                                                         \
    {                                                                                              \
        J = PJ_min;                                                                                \
        Fvector = CM.s456 - (J - HALF) * CM.s89a;                                                  \
    }                                                                                              \
    for(; J < PJ_max; J++)                                                                         \
    {                                                                                              \
        Fvector -= CM.s89a;                                                                        \
        lambda = dot(v_min, Fvector) / (v_min_minus_v_max_y * Fvector.s2);                         \
        ADD += projection[J] * (lambda - lastLambda);                                              \
        lastLambda = lambda;                                                                       \
    }                                                                                              \
    ADD += projection[PJ_max] * (leastLambda - lastLambda);

#define BACKPROJECTMINMAXF(PJ_min, PJ_max, v_min, v_min_minus_v_max_y)                             \
    int J;                                                                                         \
    float3 Fvector;                                                                                \
    float lambda;                                                                                  \
    float lastLambda = 0.0f;                                                                       \
    float leastLambda;                                                                             \
    if(PJ_max >= pdims.y)                                                                          \
    {                                                                                              \
        PJ_max = pdims.y - 1;                                                                      \
        Fvector = CM.s456 - (PJ_max + 0.5f) * CM.s89a;                                             \
        leastLambda = dot(v_min, Fvector) / (v_min_minus_v_max_y * Fvector.s2);                    \
    } else                                                                                         \
    {                                                                                              \
        leastLambda = 1.0f;                                                                        \
    }                                                                                              \
    if(PJ_min < 0)                                                                                 \
    {                                                                                              \
        J = 0;                                                                                     \
        Fvector = CM.s456 + 0.5f * CM.s89a;                                                        \
        lastLambda = dot(v_min, Fvector) / (v_min_minus_v_max_y * Fvector.s2);                     \
    } else                                                                                         \
    {                                                                                              \
        J = PJ_min;                                                                                \
        Fvector = CM.s456 - (J - 0.5f) * CM.s89a;                                                  \
    }                                                                                              \
    for(; J < PJ_max; J++)                                                                         \
    {                                                                                              \
        Fvector -= CM.s89a;                                                                        \
        lambda = dot(v_min, Fvector) / (v_min_minus_v_max_y * Fvector.s2);                         \
        ADD += projection[J] * (lambda - lastLambda);                                              \
        lastLambda = lambda;                                                                       \
    }                                                                                              \
    ADD += projection[PJ_max] * (leastLambda - lastLambda);

/// backprojectEdgeValues(INDEXfactor, V, P, projection, pdims);
float inline backprojectExactEdgeValues0(global const float* projection,
                                         private REAL16 CM,
                                         private REAL3 v,
                                         private int PX,
                                         private REAL3 voxelSizes,
                                         private int2 pdims)
{
    projection = projection + PX * pdims.y;
    const REAL3 distanceToEdge = (REAL3)(ZERO, ZERO, HALF * voxelSizes.s2);
    const REAL3 v_up = v + distanceToEdge;
    const REAL3 v_down = v - distanceToEdge;
    const REAL PY_up = PROJECTY0(CM, v_up);
    const REAL PY_down = PROJECTY0(CM, v_down);
    int PJ_up = convert_int_rtn(PY_up + HALF);
    int PJ_down = convert_int_rtn(PY_down + HALF);
    float ADD = 0.0f;
    if(PJ_up < PJ_down)
    {
        if(PJ_down >= 0 && PJ_up < pdims.y)
        {
            BACKPROJECTMINMAX(PJ_up, PJ_down, v_up, voxelSizes.s2);
        }
    } else if(PJ_down < PJ_up)
    {
        if(PJ_up >= 0 && PJ_down < pdims.y)
        {
            BACKPROJECTMINMAX(PJ_down, PJ_up, v_down, -voxelSizes.s2);
        }

    } else if(PJ_up >= 0 && PJ_up < pdims.y)
    {
        ADD = projection[PJ_up];
    }
    return ADD; // Scaling by value is performed at the end
}

float inline backprojectExactEdgeValuesF0(global const float* projection,
                                          private float16 CM,
                                          private float3 v,
                                          private int PX,
                                          private float3 voxelSizes,
                                          private int2 pdims)
{
    projection = projection + PX * pdims.y;
    const float3 distanceToEdge = (float3)(0.0f, 0.0f, 0.5f * voxelSizes.s2);
    const float3 v_up = v + distanceToEdge;
    const float3 v_down = v - distanceToEdge;
    const float PY_up = PROJECTY0(CM, v_up);
    const float PY_down = PROJECTY0(CM, v_down);
    int PJ_up = convert_int_rtn(PY_up + 0.5);
    int PJ_down = convert_int_rtn(PY_down + 0.5);
    float ADD = 0.0;
    if(PJ_up < PJ_down)
    {
        if(PJ_down >= 0 && PJ_up < pdims.y)
        {
            BACKPROJECTMINMAXF(PJ_up, PJ_down, v_up, voxelSizes.s2);
        }
    } else if(PJ_down < PJ_up)
    {
        if(PJ_up >= 0 && PJ_down < pdims.y)
        {
            BACKPROJECTMINMAXF(PJ_down, PJ_up, v_down, -voxelSizes.s2);
        }

    } else if(PJ_up >= 0 && PJ_up < pdims.y)
    {
        ADD = projection[PJ_up];
    }
    return ADD; // Scaling by value is performed at the end
}

/// backprojectEdgeValues(INDEXfactor, V, P, projection, pdims);
float inline backprojectExactEdgeValues(global const float* projection,
                                        private double16 CM,
                                        private double3 v,
                                        private int PX,
                                        private double value,
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
    float ADD = 0.0;
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
                ADD += projection[PX * pdims.y + J] * (lambda - lastLambda) * value;
                // Atomic version of projection[ind] += value;
                lastLambda = lambda;
            }
            // PJ_max
            ADD += projection[PX * pdims.y + PJ_max] * (leastLambda - lastLambda)
                * value; // Atomic version of projection[ind] += value;
        }
    } else if(PJ_down > PJ_up)
    {
        if(PJ_down >= 0 && PJ_up < pdims.y)
        {
            // We will count with negative value of lambda by dividing by dot(v_diff, Fvector)
            // instead of dot(-v_diff, Fvector)  Because valuePerUnit is negative the value (lambda
            // - lastLambda)*valuePerUnit will be positive
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
                Fvector -= CM.s89a; // Fvector = CM.s456 - (J + 0.5) * CM.s89a;
                lambda = (dot(v_up, Fvector) + CM.s7 - ((double)J + 0.5) * CM.sb)
                    / (negativeEdgeLength * Fvector.s2);
                ADD += projection[PX * pdims.y + J] * (lastLambda - lambda)
                    * value; // Atomic version of projection[ind] += value;
                lastLambda = lambda;
            }
            // PJ_max
            ADD += projection[PX * pdims.y + PJ_max] * (lastLambda - leastLambda)
                * value; // Atomic version of projection[ind] += value;
        }
    } else if(PJ_down == PJ_up && PJ_down >= 0 && PJ_down < pdims.y)
    {
        ADD += projection[PX * pdims.y + PJ_down]
            * value; // Atomic version of projection[ind] += value;
    }
    return ADD;
}

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
        factor = value;
        ADD = projection[PX * pdims.y + PJ_down] * factor;
        return ADD;
    }
    float stepSize
        = value / (PY_up - PY_down); // Lenght of z in volume to increase y in projection by 1
    // Add part that maps to PJ_down
    int j, j_STOP;
    if(PJ_down >= 0)
    {
        double nextGridY;
        nextGridY = (double)PJ_down + 0.5;
        factor = (nextGridY - PY_down) * stepSize;
        ADD = projection[PX * pdims.y + PJ_down] * factor;
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
        ADD += projection[PX * pdims.y + PJ_up] * factor;
        j_STOP = PJ_up;
    } else
    {
        j_STOP = pdims.y;
    }
    for(; j < j_STOP; j++)
    {
        ADD += projection[PX * pdims.y + j] * stepSize;
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
void kernel FLOATcutting_voxel_backproject(global float* restrict volume,
                                           global const float* restrict projection,
                                           private uint projectionOffset,
                                           private double16 _CM,
                                           private double3 _sourcePosition,
                                           private double3 _normalToDetector,
                                           private int3 vdims,
                                           private double3 _voxelSizes,
                                           private double3 _volumeCenter,
                                           private int2 pdims,
                                           private float scalingFactor)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);
    // Not uint for correct int subtraction
    // Shift projection array by offset
    projection += projectionOffset;
//_normalToDetector is not used in this implementation
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
    float ADD = 0.0;

#ifdef DROPCENTEROFFPROJECTORVOXELS
    int xindex = INDEX(PROJECTX0(CM, voxelcenter_xyz));
    int yindex = INDEX(PROJECTY0(CM, voxelcenter_xyz));
    if(xindex < 0 || yindex < 0 || xindex >= pdims.x || yindex >= pdims.y)
    {
        return;
    }
#endif

#ifdef DROPINCOMPLETEVOXELS // Here I need to do more
    int xindex = INDEX(PROJECTX0(CM, voxelcenter_xyz));
    int yindex = INDEX(PROJECTY0(CM, voxelcenter_xyz));
    if(xindex < 0 || yindex < 0 || xindex >= pdims.x || yindex >= pdims.y)
    {
        return;
    }
#endif
    const REAL voxelVolume = voxelSizes.x * voxelSizes.y * voxelSizes.z;
    REAL sourceToVoxel_xyz_norm2 = dot(voxelcenter_xyz, voxelcenter_xyz);
#ifdef RELAXED
    float value = (voxelVolume * scalingFactor / sourceToVoxel_xyz_norm2);
#else
    float value = (float)(voxelVolume * scalingFactor / sourceToVoxel_xyz_norm2);
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
    // printf("X projections are %f, %f, %f, %f", px00, px10, px01, px11);
    // We now figure out the vertex that projects to minimum and maximum px
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
            V_ccw[1] = &vx11;
            V_ccw[2] = &vx10;
            V_ccw[3] = &vx00;
            PX_ccw[0] = &px01;
            PX_ccw[1] = &px11;
            PX_ccw[2] = &px10;
            PX_ccw[3] = &px00;
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
        V_ccw[1] = &vx00;
        V_ccw[2] = &vx01;
        V_ccw[3] = &vx11;
        PX_ccw[0] = &px10;
        PX_ccw[1] = &px00;
        PX_ccw[2] = &px01;
        PX_ccw[3] = &px11;
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
        V_ccw[1] = &vx11;
        V_ccw[2] = &vx10;
        V_ccw[3] = &vx00;
        PX_ccw[0] = &px01;
        PX_ccw[1] = &px11;
        PX_ccw[2] = &px10;
        PX_ccw[3] = &px00;
    }

    min_PX = convert_int_rtn(pxx_min + zeroPrecisionTolerance + HALF);
    max_PX = convert_int_rtn(pxx_max - zeroPrecisionTolerance + HALF);
    if(max_PX >= 0 && min_PX < pdims.x)
    {
        REAL vd1 = V_ccw[1]->x - V_ccw[0]->x;
        REAL vd3 = V_ccw[3]->y - V_ccw[0]->y;
        if(max_PX <= min_PX) // These indices are in the admissible range
        {
            min_PX = convert_int_rtn(HALF * (pxx_min + pxx_max) + HALF);
            ADD = backprojectExactEdgeValues0(projection, CM, HALF * (vx10 + vx01), min_PX,
                                              voxelSizes, pdims);
            volume[IND] += value * ADD;
        } else
        {
            REAL lastSectionSize, nextSectionSize, polygonSize;
            REAL3 lastInt, nextInt, Int;
            REAL factor;
            int I = max(-1, min_PX);
            int I_STOP = min(max_PX, pdims.x);
            // Section of the square that corresponds to the indices < i
            // CCW and CW coordinates of the last intersection on the lines specified by the points
            // in V_ccw lastSectionSize
            //    = findIntersectionPoints(((double)I) + 0.5, V_ccw[0], V_ccw[1], V_ccw[2],
            //    V_ccw[3],
            //                             PX_ccw[0], PX_ccw[1], PX_ccw[2], PX_ccw[3], &lastInt);
            lastSectionSize = exactIntersectionPoints0_extended(
                ((REAL)I) + HALF, V_ccw[0], V_ccw[1], V_ccw[2], V_ccw[3], vd1, vd3, PX_ccw[0],
                PX_ccw[1], PX_ccw[2], PX_ccw[3], CM, &lastInt);
            if(I >= 0)
            {
                factor = backprojectExactEdgeValues0(projection, CM, lastInt, I, voxelSizes, pdims);
                ADD += lastSectionSize * factor;
            }
            for(I = I + 1; I < I_STOP; I++)
            {
                nextSectionSize = exactIntersectionPoints0_extended(
                    ((REAL)I) + HALF, V_ccw[0], V_ccw[1], V_ccw[2], V_ccw[3], vd1, vd3, PX_ccw[0],
                    PX_ccw[1], PX_ccw[2], PX_ccw[3], CM, &nextInt);
                polygonSize = nextSectionSize - lastSectionSize;
                Int = (nextSectionSize * nextInt - lastSectionSize * lastInt) / polygonSize;
                factor = backprojectExactEdgeValues0(projection, CM, Int, I, voxelSizes, pdims);
                ADD += polygonSize * factor;
                lastSectionSize = nextSectionSize;
                lastInt = nextInt;
            }
            if(I_STOP < pdims.x)
            {
                polygonSize = ONE - lastSectionSize;
                Int = ((*V_ccw[0] + *V_ccw[2]) * HALF - lastSectionSize * lastInt) / polygonSize;
                factor = backprojectExactEdgeValues0(projection, CM, Int, I, voxelSizes, pdims);
                ADD += polygonSize * factor;
            }
            volume[IND] += value * ADD;
        }
    }
}
//==============================END backprojector.cl=====================================
