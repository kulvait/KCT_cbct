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
    double3 Fvector;                                                                               \
    double lambda;                                                                                 \
    double lastLambda = 0.0;                                                                       \
    double leastLambda;                                                                            \
    if(PJ_max >= pdims.y)                                                                          \
    {                                                                                              \
        PJ_max = pdims.y - 1;                                                                      \
        Fvector = CM.s456 - (PJ_max + 0.5) * CM.s89a;                                              \
        leastLambda = dot(v_min, Fvector) / (v_min_minus_v_max_y * Fvector.s2);                    \
    } else                                                                                         \
    {                                                                                              \
        leastLambda = 1.0;                                                                         \
    }                                                                                              \
    if(PJ_min < 0)                                                                                 \
    {                                                                                              \
        J = 0;                                                                                     \
        Fvector = CM.s456 + 0.5 * CM.s89a;                                                         \
        lastLambda = dot(v_min, Fvector) / (v_min_minus_v_max_y * Fvector.s2);                     \
    } else                                                                                         \
    {                                                                                              \
        J = PJ_min;                                                                                \
        Fvector = CM.s456 - (J - 0.5) * CM.s89a;                                                   \
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
        leastLambda = 1.0f;                                                                         \
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
                                         private double16 CM,
                                         private double3 v,
                                         private int PX,
                                         private double3 voxelSizes,
                                         private int2 pdims)
{
    projection = projection + PX * pdims.y;
    const double3 distanceToEdge = (double3)(0.0, 0.0, 0.5 * voxelSizes.s2);
    const double3 v_up = v + distanceToEdge;
    const double3 v_down = v - distanceToEdge;
    const double PY_up = PROJECTY0(CM, v_up);
    const double PY_down = PROJECTY0(CM, v_down);
    int PJ_up = convert_int_rtn(PY_up + 0.5);
    int PJ_down = convert_int_rtn(PY_down + 0.5);
    float ADD = 0.0;
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
    uint k = get_global_id(2); // This is more effective from the perspective of atomic colisions
    const uint IND = voxelIndex(i, j, k, vdims);
    float ADD = 0.0;
    const float voxelVolume = voxelSizes.x * voxelSizes.y * voxelSizes.z;

#ifdef RELAXED
    const float16 CMF = convert_float16(CM);
    const float3 sourcePositionF = convert_float3(sourcePosition);
    const float3 voxelSizesF = convert_float3(voxelSizes);
    const float3 volumeCenterF = convert_float3(volumeCenter);
    const float3 IND_ijk = { (float)(i), (float)(j), (float)(k) };
    const float3 zerocorner_xyz
        = volumeCenterF - sourcePositionF - 0.5f * convert_float3(vdims) * voxelSizesF;
    const float3 voxelcorner_xyz = zerocorner_xyz + (IND_ijk * voxelSizesF);
    const float3 voxelcenter_xyz = voxelcorner_xyz + 0.5f * voxelSizesF;
    int cornerProjectionIndex = projectionIndexF0(CMF, voxelcenter_xyz, pdims);
    if(cornerProjectionIndex == -1)
    {
        return;
    }
    float sourceToVoxel_xyz_norm2 = dot(voxelcenter_xyz, voxelcenter_xyz);
    float value = scalingFactor * voxelVolume / sourceToVoxel_xyz_norm2;
    // I assume that the volume point (x,y,z_1) projects to the same px as (x,y,z_2) for any z_1,
    // z_2  This assumption is restricted to the voxel edges, where it holds very accurately  We
    // project the rectangle that lies on the z midline of the voxel on the projector
    float px00, px01, px10, px11;
    float3 vx00, vx01, vx10, vx11;
    vx00 = voxelcorner_xyz + voxelSizesF * (float3)(0.0, 0.0, 0.5);
    vx01 = voxelcorner_xyz + voxelSizesF * (float3)(1.0, 0.0, 0.5);
    vx10 = voxelcorner_xyz + voxelSizesF * (float3)(0.0, 1.0, 0.5);
    vx11 = voxelcorner_xyz + voxelSizesF * (float3)(1.0, 1.0, 0.5);
    px00 = PROJECTX0(CMF, vx00);
    px01 = PROJECTX0(CMF, vx01);
    px10 = PROJECTX0(CMF, vx10);
    px11 = PROJECTX0(CMF, vx11);
    // We now figure out the vertex that projects to minimum and maximum px
    float pxx_min, pxx_max; // Minimum and maximum values of projector x coordinate
    int max_PX,
        min_PX; // Pixel to which are the voxels with minimum and maximum values are
                // projected
    // pxx_min = fmin(fmin(px00, px01), fmin(px10, px11));
    // pxx_max = fmax(fmax(px00, px01), fmax(px10, px11));
    float3* V_ccw[4]; // Point in which minimum is achieved and counter clock wise points
    // from the minimum voxel
    float* PX_ccw[4]; // Point in which minimum is achieved and counter clock wise  points
    // from the minimum voxel
#else
    const double3 IND_ijk = { (double)(i), (double)(j), (double)(k) };
    const double3 zerocorner_xyz
        = volumeCenter - sourcePosition - 0.5 * convert_double3(vdims) * voxelSizes;
    const double3 voxelcorner_xyz = zerocorner_xyz + (IND_ijk * voxelSizes);
    const double3 voxelcenter_xyz = voxelcorner_xyz + 0.5 * voxelSizes;
    int cornerProjectionIndex = projectionIndex0(CM, voxelcorner_xyz, pdims);
    if(cornerProjectionIndex == -1)
    {
        if(projectionIndex0(CM, voxelcorner_xyz + voxelSizes * (double3)(1.0, 1.0, 1.0), pdims)
               == -1
           && projectionIndex0(CM, voxelcorner_xyz + voxelSizes * (double3)(1.0, 1.0, 0.0), pdims)
               == -1
           && projectionIndex0(CM, voxelcorner_xyz + voxelSizes * (double3)(1.0, 0.0, 1.0), pdims)
               == -1
           && projectionIndex0(CM, voxelcorner_xyz + voxelSizes * (double3)(0.0, 1.0, 1.0), pdims)
               == -1
           && projectionIndex0(CM, voxelcorner_xyz + voxelSizes * (double3)(1.0, 0.0, 0.0), pdims)
               == -1
           && projectionIndex0(CM, voxelcorner_xyz + voxelSizes * (double3)(0.0, 1.0, 0.0), pdims)
               == -1
           && projectionIndex0(CM, voxelcorner_xyz + voxelSizes * (double3)(0.0, 0.0, 1.0),
                               pdims)
               == -1) // When all projections are the same
        {
            return;
        }
    }
    /*
        if(cornerProjectionIndex
               == projectionIndex0(CM, voxelcorner_xyz + voxelSizes * (double3)(1.0, 1.0, 1.0),
       pdims)
           && cornerProjectionIndex
               == projectionIndex0(CM, voxelcorner_xyz + voxelSizes * (double3)(1.0, 1.0, 0.0),
       pdims)
           && cornerProjectionIndex
               == projectionIndex0(CM, voxelcorner_xyz + voxelSizes * (double3)(1.0, 0.0, 1.0),
       pdims)
           && cornerProjectionIndex
               == projectionIndex0(CM, voxelcorner_xyz + voxelSizes * (double3)(0.0, 1.0, 1.0),
       pdims)
           && cornerProjectionIndex
               == projectionIndex0(CM, voxelcorner_xyz + voxelSizes * (double3)(1.0, 0.0, 0.0),
       pdims)
           && cornerProjectionIndex
               == projectionIndex0(CM, voxelcorner_xyz + voxelSizes * (double3)(0.0, 1.0, 0.0),
       pdims)
           && cornerProjectionIndex
               == projectionIndex0(CM, voxelcorner_xyz + voxelSizes * (double3)(0.0, 0.0, 1.0),
                                   pdims)) // When all projections are the same
        {
            if(cornerProjectionIndex != -1)
            {
                double sourceToVoxel_xyz_norm2 = dot(voxelcenter_xyz, voxelcenter_xyz);
                ADD = projection[projectionOffset + cornerProjectionIndex] * scalingFactor *
       voxelVolume / sourceToVoxel_xyz_norm2; volume[IND] += ADD;
            }
            return;
        }*/
    double sourceToVoxel_xyz_norm2 = dot(voxelcenter_xyz, voxelcenter_xyz);
    float value = scalingFactor * voxelVolume / sourceToVoxel_xyz_norm2;
    // I assume that the volume point (x,y,z_1) projects to the same px as (x,y,z_2) for any z_1,
    // z_2  This assumption is restricted to the voxel edges, where it holds very accurately  We
    // project the rectangle that lies on the z midline of the voxel on the projector
    double px00, px01, px10, px11;
    double3 vx00, vx01, vx10, vx11;
    vx00 = voxelcorner_xyz + voxelSizes * (double3)(0.0, 0.0, 0.5);
    vx01 = voxelcorner_xyz + voxelSizes * (double3)(1.0, 0.0, 0.5);
    vx10 = voxelcorner_xyz + voxelSizes * (double3)(0.0, 1.0, 0.5);
    vx11 = voxelcorner_xyz + voxelSizes * (double3)(1.0, 1.0, 0.5);
    px00 = PROJECTX0(CM, vx00);
    px01 = PROJECTX0(CM, vx01);
    px10 = PROJECTX0(CM, vx10);
    px11 = PROJECTX0(CM, vx11);
    // We now figure out the vertex that projects to minimum and maximum px
    double pxx_min, pxx_max; // Minimum and maximum values of projector x coordinate
    int max_PX,
        min_PX; // Pixel to which are the voxels with minimum and maximum values are
                // projected
    // pxx_min = fmin(fmin(px00, px01), fmin(px10, px11));
    // pxx_max = fmax(fmax(px00, px01), fmax(px10, px11));
    double3* V_ccw[4]; // Point in which minimum is achieved and counter clock wise points
    // from the minimum voxel
    double* PX_ccw[4]; // Point in which minimum is achieved and counter clock wise  points
    // from the minimum voxel
#endif
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
#ifdef RELAXED
        float3 vd1 = (*V_ccw[1]) - (*V_ccw[0]);
        float3 vd3 = (*V_ccw[3]) - (*V_ccw[0]);
#else
        double3 vd1 = (*V_ccw[1]) - (*V_ccw[0]);
        double3 vd3 = (*V_ccw[3]) - (*V_ccw[0]);
#endif
        if(max_PX <= min_PX) // These indices are in the admissible range
        {
#ifdef RELAXED
            min_PX = convert_int_rtn(0.5f * (pxx_min + pxx_max) + 0.5f);
            ADD = backprojectExactEdgeValuesF0(&projection[projectionOffset], CMF,
                                               (vx10 + vx01) / 2.0f, min_PX, voxelSizesF, pdims);
#else
            min_PX = convert_int_rtn(0.5 * (pxx_min + pxx_max) + 0.5);
            ADD = backprojectExactEdgeValues0(&projection[projectionOffset], CM,
                                              (vx10 + vx01) / 2.0, min_PX, voxelSizes, pdims);
#endif
            volume[IND] += value * ADD;
        } else
        {
#ifdef RELAXED
            float lastSectionSize, nextSectionSize, polygonSize;
            float3 lastInt, nextInt, Int;
#else
            double lastSectionSize, nextSectionSize, polygonSize;
            double3 lastInt, nextInt, Int;
#endif
            int I = max(-1, min_PX);
            int I_STOP = min(max_PX, pdims.x);
            float factor;
            // Section of the square that corresponds to the indices < i
            // CCW and CW coordinates of the last intersection on the lines specified by the points
            // in V_ccw lastSectionSize
            //    = findIntersectionPoints(((double)I) + 0.5, V_ccw[0], V_ccw[1], V_ccw[2],
            //    V_ccw[3],
            //                             PX_ccw[0], PX_ccw[1], PX_ccw[2], PX_ccw[3], &lastInt);
#ifdef RELAXED
            lastSectionSize = exactIntersectionPointsF0_extended(
                ((float)I) + 0.5f, V_ccw[0], V_ccw[1], V_ccw[2], V_ccw[3], vd1, vd3, PX_ccw[0],
                PX_ccw[1], PX_ccw[2], PX_ccw[3], CMF, &lastInt);
#else
            lastSectionSize = exactIntersectionPoints0_extended(
                ((double)I) + 0.5, V_ccw[0], V_ccw[1], V_ccw[2], V_ccw[3], vd1, vd3, PX_ccw[0],
                PX_ccw[1], PX_ccw[2], PX_ccw[3], CM, &lastInt);
#endif
            if(I >= 0)
            {
#ifdef RELAXED
                factor = backprojectExactEdgeValuesF0(&projection[projectionOffset], CMF, lastInt,
                                                      I, voxelSizesF, pdims);
#else
                factor = backprojectExactEdgeValues0(&projection[projectionOffset], CM, lastInt, I,
                                                     voxelSizes, pdims);
#endif
                ADD += lastSectionSize * factor;
            }
            for(I = I + 1; I < I_STOP; I++)
            {
#ifdef RELAXED
                nextSectionSize = exactIntersectionPointsF0_extended(
                    ((float)I) + 0.5F, V_ccw[0], V_ccw[1], V_ccw[2], V_ccw[3], vd1, vd3, PX_ccw[0],
                    PX_ccw[1], PX_ccw[2], PX_ccw[3], CMF, &nextInt);
#else
                nextSectionSize = exactIntersectionPoints0_extended(
                    ((double)I) + 0.5, V_ccw[0], V_ccw[1], V_ccw[2], V_ccw[3], vd1, vd3, PX_ccw[0],
                    PX_ccw[1], PX_ccw[2], PX_ccw[3], CM, &nextInt);
#endif
                polygonSize = nextSectionSize - lastSectionSize;
                Int = (nextSectionSize * nextInt - lastSectionSize * lastInt) / polygonSize;
#ifdef RELAXED
                factor = backprojectExactEdgeValuesF0(&projection[projectionOffset], CMF, Int, I,
                                                      voxelSizesF, pdims);
#else
                factor = backprojectExactEdgeValues0(&projection[projectionOffset], CM, Int, I,
                                                     voxelSizes, pdims);
#endif
                ADD += polygonSize * factor;
                lastSectionSize = nextSectionSize;
                lastInt = nextInt;
            }
            if(I_STOP < pdims.x)
            {
#ifdef RELAXED
                polygonSize = 1.0f - lastSectionSize;
                Int = ((*V_ccw[0] + *V_ccw[2]) * 0.5f - lastSectionSize * lastInt) / polygonSize;
                factor = backprojectExactEdgeValuesF0(&projection[projectionOffset], CMF, Int, I,
                                                      voxelSizesF, pdims);
#else
                polygonSize = 1.0 - lastSectionSize;
                Int = ((*V_ccw[0] + *V_ccw[2]) * 0.5 - lastSectionSize * lastInt) / polygonSize;
                factor = backprojectExactEdgeValues0(&projection[projectionOffset], CM, Int, I,
                                                     voxelSizes, pdims);
#endif
                ADD += polygonSize * factor;
            }
            volume[IND] += value * ADD;
        }
    }
}
//==============================END backprojector.cl=====================================
