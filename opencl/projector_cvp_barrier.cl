//==============================projector_cvp_barrier.cl=====================================
#ifndef zeroPrecisionTolerance
#define zeroPrecisionTolerance 1e-10
#endif

/** Projection of a volume point v onto X coordinate on projector.
 * No checks for boundaries.
 *
 * @param CM Projection camera matrix
 * @param v Volume point
 * @param PX_out Output
 */
inline double projectX(private const double16 CM, private const double3 v)
{
    return (dot(v, CM.s012) + CM.s3) / (dot(v, CM.s89a) + CM.sb);
}

/** Projection of a volume point v onto Y coordinate on projector.
 * No checks for boundaries.
 *
 * @param CM Projection camera matrix
 * @param v Volume point
 * @param PY_out Output
 */
inline double projectY(private const double16 CM, private const double3 v)
{
    return (dot(v, CM.s456) + CM.s7) / (dot(v, CM.s89a) + CM.sb);
}

/** Projection of a volume point v onto P coordinate on projector.
 * No checks for boundaries.
 *
 * @param CM Projection camera matrix
 * @param v Volume point
 * @param P_out Output
 */
inline void project(private const double16* CM, private const double3* v, private double2* P_out)
{

    double3 coord;
    coord.x = dot(*v, CM->s012) + CM->s3;
    coord.y = dot(*v, CM->s456) + CM->s7;
    coord.z = dot(*v, CM->s89a) + CM->sb;
    P_out->x = coord.x / coord.z;
    P_out->y = coord.y / coord.z;
}

int2 projectionIndices(private double16 CM, private double3 v, int2 pdims)
{
    double3 coord;
    coord.x = dot(v, CM.s012);
    coord.y = dot(v, CM.s456);
    coord.z = dot(v, CM.s89a);
    coord += CM.s37b;
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

int projectionIndex(private double16 CM, private double3 v, int2 pdims)
{
    double3 coord;
    coord.x = dot(v, CM.s012);
    coord.y = dot(v, CM.s456);
    coord.z = dot(v, CM.s89a);
    coord += CM.s37b;
    coord.x /= coord.z;
    coord.y /= coord.z;
    int2 ind;
    ind.x = convert_int_rtn(coord.x + 0.5);
    ind.y = convert_int_rtn(coord.y + 0.5);
    if(ind.x >= 0 && ind.y >= 0 && ind.x < pdims.x && ind.y < pdims.y)
    {
        return ind.x * pdims.y + ind.y;
    } else
    {
        return -1;
    }
}

#define DROPINCOMPLETEVOXELS

#define LOCALMINMAX(PJ_min, PJ_max, v_min, v_min_minus_v_max_y)                                    \
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
        AtomicAdd_l_f(&projection[J], (lambda - lastLambda) * value);                              \
        lastLambda = lambda;                                                                       \
    }                                                                                              \
    AtomicAdd_l_f(&projection[PJ_max], (leastLambda - lastLambda) * value);

#define EDGEMINMAX(PJ_min, PJ_max, v_min, v_min_minus_v_max_y)                                     \
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
        AtomicAdd_g_f(&projection[J], (lambda - lastLambda) * value);                              \
        lastLambda = lambda;                                                                       \
    }                                                                                              \
    AtomicAdd_g_f(&projection[PJ_max], (leastLambda - lastLambda) * value);

#define EDGEMINMAXP(PJ_min, PJ_max, v_min, v_min_minus_v_max_y)                                    \
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
        AtomicAdd_g_f(&projection[J], (lambda - lastLambda) * value);                              \
        lastLambda = lambda;                                                                       \
    }                                                                                              \
    AtomicAdd_g_f(&projection[PJ_max], (leastLambda - lastLambda) * value);

/// insertEdgeValues(factor, V, P, projection, pdims);
void inline exactEdgeValues0(global float* projection,
                             private double16 CM,
                             private double3 v,
                             private int PX,
                             private double value,
                             private double3 voxelSizes,
                             private int2 pdims)
{
    projection = projection + PX * pdims.y;
    const double3 distanceToEdge = (double3)(0.0, 0.0, 0.5 * voxelSizes.s2);
    const double3 v_up = v + distanceToEdge;
    const double3 v_down = v - distanceToEdge;
    const double PY_up = projectY0(CM, v_up);
    const double PY_down = projectY0(CM, v_down);
    // const double3 v_diff = v_down - v_up;
    int PJ_up = convert_int_rtn(PY_up + 0.5);
    int PJ_down = convert_int_rtn(PY_down + 0.5);

    int J;
    double lambda;
    double lastLambda = 0.0;
    double leastLambda;
    double3 Fvector;
    int PJ_max;
    if(PJ_up < PJ_down)
    {
#ifdef DROPINCOMPLETEVOXELS
        if(PJ_up >= 0 && PJ_down < pdims.y)
#else
        if(PJ_down >= 0 && PJ_up < pdims.y)
#endif
        {
            EDGEMINMAX(PJ_up, PJ_down, v_up, voxelSizes.s2);
        }
    } else if(PJ_down < PJ_up)
    {
#ifdef DROPINCOMPLETEVOXELS
        if(PJ_down >= 0 && PJ_up < pdims.y)
#else
        if(PJ_up >= 0 && PJ_down < pdims.y)
#endif
        {
            EDGEMINMAX(PJ_down, PJ_up, v_down, -voxelSizes.s2);
        }
    } else if(PJ_down >= 0 && PJ_down < pdims.y)
    {
        AtomicAdd_g_f(&projection[PJ_down], value);
    }
}

void inline localEdgeValues0(local float* projection,
                             private REAL16 CM,
                             private REAL3 v,
                             private int PX,
                             private REAL value,
                             private REAL3 voxelSizes,
                             private int2 pdims)
{
    projection = projection + PX * pdims.y;
    const REAL3 distanceToEdge = (REAL3)(ZERO, ZERO, HALF * voxelSizes.s2);
    const REAL3 v_up = v + distanceToEdge;
    const REAL3 v_down = v - distanceToEdge;
    const REAL PY_up = PROJECTY0(CM, v_up);
    const REAL PY_down = PROJECTY0(CM, v_down);
    // const double3 v_diff = v_down - v_up;
    int PJ_up = INDEX(PY_up);
    int PJ_down = INDEX(PY_down);

    int J;
    REAL lambda;
    REAL lastLambda = ZERO;
    REAL leastLambda;
    REAL3 Fvector;
    int PJ_max;
    if(PJ_up < PJ_down)
    {
        if(PJ_down >= 0 && PJ_up < pdims.y)
        {
            LOCALMINMAX(PJ_up, PJ_down, v_up, voxelSizes.s2);
        }
    } else if(PJ_down < PJ_up)
    {
        if(PJ_up >= 0 && PJ_down < pdims.y)
        {
            LOCALMINMAX(PJ_down, PJ_up, v_down, -voxelSizes.s2);
        }
    } else if(PJ_down >= 0 && PJ_down < pdims.y)
    {
        AtomicAdd_l_f(&projection[PJ_down], value);
    }
}

void inline exactEdgeValuesF0(global float* projection,
                              private float16 CM,
                              private float3 v,
                              private int PX,
                              private float value,
                              private float3 voxelSizes,
                              private int2 pdims)
{
    projection = projection + PX * pdims.y;
    const float3 distanceToEdge = (float3)(0.0f, 0.0f, 0.5f * voxelSizes.s2);
    const float3 v_up = v + distanceToEdge;
    const float3 v_down = v - distanceToEdge;
    const float PY_up = PROJECTY0(CM, v_up);
    const float PY_down = PROJECTY0(CM, v_down);
    // const double3 v_diff = v_down - v_up;
    int PJ_up = convert_int_rtn(PY_up + 0.5);
    int PJ_down = convert_int_rtn(PY_down + 0.5);

    int J;
    float lambda;
    float lastLambda = 0.0;
    float leastLambda;
    float3 Fvector;
    int PJ_max;
    if(PJ_up < PJ_down)
    {
        if(PJ_down >= 0 && PJ_up < pdims.y)
        {
            EDGEMINMAXP(PJ_up, PJ_down, v_up, voxelSizes.s2);
        }
    } else if(PJ_down < PJ_up)
    {
        if(PJ_up >= 0 && PJ_down < pdims.y)
        {
            EDGEMINMAXP(PJ_down, PJ_up, v_down, -voxelSizes.s2);
        }
    } else if(PJ_down >= 0 && PJ_down < pdims.y)
    {
        AtomicAdd_g_f(&projection[PJ_down], value);
    }
}

/// insertEdgeValues(factor, V, P, projection, pdims);
void inline exactEdgeValues(global float* projection,
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
                AtomicAdd_g_f(&projection[PX * pdims.y + J],
                              (lambda - lastLambda)
                                  * value); // Atomic version of projection[ind] += value;
                lastLambda = lambda;
            }
            // PJ_max
            AtomicAdd_g_f(&projection[PX * pdims.y + PJ_max],
                          (leastLambda - lastLambda)
                              * value); // Atomic version of projection[ind] += value;
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
                AtomicAdd_g_f(&projection[PX * pdims.y + J],
                              (lastLambda - lambda)
                                  * value); // Atomic version of projection[ind] += value;
                lastLambda = lambda;
            }
            // PJ_max
            AtomicAdd_g_f(&projection[PX * pdims.y + PJ_max],
                          (lastLambda - leastLambda)
                              * value); // Atomic version of projection[ind] += value;
        }
    } else if(PJ_down == PJ_up && PJ_down >= 0 && PJ_down < pdims.y)
    {
        AtomicAdd_g_f(&projection[PX * pdims.y + PJ_down],
                      value); // Atomic version of projection[ind] += value;
    }
}

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
                      value); // Atomic version of projection[ind] += value;
        return;
    }
    double stepSize = value
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

__constant double onethird = 1.0 / 3.0;
__constant double twothirds = 2.0 / 3.0;
__constant double onesixth = 1.0 / 6.0;

inline double exactIntersectionPoints0_extended(const double PX,
                                                const double3* v0,
                                                const double3* v1,
                                                const double3* v2,
                                                const double3* v3,
                                                const double3 vd1,
                                                const double3 vd3,
                                                const double* PX_ccw0,
                                                const double* PX_ccw1,
                                                const double* PX_ccw2,
                                                const double* PX_ccw3,
                                                const double16 CM,
                                                double3* centroid)
{
    const double3 Fvector = CM.s012 - PX * CM.s89a;
    // const double3 vd1 = (*v1) - (*v0);
    // const double3 vd3 = (*v3) - (*v0);
    double Fproduct, FproductVD;
    double p, q;
    double A, w, wcomplement;
    if(PX < (*PX_ccw1))
    {
        Fproduct = -dot(*v0, Fvector);
        FproductVD = dot(vd1, Fvector); // VD1
        p = Fproduct / FproductVD; // v0+p*(v1-v0)
        if(PX < (*PX_ccw3))
        {
            q = Fproduct / dot(vd3, Fvector);
            (*centroid) = (*v0) + (onethird * p) * vd1 + (onethird * q) * vd3;
            return 0.5 * p * q;
        } else if(PX < (*PX_ccw2))
        {
            q = -dot(*v3, Fvector) / FproductVD;
            A = 0.5 * (p + q);
            if(A != 0.0) // Due to rounding errors equality might happen producing nan
            {
                w = p / A;
                //(*centroid) = (*v0)
                //    + mad(p, mad(-1.0 / 6.0, w, 2.0 / 3.0), mad(-q, w, q) / 3.0) * (vd1)
                //    + mad(-1.0 / 6.0, w, 2.0 / 3.0) * (vd3);
                //(*centroid) = (*v0) + (p * (2.0 / 3.0 - w / 6.0) + q * (1 - w) / 3.0) * (vd1)
                //    + (2.0 / 3.0 - w / 6.0) * (vd3);
                wcomplement = twothirds - onesixth * w;
                (*centroid) = (*v0) + (p * wcomplement + onethird * q * (1.0 - w)) * (vd1)
                    + wcomplement * (vd3);
            } else
            {
                (*centroid) = (*v0);
            }
            return A;
        } else
        {
            p = 1.0 - p;
            q = -dot(*v1, Fvector) / dot(vd3, Fvector);
            A = 1.0 - 0.5 * p * q;
            // w = 1.0 / A;
            //(*centroid) = (*v0) - mad(0.5, w, mad(p, -w, p) / 3.0) * vd1
            //    + mad(0.5, w, mad(q, -w, q) / 3.0) * vd3;
            //(*centroid) = (*v1) - (0.5 * w + (p * (1 - w)) / 3.0) * vd1
            //    + (0.5 * w + (q * (1 - w)) / 3.0) * vd3;
            w = 0.5 / A;
            wcomplement = twothirds * (0.5 - w);
            (*centroid) = (*v1) - (w + p * wcomplement) * vd1 + (w + q * wcomplement) * vd3;
            return A;
        }
    } else if(PX < (*PX_ccw2))
    {
        Fproduct = dot(*v2, Fvector);
        FproductVD = dot(vd3, Fvector);
        p = Fproduct / FproductVD; // V2 + p * (V1-V2)
        if(PX < (*PX_ccw3))
        {
            p = 1.0 - p; // V1 + p * (V2-V1)
            q = -dot(*v0, Fvector) / FproductVD; // V0 + q (V3-V0)
            A = 0.5 * (p + q);
            if(A != 0.0) // Due to rounding errors equality might happen producing nan
            {
                w = q / A;
                //(*centroid) = (*v0)
                //    + mad(q, mad(-1.0 / 6.0, w, 2.0 / 3.0), mad(-p, w, p) / 3.0) * (vd3)
                //    + mad(-1.0 / 6.0, w, 2.0 / 3.0) * (vd1);
                //(*centroid) = (*v0) + (q * (2.0 / 3.0 - w / 6.0) + p * (1 - w) / 3.0) * (vd3)
                //    + (2.0 / 3.0 - w / 6.0) * (vd1);
                wcomplement = twothirds - onesixth * w;
                (*centroid) = (*v0) + (q * wcomplement + onethird * p * (1.0 - w)) * (vd3)
                    + wcomplement * (vd1);
            } else
            {
                (*centroid) = (*v0);
            }
            return A;
        } else
        {
            q = Fproduct / dot(vd1, Fvector); // v2+q(v3-v2)
            A = 1.0 - 0.5 * p * q;
            // w = 1.0 / A;
            //(*centroid) = (*v2) - mad(0.5, w, mad(q, -w, q) / 3.0) * vd1
            //    - mad(0.5, w, mad(p, -w, p) / 3.0) * vd3;
            //(*centroid) = (*v2) - (0.5 * w + (q * (1 - w)) / 3.0) * vd1
            //    - (0.5 * w + (p * (1 - w)) / 3.0) * vd3;
            w = 0.5 / A;
            wcomplement = twothirds * (0.5 - w);
            (*centroid) = (*v2) - (w + q * wcomplement) * vd1 - (w + p * wcomplement) * vd3;
            return A;
        }
    } else if(PX >= *PX_ccw3)
    {
        (*centroid) = 0.5 * ((*v0) + (*v2));
        return 1.0;

    } else
    {
        Fproduct = dot(*v3, Fvector);
        p = Fproduct / dot(vd3, Fvector);
        q = -Fproduct / dot(vd1, Fvector);
        A = 1.0 - 0.5 * p * q;
        // w = 1.0 / A;
        //(*centroid) = (*v3) + mad(0.5, w, mad(q, -w, q) / 3.0) * vd1
        //    - mad(0.5, w, mad(p, -w, p) / 3.0) * vd3;
        //(*centroid)
        //    = (*v3) + (0.5 * w + (p * (1 - w)) / 3.0) * vd1 - (0.5 * w + (q * (1 - w)) / 3.0) *
        //    vd3;
        w = 0.5 / A;
        wcomplement = twothirds * (0.5 - w);
        (*centroid) = (*v3) + (w + p * wcomplement) * vd1 - (w + q * wcomplement) * vd3;
        return A;
    }
}

#define ONETHIRDF 0.33333333f
#define TWOTHIRDSF 0.66666666f
#define ONESIXTHF 0.16666667f

inline float exactIntersectionPointsF0_extended(const float PX,
                                                const float3* v0,
                                                const float3* v1,
                                                const float3* v2,
                                                const float3* v3,
                                                const float3 vd1,
                                                const float3 vd3,
                                                const float* PX_ccw0,
                                                const float* PX_ccw1,
                                                const float* PX_ccw2,
                                                const float* PX_ccw3,
                                                const float16 CM,
                                                float3* centroid)
{
    const float3 Fvector = CM.s012 - PX * CM.s89a;
    // const double3 vd1 = (*v1) - (*v0);
    // const double3 vd3 = (*v3) - (*v0);
    float Fproduct, FproductVD;
    float p, q;
    float A, w, wcomplement;
    if(PX < (*PX_ccw1))
    {
        Fproduct = -dot(*v0, Fvector);
        FproductVD = dot(vd1, Fvector); // VD1
        p = Fproduct / FproductVD; // v0+p*(v1-v0)
        if(PX < (*PX_ccw3))
        {
            q = Fproduct / dot(vd3, Fvector);
            (*centroid) = (*v0) + (ONETHIRDF * p) * vd1 + (ONETHIRDF * q) * vd3;
            return 0.5f * p * q;
        } else if(PX < (*PX_ccw2))
        {
            q = -dot(*v3, Fvector) / FproductVD;
            A = 0.5f * (p + q);
            if(A != 0.0f) // Due to rounding errors equality might happen producing nan
            {
                w = p / A;
                //(*centroid) = (*v0)
                //    + mad(p, mad(-1.0 / 6.0, w, 2.0 / 3.0), mad(-q, w, q) / 3.0) * (vd1)
                //    + mad(-1.0 / 6.0, w, 2.0 / 3.0) * (vd3);
                //(*centroid) = (*v0) + (p * (2.0 / 3.0 - w / 6.0) + q * (1 - w) / 3.0) * (vd1)
                //    + (2.0 / 3.0 - w / 6.0) * (vd3);
                wcomplement = TWOTHIRDSF - ONESIXTHF * w;
                (*centroid) = (*v0) + (p * wcomplement + ONETHIRDF * q * (1.0f - w)) * (vd1)
                    + wcomplement * (vd3);
            } else
            {
                (*centroid) = (*v0);
            }
            return A;
        } else
        {
            p = 1.0f - p;
            q = -dot(*v1, Fvector) / dot(vd3, Fvector);
            A = 1.0f - 0.5f * p * q;
            // w = 1.0 / A;
            //(*centroid) = (*v0) - mad(0.5, w, mad(p, -w, p) / 3.0) * vd1
            //    + mad(0.5, w, mad(q, -w, q) / 3.0) * vd3;
            //(*centroid) = (*v1) - (0.5 * w + (p * (1 - w)) / 3.0) * vd1
            //    + (0.5 * w + (q * (1 - w)) / 3.0) * vd3;
            w = 0.5f / A;
            wcomplement = TWOTHIRDSF * (0.5f - w);
            (*centroid) = (*v1) - (w + p * wcomplement) * vd1 + (w + q * wcomplement) * vd3;
            return A;
        }
    } else if(PX < (*PX_ccw2))
    {
        Fproduct = dot(*v2, Fvector);
        FproductVD = dot(vd3, Fvector);
        p = Fproduct / FproductVD; // V2 + p * (V1-V2)
        if(PX < (*PX_ccw3))
        {
            p = 1.0f - p; // V1 + p * (V2-V1)
            q = -dot(*v0, Fvector) / FproductVD; // V0 + q (V3-V0)
            A = 0.5f * (p + q);
            if(A != 0.0f) // Due to rounding errors equality might happen producing nan
            {
                w = q / A;
                //(*centroid) = (*v0)
                //    + mad(q, mad(-1.0 / 6.0, w, 2.0 / 3.0), mad(-p, w, p) / 3.0) * (vd3)
                //    + mad(-1.0 / 6.0, w, 2.0 / 3.0) * (vd1);
                //(*centroid) = (*v0) + (q * (2.0 / 3.0 - w / 6.0) + p * (1 - w) / 3.0) * (vd3)
                //    + (2.0 / 3.0 - w / 6.0) * (vd1);
                wcomplement = TWOTHIRDSF - ONESIXTHF * w;
                (*centroid) = (*v0) + (q * wcomplement + ONETHIRDF * p * (1.0f - w)) * (vd3)
                    + wcomplement * (vd1);
            } else
            {
                (*centroid) = (*v0);
            }
            return A;
        } else
        {
            q = Fproduct / dot(vd1, Fvector); // v2+q(v3-v2)
            A = 1.0f - 0.5f * p * q;
            // w = 1.0 / A;
            //(*centroid) = (*v2) - mad(0.5, w, mad(q, -w, q) / 3.0) * vd1
            //    - mad(0.5, w, mad(p, -w, p) / 3.0) * vd3;
            //(*centroid) = (*v2) - (0.5 * w + (q * (1 - w)) / 3.0) * vd1
            //    - (0.5 * w + (p * (1 - w)) / 3.0) * vd3;
            w = 0.5f / A;
            wcomplement = TWOTHIRDSF * (0.5f - w);
            (*centroid) = (*v2) - (w + q * wcomplement) * vd1 - (w + p * wcomplement) * vd3;
            return A;
        }
    } else if(PX >= *PX_ccw3)
    {
        (*centroid) = 0.5f * ((*v0) + (*v2));
        return 1.0f;

    } else
    {
        Fproduct = dot(*v3, Fvector);
        p = Fproduct / dot(vd3, Fvector);
        q = -Fproduct / dot(vd1, Fvector);
        A = 1.0f - 0.5f * p * q;
        // w = 1.0 / A;
        //(*centroid) = (*v3) + mad(0.5, w, mad(q, -w, q) / 3.0) * vd1
        //    - mad(0.5, w, mad(p, -w, p) / 3.0) * vd3;
        //(*centroid)
        //    = (*v3) + (0.5 * w + (p * (1 - w)) / 3.0) * vd1 - (0.5 * w + (q * (1 - w)) / 3.0) *
        //    vd3;
        w = 0.5f / A;
        wcomplement = TWOTHIRDSF * (0.5f - w);
        (*centroid) = (*v3) + (w + p * wcomplement) * vd1 - (w + q * wcomplement) * vd3;
        return A;
    }
}

inline uint voxelIndex(uint i, uint j, uint k, int3 vdims)
{
    return i + j * vdims.x + k * vdims.x * vdims.y;
}

//#define  Theoretical maximum of 65536 bytes AMD, 49152 NVIDIA, 32768 Intel

/** Kernel to precompute projection indices to spare some redundancy.
 *
 * @param vertexProjectionIndices
 * @param CM
 * @param voxelSizes
 * @param vdims
 *
 * @return
 */
void kernel computeProjectionIndices(global int* vertexProjectionIndices,
                                     private double16 CM,
                                     double3 voxelSizes,
                                     int3 vdims,
                                     int2 pdims)
{
    uint i = get_global_id(2);
    uint j = get_global_id(1);
    uint k = get_global_id(0); // This is more effective from the perspective of atomic colisions
    const double3 IND_ijk = { (double)(i), (double)(j), (double)(k) };
    const double3 zerocorner_xyz = { -0.5 * (double)vdims.x, -0.5 * (double)vdims.y,
                                     -0.5 * (double)vdims.z }; // -convert_double3(vdims) / 2.0;
    vertexProjectionIndices[i + j * (vdims.x + 1) + k * (vdims.x + 1) * (vdims.y + 1)]
        = projectionIndex(CM, zerocorner_xyz + voxelSizes * IND_ijk, pdims);
}

// Finds ranges of corners when center v is given
inline void
getVoxelRanges(REAL3 v, REAL3 voxelSizes, REAL16 CM, int* I_min, int* I_max, int* J_min, int* J_max)
{
    REAL nI = dot(CM.s012, v);
    REAL nJ = dot(CM.s456, v);
    REAL dv = dot(CM.s89a, v);
    REAL hxI = CM.s0 * voxelSizes.x;
    REAL hyI = CM.s1 * voxelSizes.y;
    REAL hzI = CM.s2 * voxelSizes.z;
    REAL hxJ = CM.s4 * voxelSizes.x;
    REAL hyJ = CM.s5 * voxelSizes.y;
    REAL hzJ = CM.s6 * voxelSizes.z;
    REAL hxD = CM.s8 * voxelSizes.x;
    REAL hyD = CM.s9 * voxelSizes.y;
    REAL hzD = CM.sa * voxelSizes.z;
    REAL numeratorI = nI - HALF * (hxI + hyI + hzI);
    REAL numeratorJ = nJ - HALF * (hxJ + hyJ + hzJ);
    REAL denominator = dv - HALF * (hxD + hyD + hzD);
    int IND1 = INDEX(numeratorI / denominator); // 000
    int JND1 = INDEX(numeratorJ / denominator); // 000
    numeratorI += hzI;
    numeratorJ += hzJ;
    denominator += hzD;
    int IND2 = INDEX(numeratorI / denominator); // 001
    int JND2 = INDEX(numeratorJ / denominator); // 001
    numeratorI += hyI;
    numeratorJ += hyJ;
    denominator += hyD;
    int IND3 = INDEX(numeratorI / denominator); // 011
    int JND3 = INDEX(numeratorJ / denominator); // 011
    numeratorI -= hzI;
    numeratorJ -= hzJ;
    denominator -= hzD;
    int IND4 = INDEX(numeratorI / denominator); // 010
    int JND4 = INDEX(numeratorJ / denominator); // 010
    numeratorI += hxI;
    numeratorJ += hxJ;
    denominator += hxD;
    int IND5 = INDEX(numeratorI / denominator); // 110
    int JND5 = INDEX(numeratorJ / denominator); // 110
    numeratorI -= hyI;
    numeratorJ -= hyJ;
    denominator -= hyD;
    int IND6 = INDEX(numeratorI / denominator); // 100
    int JND6 = INDEX(numeratorJ / denominator); // 100
    numeratorI += hzI;
    numeratorJ += hzJ;
    denominator += hzD;
    int IND7 = INDEX(numeratorI / denominator); // 101
    int JND7 = INDEX(numeratorJ / denominator); // 101
    numeratorI += hyI;
    numeratorJ += hyJ;
    denominator += hyD;
    int IND8 = INDEX(numeratorI / denominator); // 111
    int JND8 = INDEX(numeratorJ / denominator); // 111
    *I_min = min(min(min(IND1, IND2), min(IND3, IND4)), min(min(IND5, IND6), min(IND7, IND8)));
    *I_max = max(max(max(IND1, IND2), max(IND3, IND4)), max(max(IND5, IND6), max(IND7, IND8)));
    *J_min = min(min(min(JND1, JND2), min(JND3, JND4)), min(min(JND5, JND6), min(JND7, JND8)));
    *J_max = max(max(max(JND1, JND2), max(JND3, JND4)), max(max(JND5, JND6), max(JND7, JND8)));
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
 */
void kernel FLOATcutting_voxel_project_barrier(global const float* restrict volume,
                                               global float* restrict projection,
                                               local float* restrict localProjection,
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
    int li = get_local_id(0);
    int lj = get_local_id(1);
    int lk = get_local_id(2);
    int lis = get_local_size(0);
    int ljs = get_local_size(1);
    int lks = get_local_size(2);
    // Not uint for correct int subtraction
    // Shift projection array by offset
    projection += projectionOffset;
    // initialize localProjection in given workGroup
    uint LOCSIZE = lis * ljs * lks;
    uint LID = li * ljs * lks + lj * lks + lk;
    uint LIDRANGE = (LOCALARRAYSIZE + LOCSIZE - 1) / LOCSIZE;
    uint fillStart = LID * LIDRANGE;
    uint fillStop = min(fillStart + LIDRANGE, (uint)LOCALARRAYSIZE);
    for(uint IND = fillStart; IND != fillStop; IND++)
    {
        localProjection[IND] = 0.0f;
    }
    int2 Lpdims;
    uint mappedLocalRange, Jrange, ILocalRange; // Memory used only in cornerWorkItem
    local bool offAxisPosition; // If true, position of local cuboid is such that the direction
                                // of the increase/decrease of the X/Y projection indices is
                                // the same on colinear edges
    local bool partlyOffProjectorPosition; // If true, some vertices of the local cuboid are
                                           // projected outside the projector
    local bool fullyOffProjectorPosition; // If true, shall end the execution
    local bool stopNextIteration_local;
    bool stopNextIteration;
    local REAL16 CML;
    local REAL3 positiveShift[2];
    local int projectorLocalRange[7]; //
    int PILocalMin, PILocalMax, PJLocalMin, PJLocalMax;

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
    if(LID == 0) // Get dimension
    {
        const REAL3 volumeCenter_localVoxelcenter_offset
            = (REAL3)(2 * i + lis - vdims.x, 2 * j + ljs - vdims.y, 2 * k + lks - vdims.z)
            * halfVoxelSizes;
        const REAL3 voxelcenter_local_xyz
            = volumeCenter + volumeCenter_localVoxelcenter_offset - sourcePosition;

        const REAL3 halfLocalSizes
            = { HALF * lis * voxelSizes.x, HALF * ljs * voxelSizes.y, HALF * lks * voxelSizes.z };
        /*
const REAL3 IND_local_ijk = { i + HALF * lis, j + HALF * ljs, k + HALF * lks };
const REAL3 voxelcenter_local_xyz = zerocorner_xyz + (IND_local_ijk * voxelSizes);
*/
        positiveShift[0] = halfLocalSizes; // X direction
        positiveShift[1] = halfLocalSizes; // Y direction
        if(all(fabs(voxelcenter_local_xyz) > halfLocalSizes)) // Increase or decrease of the value
                                                              // will be preserved on colinear edges
        {
            // printf("TRUE i,j,k=(%d, %d, %d) %d %d %d\n", i, j, k, lis, ljs, lks);
            offAxisPosition = true;
            const REAL3 CMX_CROSS = cross(CM.s012, CM.s89a);
            const REAL3 CMY_CROSS = cross(CM.s456, CM.s89a);
            if(voxelcenter_local_xyz.y * CMX_CROSS.z - voxelcenter_local_xyz.z * CMX_CROSS.y < 0)
            {
                positiveShift[0].x = -positiveShift[0].x;
            }
            if(voxelcenter_local_xyz.z * CMX_CROSS.x - voxelcenter_local_xyz.x * CMX_CROSS.z < 0)
            {
                positiveShift[0].y = -positiveShift[0].y;
            }
            if(voxelcenter_local_xyz.x * CMX_CROSS.y - voxelcenter_local_xyz.y * CMX_CROSS.x < 0)
            {
                positiveShift[0].z = -positiveShift[0].z;
            }

            if(voxelcenter_local_xyz.y * CMY_CROSS.z - voxelcenter_local_xyz.z * CMY_CROSS.y < 0)
            {
                positiveShift[1].x *= -1.0;
            }
            if(voxelcenter_local_xyz.z * CMY_CROSS.x - voxelcenter_local_xyz.x * CMY_CROSS.z < 0)
            {
                positiveShift[1].y *= -1.0;
            }
            if(voxelcenter_local_xyz.x * CMY_CROSS.y - voxelcenter_local_xyz.y * CMY_CROSS.x < 0)
            {
                positiveShift[1].z *= -1.0;
            }
            REAL nvI = dot(CM.s012, voxelcenter_local_xyz);
            REAL n0I = dot(CM.s012, positiveShift[0]);
            REAL nvJ = dot(CM.s456, voxelcenter_local_xyz);
            REAL n1J = dot(CM.s456, positiveShift[1]);
            REAL dv = dot(CM.s89a, voxelcenter_local_xyz);
            REAL d0 = dot(CM.s89a, positiveShift[0]);
            REAL d1 = dot(CM.s89a, positiveShift[1]);
            PILocalMin = INDEX((nvI - n0I) / (dv - d0));
            PILocalMax = INDEX((nvI + n0I) / (dv + d0));
            PJLocalMin = INDEX((nvJ - n1J) / (dv - d1));
            PJLocalMax = INDEX((nvJ + n1J) / (dv + d1));
        } else
        {
            // printf("FALSE i,j,k=(%d, %d, %d) %d %d %d\n", i, j, k, lis, ljs, lks);
            // printf("voxelcenter_local_xyz=[%f, %f, %f], halfLocalSizes=[%f, %f, %f]",
            //       voxelcenter_local_xyz.s0, voxelcenter_local_xyz.s1, voxelcenter_local_xyz.s2,
            //       halfLocalSizes.s0, halfLocalSizes.s1, halfLocalSizes.s2);
            offAxisPosition = false;
            getVoxelRanges(voxelcenter_local_xyz, 2 * halfLocalSizes, CM, &PILocalMin, &PILocalMax,
                           &PJLocalMin, &PJLocalMax);
        }
        // printf("i,j,k=(%d, %d, %d) localSizes=(%d %d %d), PIRANGE=[%d %d] PJRANGE=[%d %d]\n", i,
        // j,
        //       k, lis, ljs, lks, PILocalMin, PILocalMax, PJLocalMin, PJLocalMax);
        if(PILocalMax < 0 || PILocalMin >= pdims.x || PJLocalMax < 0 || PJLocalMin >= pdims.y)
        {
            fullyOffProjectorPosition = true;
            partlyOffProjectorPosition = true;
        } else if(PILocalMin < 0 || PILocalMax >= pdims.x || PJLocalMin < 0
                  || PJLocalMax >= pdims.y)
        {
            fullyOffProjectorPosition = false;
            partlyOffProjectorPosition = true;
            projectorLocalRange[0] = max(0, PILocalMin);
            projectorLocalRange[1] = min(pdims.x, PILocalMax + 1);
            projectorLocalRange[2] = max(0, PJLocalMin);
            projectorLocalRange[3] = min(pdims.y, PJLocalMax + 1);
        } else
        {
            fullyOffProjectorPosition = false;
            partlyOffProjectorPosition = false;
            projectorLocalRange[0] = PILocalMin;
            projectorLocalRange[1] = PILocalMax + 1;
            projectorLocalRange[2] = PJLocalMin;
            projectorLocalRange[3] = PJLocalMax + 1;
        }
        // Prepare local memory
        if(!fullyOffProjectorPosition)
        {
            uint Irange = projectorLocalRange[1] - projectorLocalRange[0];
            uint Jrange = projectorLocalRange[3] - projectorLocalRange[2];
            uint FullLocalRange = Irange * Jrange;
            if(FullLocalRange <= LOCALARRAYSIZE)
            {
                ILocalRange = Irange; // How many columns fits to local memory
                stopNextIteration_local = true;
            } else
            {
                ILocalRange = LOCALARRAYSIZE / Jrange;
                stopNextIteration_local = false;
                // printf("%zu %zu %zu", i, j, k);
                // printf("FullLocalRange=%zu exceeds range\n", FullLocalRange);
            }
            projectorLocalRange[4] = ILocalRange;
            projectorLocalRange[5] = Jrange;
            projectorLocalRange[6]
                = projectorLocalRange[0]; // Where current local array has start IRange

            // 0..PIMIN, 1..PIMAX, 2..PJMIN, 3..PJMAX, 4 .. ILocalRange, 5 .. PJMAX-PJMIN, 6
            // CurrentPISTART
            CML.s0123 = CM.s0123 - projectorLocalRange[0] * CM.s89ab;
            CML.s4567 = CM.s4567 - projectorLocalRange[2] * CM.s89ab;
            CML.s89ab = CM.s89ab;
        } else
        {
            projectorLocalRange[1] = -1;
            projectorLocalRange[2] = 0;
            projectorLocalRange[3] = 0;
            projectorLocalRange[4] = 0;
            projectorLocalRange[5] = 0;
            projectorLocalRange[6] = 0;
            stopNextIteration_local = true;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE); // Cutting voxel projector
                                  // 0..PIMIN, 1..PIMAX, 2..PJMIN, 3..PJMAX, 4 .. ILocalRange, 5
                                  // .. PJMAX-PJMIN, 6 CurrentPISTART
    if(fullyOffProjectorPosition)
        return;
    uint prevStartRange;
    const uint IND = voxelIndex(i, j, k, vdims);
    const float voxelValue = volume[IND];
    bool dropVoxel = false;
    if(voxelValue == 0.0f)
    {
        dropVoxel = true;
    }
#ifdef DROPINCOMPLETEVOXELS
    if(partlyOffProjectorPosition)
    {
        int xindex = INDEX(PROJECTX0(CM, voxelcenter_xyz));
        int yindex = INDEX(PROJECTY0(CM, voxelcenter_xyz));
        if(xindex < 0 || yindex < 0 || xindex >= pdims.x || yindex >= pdims.y)
        {
            dropVoxel = true;
        }
    }
#endif
    const REAL voxelVolume = voxelSizes.x * voxelSizes.y * voxelSizes.z;
    REAL sourceToVoxel_xyz_norm2 = dot(voxelcenter_xyz, voxelcenter_xyz);
#ifdef RELAXED
    float value = (voxelValue * voxelVolume * scalingFactor / sourceToVoxel_xyz_norm2);
#else
    float value = (float)(voxelValue * voxelVolume * scalingFactor / sourceToVoxel_xyz_norm2);
#endif

    // I assume that the volume point (x,y,z_1) projects to the same px as (x,y,z_2)
    // for any z_1, z_2  This assumption is restricted to the voxel edges, where it
    // holds very accurately  We project the rectangle that lies on the z midline of
    // the voxel on the projector
    REAL px00, px10, px01, px11;
    REAL3 vx00, vx10, vx01, vx11;
    do
    {
        stopNextIteration = stopNextIteration_local;
        prevStartRange = projectorLocalRange[6];
        // printf("i,j,k=%d,%d,%d", i, j, k);
        // Start CVP
        Lpdims = (int2)(projectorLocalRange[4], projectorLocalRange[5]);
        if(!dropVoxel)
        {
            /*
                        if(offAxisPosition)
                        {
                            int Imax = INDEX(PROJECTX0(CM, voxelcenter_xyz + positiveShift[0]));
                            int Imin = INDEX(PROJECTX0(CM, voxelcenter_xyz - positiveShift[0]));
                            int Jmax = INDEX(PROJECTY0(CM, voxelcenter_xyz + positiveShift[1]));
                            int Jmin = INDEX(PROJECTY0(CM, voxelcenter_xyz - positiveShift[1]));
                            if(Imax < 0 || Jmax < 0 || Imin >= pdims.x || Jmin >= pdims.y)
                            {
                                dropVoxel = true;
                            }
                            if(LID==0 && voxelValue)
                            {
                                // Debug
                                int id, iu, jd, ju;
                                getVoxelRanges(voxelcorner_xyz, voxelSizes, CM, &id, &iu, &jd, &ju);
                                if((Imin != id || Imax != iu || Jmin != jd || Jmax != ju)
                                   && positiveShift[1].x > 0 && positiveShift[1].y < 0
                                   && positiveShift[1].z < 0)
                                {
                                    uint i0 = i - li;
                                    uint j0 = j - lj;
                                    uint k0 = k - lk;

                                    const REAL3 halfLocalSizes
                                        = { HALF * lis * voxelSizes.x, HALF * ljs * voxelSizes.y,
                                            HALF * lks * voxelSizes.z };
                                    const REAL3 IND_local_ijk
                                        = { i0 + HALF * lis, j0 + HALF * ljs, k0 + HALF * lks };
                                    const REAL3 voxelcenter_local_xyz
                                        = zerocorner_xyz + (IND_local_ijk * voxelSizes);
                                    printf("Local voxel center [%f, %f, %f] and half sizes [%f, %f,
               %f].\n", voxelcenter_local_xyz.x, voxelcenter_local_xyz.y, voxelcenter_local_xyz.z,
               halfLocalSizes.x, halfLocalSizes.y, halfLocalSizes.z); printf( "Value curl=%f
               value=%f\n", PROJECTY0(CM, voxelcenter_xyz - positiveShift[1]), PROJECTY0(CM,
               voxelcorner_xyz + voxelSizes * (REAL3)(ZERO, ONE, ONE))); REAL3 centerXX =
               voxelcenter_xyz - positiveShift[1]; REAL3 cornerXX = voxelcorner_xyz + voxelSizes *
               (REAL3)(ZERO, ONE, ONE); printf( "X_center=(%f, %f, %f) X_corner=(%f, %f, %f)
               voxelSizes=(%f, %f, %f)\n", centerXX.x, centerXX.y, centerXX.z, cornerXX.x,
               cornerXX.y, cornerXX.z, voxelSizes.x, voxelSizes.y, voxelSizes.z);
                                    if(all(fabs(voxelcenter_local_xyz)
                                           > halfLocalSizes
                                               + (REAL3)(
                                                     zeroPrecisionTolerance, zeroPrecisionTolerance,
                                                     zeroPrecisionTolerance))) // Increase or
               decrease of the
                                                                               // value will be
               preserved on
                                                                               // colinear edges
                                    {
                                        printf("OFFAXISPOSITION\n");
                                    }

                                    printf("(i,j,k)=(%d,%d,%d) center=[%f, %f, %f] is Imax_curl[%d,
               %d] I[%d, "
                                           "%d] Jmax_curl "
                                           "[%d "
                                           "%d] J[%d, %d]\n",
                                           i, j, k, voxelcenter_xyz.x, voxelcenter_xyz.y,
               voxelcenter_xyz.z, Imin, Imax, id, iu, Jmin, Jmax, jd, ju); printf("Local (li, lj,
               lk) = [ % d, % d, % d ] lis, ljs, lks = [ % d, % d, "
                                           "% d ]  \n",
                                           li, lj, lk, lis, ljs, lks);
                                    printf("Positive shift Y %f %f %f value=%f minusvalue=%f\n",
                                           positiveShift[1].x, positiveShift[1].y,
               positiveShift[1].z, PROJECTY0(CM, voxelcenter_xyz + positiveShift[1]), PROJECTY0(CM,
               voxelcenter_xyz - positiveShift[1])); REAL py000, py010, py100, py110, py001, py011,
               py101, py111; py000 = PROJECTY0(CM, voxelcorner_xyz + voxelSizes * (REAL3)(ZERO,
               ZERO, ZERO)); py100 = PROJECTY0(CM, voxelcorner_xyz + voxelSizes * (REAL3)(ONE, ZERO,
               ZERO)); py010 = PROJECTY0(CM, voxelcorner_xyz + voxelSizes * (REAL3)(ZERO, ONE,
               ZERO)); py110 = PROJECTY0(CM, voxelcorner_xyz + voxelSizes * (REAL3)(ONE, ONE,
               ZERO)); py001 = PROJECTY0(CM, voxelcorner_xyz + voxelSizes * (REAL3)(ZERO, ZERO,
               ONE)); py101 = PROJECTY0(CM, voxelcorner_xyz + voxelSizes * (REAL3)(ONE, ZERO, ONE));
                                    py011
                                        = PROJECTY0(CM, voxelcorner_xyz + voxelSizes * (REAL3)(ZERO,
               ONE, ONE)); py111 = PROJECTY0(CM, voxelcorner_xyz + voxelSizes * (REAL3)(ONE, ONE,
               ONE)); printf("py000=%f, py010=%f, py100=%f, py110=%f, py001=%f, py011=%f, "
                                           "py101=%f, py111=%f\n\n\n",
                                           py000, py010, py100, py110, py001, py011, py101, py111);

                                    printf("%f", PROJECTY0(CM, voxelcenter_xyz + positiveShift[1]));
                                    positiveShift[1].x *= -1;
                                    printf("x %f", PROJECTY0(CM, voxelcenter_xyz +
               positiveShift[1])); positiveShift[1].y *= -1; printf("xy %f", PROJECTY0(CM,
               voxelcenter_xyz + positiveShift[1])); positiveShift[1].x *= -1; printf("y %f\n",
               PROJECTY0(CM, voxelcenter_xyz + positiveShift[1])); positiveShift[1].y *= -1;
                                    positiveShift[1].z *= -1;
                                    printf("xz %f", PROJECTY0(CM, voxelcenter_xyz +
               positiveShift[1])); positiveShift[1].y *= -1; printf("xyz %f", PROJECTY0(CM,
               voxelcenter_xyz + positiveShift[1])); positiveShift[1].x *= -1; printf("yz %f\n",
               PROJECTY0(CM, voxelcenter_xyz + positiveShift[1])); positiveShift[1].y *= -1;
                                    positiveShift[1].z *= -1;
                                }
                                // Debug
                            }
                        }
            */
            vx00 = voxelcenter_xyz + halfVoxelSizes * (REAL3)(-ONE, -ONE, ZERO);
            vx10 = voxelcenter_xyz + halfVoxelSizes * (REAL3)(ONE, -ONE, ZERO);
            vx01 = voxelcenter_xyz + halfVoxelSizes * (REAL3)(-ONE, ONE, ZERO);
            vx11 = voxelcenter_xyz + halfVoxelSizes * (REAL3)(ONE, ONE, ZERO);
            {
                REAL nx = dot(voxelcenter_xyz, CML.s012);
                REAL dv = dot(voxelcenter_xyz, CML.s89a);
                REAL nhx = halfVoxelSizes.x * CML.s0;
                REAL nhy = halfVoxelSizes.y * CML.s1;
                REAL dhx = halfVoxelSizes.x * CML.s8;
                REAL dhy = halfVoxelSizes.y * CML.s9;
                px00 = (nx - nhx - nhy) / (dv - dhx - dhy);
                px01 = (nx - nhx + nhy) / (dv - dhx + dhy);
                px10 = (nx + nhx - nhy) / (dv + dhx - dhy);
                px11 = (nx + nhx + nhy) / (dv + dhx + dhy);
            }
            /*
px00 = PROJECTX0(CML, vx00);
px10 = PROJECTX0(CML, vx10);
px01 = PROJECTX0(CML, vx01);
px11 = PROJECTX0(CML, vx11);*/
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
            // from the minimum voxel
            if(offAxisPosition)
            {
                if(positiveShift[0].s0 > 0)
                {
                    if(positiveShift[0].s1 > 0)
                    {
                        pxx_max = px11;
                        pxx_min = px00;
                        V_ccw[0] = &vx00;
                        V_ccw[1] = &vx10;
                        V_ccw[2] = &vx11;
                        V_ccw[3] = &vx01;
                        PX_ccw[0] = &px00;
                        PX_ccw[1] = &px10;
                        PX_ccw[2] = &px11;
                        PX_ccw[3] = &px01;
                    } else
                    {
                        pxx_max = px10;
                        pxx_min = px01;
                        V_ccw[0] = &vx01;
                        V_ccw[1] = &vx00;
                        V_ccw[2] = &vx10;
                        V_ccw[3] = &vx11;
                        PX_ccw[0] = &px01;
                        PX_ccw[1] = &px00;
                        PX_ccw[2] = &px10;
                        PX_ccw[3] = &px11;
                    }
                } else
                {
                    if(positiveShift[0].s1 > 0)
                    {
                        pxx_max = px01;
                        pxx_min = px10;
                        V_ccw[0] = &vx10;
                        V_ccw[1] = &vx11;
                        V_ccw[2] = &vx01;
                        V_ccw[3] = &vx00;
                        PX_ccw[0] = &px10;
                        PX_ccw[1] = &px11;
                        PX_ccw[2] = &px01;
                        PX_ccw[3] = &px00;
                    } else
                    {
                        pxx_max = px00;
                        pxx_min = px11;
                        V_ccw[0] = &vx11;
                        V_ccw[1] = &vx01;
                        V_ccw[2] = &vx00;
                        V_ccw[3] = &vx10;
                        PX_ccw[0] = &px11;
                        PX_ccw[1] = &px01;
                        PX_ccw[2] = &px00;
                        PX_ccw[3] = &px10;
                    }
                }
            } else
            {
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
            }
            min_PX = convert_int_rtn(pxx_min + zeroPrecisionTolerance + HALF);
            max_PX = convert_int_rtn(pxx_max - zeroPrecisionTolerance + HALF);
            if(max_PX >= 0 && min_PX < Lpdims.x)
            {
                REAL3 vd1 = (*V_ccw[1]) - (*V_ccw[0]);
                REAL3 vd3 = (*V_ccw[3]) - (*V_ccw[0]);
                if(max_PX <= min_PX) // These indices are in the admissible range
                {
                    min_PX = convert_int_rtn(HALF * (pxx_min + pxx_max) + HALF);
                    localEdgeValues0(localProjection, CML, HALF * (vx00 + vx11), min_PX, value,
                                     voxelSizes, Lpdims);
                } else
                {
                    REAL lastSectionSize, nextSectionSize, polygonSize;
                    REAL3 lastInt, nextInt, Int;
                    REAL factor;
                    int I = max(-1, min_PX);
                    int I_STOP = min(max_PX, Lpdims.x);
                    // Section of the square that corresponds to the indices < i
                    // CCW and CW coordinates of the last intersection on the lines
                    // specified by the points in V_ccw
#ifdef RELAXED
                    lastSectionSize = exactIntersectionPointsF0_extended(
                        ((float)I) + 0.5f, V_ccw[0], V_ccw[1], V_ccw[2], V_ccw[3], vd1, vd3,
                        PX_ccw[0], PX_ccw[1], PX_ccw[2], PX_ccw[3], CML, &lastInt);
#else
                    lastSectionSize = exactIntersectionPoints0_extended(
                        ((double)I) + 0.5, V_ccw[0], V_ccw[1], V_ccw[2], V_ccw[3], vd1, vd3,
                        PX_ccw[0], PX_ccw[1], PX_ccw[2], PX_ccw[3], CML, &lastInt);
#endif
                    if(I >= 0)
                    {
                        factor = value * lastSectionSize;
                        // insertEdgeValues(&projection[projectionOffset], CM, lastInt, I,
                        // factor, voxelSizes, pdims);
                        //                                printf("lastInt=(%f, %f, %f),
                        //                                I=%d, factor=%f\n", lastInt.x,
                        //                                lastInt.y, lastInt.z, I, factor);
                        localEdgeValues0(localProjection, CML, lastInt, I, factor, voxelSizes,
                                         Lpdims);
                    }
                    for(I = I + 1; I < I_STOP; I++)
                    {
#ifdef RELAXED
                        nextSectionSize = exactIntersectionPointsF0_extended(
                            ((float)I) + 0.5f, V_ccw[0], V_ccw[1], V_ccw[2], V_ccw[3], vd1, vd3,
                            PX_ccw[0], PX_ccw[1], PX_ccw[2], PX_ccw[3], CML, &nextInt);
#else
                        nextSectionSize = exactIntersectionPoints0_extended(
                            ((double)I) + 0.5, V_ccw[0], V_ccw[1], V_ccw[2], V_ccw[3], vd1, vd3,
                            PX_ccw[0], PX_ccw[1], PX_ccw[2], PX_ccw[3], CML, &nextInt);
#endif
                        polygonSize = nextSectionSize - lastSectionSize;
                        Int = (nextSectionSize * nextInt - lastSectionSize * lastInt) / polygonSize;
                        factor = value * polygonSize;
                        localEdgeValues0(localProjection, CML, Int, I, factor, voxelSizes, Lpdims);
                        lastSectionSize = nextSectionSize;
                        lastInt = nextInt;
                    }
                    if(I_STOP < pdims.x)
                    {
                        polygonSize = ONE - lastSectionSize;
                        Int = ((*V_ccw[0] + *V_ccw[2]) * HALF - lastSectionSize * lastInt)
                            / polygonSize;
                        factor = value * polygonSize;
                        localEdgeValues0(localProjection, CML, Int, I, factor, voxelSizes, Lpdims);
                    }
                }
            }
        }

        // End CVP
        // 0..PIMIN, 1..PIMAX, 2..PJMIN, 3..PJMAX, 4 .. ILocalRange, 5 .. PJMAX-PJMIN, 6
        // CurrentPISTART
        barrier(CLK_LOCAL_MEM_FENCE); // Local to global copy
        mappedLocalRange = Lpdims.x * Lpdims.y;
        fillStart = min(fillStart, mappedLocalRange);
        fillStop = min(fillStop, mappedLocalRange);
        uint LI, LJ, globalIndex;
        uint globalOffset = prevStartRange * pdims.y + projectorLocalRange[2];
        for(int IND = fillStart; IND != fillStop; IND++)
        // for(int IND = LID; IND < mappedLocalRange; IND+=LIDRANGE)
        {
            LI = IND / Lpdims.y;
            LJ = IND % Lpdims.y;
            globalIndex = globalOffset + LI * pdims.y + LJ;
            AtomicAdd_g_f(projection + globalIndex, localProjection[IND]);
        }
        if(LID == 0)
        {
            CML.s0123 = CML.s0123 - projectorLocalRange[4] * CML.s89ab;
            // 0..PIMIN, 1..PIMAX, 2..PJMIN, 3..PJMAX, 4 .. PIMAX-PIMIN, 5 .. PJMAX-PJMIN, 6
            // CurrentPISTART
            projectorLocalRange[6] += projectorLocalRange[4];
            if(projectorLocalRange[6] < projectorLocalRange[1])
            {
                if(projectorLocalRange[1] - projectorLocalRange[6] <= projectorLocalRange[4])
                {
                    projectorLocalRange[4] = projectorLocalRange[1] - projectorLocalRange[6];
                    stopNextIteration_local = true;
                }
            }
        }

        if(!stopNextIteration)
        {
            for(int IND = fillStart; IND != fillStop; IND++)
            {
                localProjection[IND] = 0.0f;
            }
            barrier(CLK_LOCAL_MEM_FENCE); // Cutting voxel projector
                                          // 0..PIMIN, 1..PIMAX, 2..PJMIN, 3..PJMAX, 4 ..
                                          // ILocalRange, 5
                                          // .. PJMAX-PJMIN, 6 CurrentPISTART

            // printf("Next %d %d %d. \n", i, j, k);
        }
    } while(!stopNextIteration);
    //} while(startIRange < projectorLocalRange[1]);
}
//==============================END projector_cvp_barrier.cl=====================================
