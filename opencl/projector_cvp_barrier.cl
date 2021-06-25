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

//#define  Theoretical maximum of 65536 bytes AMD, 49152 NVIDIA
#define LOCALARRAYSIZE 10560 // 22 full columns of 480 elements each

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
    uint li = get_local_id(0);
    uint lj = get_local_id(1);
    uint lk = get_local_id(2);
    uint lis = get_local_size(0);
    uint ljs = get_local_size(1);
    uint lks = get_local_size(2);
    projection += projectionOffset;
    bool cornerWorkItem = false;
    uint mappedLocalRange, Jrange, ILocalRange; // Memory used only in cornerWorkItem
    __local bool
        offAxisPosition[1]; // If true, position of local cuboid is such that the direction
                            // of the increase/decrease of the X/Y projection indices is the
                            // same on colinear edges
    __local bool partlyOffProjectorPosition[1]; // If true, some vertices of the local cuboid are
                                                // projected outside the projector
    __local bool fullyOffProjectorPosition[1]; // If true, all vertices are mapped outside projector
    __local REAL3 positiveShift[2];
    __local uint projectorLocalRange[7]; //
    __local REAL16 CML[1];
    __local float localProjection[LOCALARRAYSIZE];

#ifdef RELAXED
    const float16 CMF = convert_float16(CM);
    const float3 sourcePositionF = convert_float3(sourcePosition);
    const float3 volumeCenterF = convert_float3(volumeCenter);
    const float3 voxelSizesF = convert_float3(voxelSizes);
    const float3 halfLocalSizes
        = { 0.5f * lis * vdims.x, 0.5f * ljs * vdims.y, 0.5f * lks * vdims.z };
#else
#define CMF CM
#define voxelSizesF voxelSizes
#define volumeCenterF volumeCenter
#define sourcePositionF sourcePosition
    const double3 halfLocalSizes
        = { 0.5 * lis * vdims.x, 0.5 * ljs * vdims.y, 0.5 * lks * vdims.z };
#endif
    const REAL3 zerocorner_xyz
        = volumeCenterF - sourcePositionF - HALF * convert_REAL3(vdims) * voxelSizesF;
    const REAL3 IND_ijk = { (REAL)(i), (REAL)(j), (REAL)(k) };
    const REAL3 voxelcorner_xyz = zerocorner_xyz + (IND_ijk * voxelSizesF);
    const REAL3 voxelcenter_xyz = voxelcorner_xyz + HALF * voxelSizesF;
    if(li == 0 && lj == 0 && lk == 0) // Get dimension
    {
        cornerWorkItem = true;
#ifdef RELAXED
        const float3 IND_local_ijk
            = { (float)(i) + 0.5f * lis, (float)(j) + 0.5f * ljs,
                (float)(k) + 0.5f * lks }; // Index of the center of the local cube
        const float3 voxelcenter_local_xyz = zerocorner_xyz + (IND_local_ijk * voxelSizesF);
#else
        const double3 IND_local_ijk
            = { (double)(i) + 0.5 * lis, (double)(j) + 0.5 * ljs,
                (double)(k) + 0.5 * lks }; // Index of the center of the local cube
        const double3 voxelcenter_local_xyz = zerocorner_xyz + (IND_local_ijk * voxelSizes);
#endif
        positiveShift[0] = voxelSizesF * HALF; // X direction
        positiveShift[1] = voxelSizesF * HALF; // Y direction
        int PILocalMin, PILocalMax, PJLocalMin, PJLocalMax;
        if(all(fabs(voxelcenter_local_xyz) > halfLocalSizes)) // Increase or decrease of the value
                                                              // will be preserved on colinear edges
        {
            offAxisPosition[0] = true;
            const REAL3 CMX_CROSS = cross(CMF.s012, CMF.s89a);
            const REAL3 CMY_CROSS = cross(CMF.s456, CMF.s89a);
            if(voxelcenter_local_xyz.y * CMX_CROSS.z - voxelcenter_local_xyz.z * CMX_CROSS.y < 0)
            {
                positiveShift[0].x *= -1.0;
            }
            if(voxelcenter_local_xyz.z * CMX_CROSS.x - voxelcenter_local_xyz.x * CMX_CROSS.z < 0)
            {
                positiveShift[0].y *= -1.0;
            }
            if(voxelcenter_local_xyz.x * CMX_CROSS.y - voxelcenter_local_xyz.y * CMX_CROSS.z < 0)
            {
                positiveShift[0].z *= -1.0;
            }
            if(voxelcenter_local_xyz.y * CMY_CROSS.z - voxelcenter_local_xyz.z * CMY_CROSS.y < 0)
            {
                positiveShift[1].x *= -1.0;
            }
            if(voxelcenter_local_xyz.z * CMY_CROSS.x - voxelcenter_local_xyz.x * CMY_CROSS.z < 0)
            {
                positiveShift[1].y *= -1.0;
            }
            if(voxelcenter_local_xyz.x * CMY_CROSS.y - voxelcenter_local_xyz.y * CMY_CROSS.z < 0)
            {
                positiveShift[1].z *= -1.0;
            }
            REAL3 minLocalVX = voxelcenter_local_xyz - positiveShift[0];
            REAL3 maxLocalVX = voxelcenter_local_xyz + positiveShift[0];
            REAL3 minLocalVY = voxelcenter_local_xyz - positiveShift[1];
            REAL3 maxLocalVY = voxelcenter_local_xyz + positiveShift[1];
            float PXLocalMin = PROJECTX0(CMF, minLocalVX);
            float PXLocalMax = PROJECTX0(CMF, maxLocalVX);
            float PYLocalMin = PROJECTY0(CMF, minLocalVY);
            float PYLocalMax = PROJECTY0(CMF, maxLocalVY);
            PILocalMin = INDEX(PXLocalMin);
            PILocalMax = INDEX(PXLocalMax);
            PJLocalMin = INDEX(PYLocalMin);
            PJLocalMax = INDEX(PYLocalMax);
        } else
        {
            offAxisPosition[0] = false;
            // TOREMOVE
            PILocalMin = -1;
            PILocalMax = -1;
            PJLocalMin = -1;
            PJLocalMax = -1;
            // TOREMOVE
        }
        if(PILocalMax < 0 || PILocalMin >= pdims.x || PJLocalMax < 0 || PJLocalMin >= pdims.y)
        {
            fullyOffProjectorPosition[0] = true;
            partlyOffProjectorPosition[0] = true;
        } else if(PILocalMin < 0 || PILocalMax >= pdims.x || PJLocalMin < 0
                  || PJLocalMax >= pdims.y)
        {
            fullyOffProjectorPosition[0] = false;
            partlyOffProjectorPosition[0] = true;
            projectorLocalRange[0] = max(0, PILocalMin);
            projectorLocalRange[1] = min(pdims.x - 1, PILocalMax);
            projectorLocalRange[2] = max(0, PJLocalMin);
            projectorLocalRange[3] = min(pdims.y - 1, PJLocalMax);
        } else
        {
            fullyOffProjectorPosition[0] = false;
            partlyOffProjectorPosition[0] = false;
            projectorLocalRange[0] = PILocalMin;
            projectorLocalRange[1] = PILocalMax;
            projectorLocalRange[2] = PJLocalMin;
            projectorLocalRange[3] = PJLocalMax;
        }
        // Prepare local memory
        if(offAxisPosition[0] == false) // Needed because its not solved otherwise
            if(!fullyOffProjectorPosition[0])
            {
                uint Irange = projectorLocalRange[1] - projectorLocalRange[0];
                uint Jrange = projectorLocalRange[3] - projectorLocalRange[2];
                uint FullLocalRange = Irange * Jrange;
                uint ILocalRange;
                if(FullLocalRange <= LOCALARRAYSIZE)
                {
                    ILocalRange = Irange; // How many columns fits to local memory
                } else
                {
                    ILocalRange = LOCALARRAYSIZE / Jrange;
                }
                projectorLocalRange[4] = ILocalRange;
                projectorLocalRange[5] = Jrange;
                projectorLocalRange[6]
                    = projectorLocalRange[0]; // Where current local array has start IRange
                mappedLocalRange = ILocalRange * Jrange;
                CML[0].s0123 = CMF.s0123 + projectorLocalRange[0] * CMF.s89ab;
                CML[0].s4567 = CMF.s4567 + projectorLocalRange[2] * CMF.s89ab;
                CML[0].s89ab = CMF.s89ab;
                for(int i = 0; i != mappedLocalRange; i++)
                {
                    localProjection[i] = 0.0f;
                }
            }
    }
    do
    {
        barrier(CLK_LOCAL_MEM_FENCE); // Cutting voxel projector
        // Start CVP
        if(offAxisPosition[0] == false) // Needed because its not solved otherwise
            if(!fullyOffProjectorPosition[0])
            {
                const uint IND = voxelIndex(i, j, k, vdims);
                const float voxelValue = volume[IND];
                const int2 Lpdims = { projectorLocalRange[4], projectorLocalRange[6] };
                bool dropVoxel = false;
                if(voxelValue == 0.0f)
                {
                    dropVoxel = true;
                }
#ifdef DROPINCOMPLETEVOXELS
                if(partlyOffProjectorPosition[0])
                {
                    int xindex = INDEX(PROJECTX0(CMF, voxelcenter_xyz));
                    int yindex = INDEX(PROJECTY0(CMF, voxelcenter_xyz));
                    if(xindex < 0 || yindex < 0 || xindex >= pdims.x || yindex >= pdims.y)
                    {
                        dropVoxel = true;
                    }
                }
#endif
                if(offAxisPosition[0])
                {
                    int Imax = INDEX(PROJECTX0(CMF, voxelcenter_xyz + positiveShift[0]));
                    int Imin = INDEX(PROJECTX0(CMF, voxelcenter_xyz - positiveShift[0]));
                    int Jmax = INDEX(PROJECTY0(CMF, voxelcenter_xyz + positiveShift[1]));
                    int Jmin = INDEX(PROJECTY0(CMF, voxelcenter_xyz - positiveShift[1]));
                    if(Imax < 0 || Jmax < 0 || Imin >= pdims.x || Jmin >= pdims.y)
                    {
                        dropVoxel = true;
                    }
                }
                if(!dropVoxel)
                {
                    const REAL voxelVolume = voxelSizes.x * voxelSizes.y * voxelSizes.z;
                    REAL sourceToVoxel_xyz_norm2 = dot(voxelcenter_xyz, voxelcenter_xyz);
#ifdef RELAXED
                    float value
                        = (voxelValue * voxelVolume * scalingFactor / sourceToVoxel_xyz_norm2);
#else
                    float value = (float)(voxelValue * voxelVolume * scalingFactor
                                          / sourceToVoxel_xyz_norm2);
#endif

                    // I assume that the volume point (x,y,z_1) projects to the same px as (x,y,z_2)
                    // for any z_1, z_2  This assumption is restricted to the voxel edges, where it
                    // holds very accurately  We project the rectangle that lies on the z midline of
                    // the voxel on the projector
                    REAL px00, px01, px10, px11;
                    REAL3 vx00, vx01, vx10, vx11;
                    vx00 = voxelcorner_xyz + voxelSizesF * (REAL3)(ZERO, ZERO, HALF);
                    vx01 = voxelcorner_xyz + voxelSizesF * (REAL3)(ONE, ZERO, HALF);
                    vx10 = voxelcorner_xyz + voxelSizesF * (REAL3)(ZERO, ONE, HALF);
                    vx11 = voxelcorner_xyz + voxelSizesF * (REAL3)(ONE, ONE, HALF);
                    px00 = PROJECTX0(CML[0], vx00);
                    px01 = PROJECTX0(CML[0], vx01);
                    px10 = PROJECTX0(CML[0], vx10);
                    px11 = PROJECTX0(CML[0], vx11);
                    // We now figure out the vertex that projects to minimum and maximum px
                    REAL pxx_min, pxx_max; // Minimum and maximum values of projector x coordinate
                    int max_PX,
                        min_PX; // Pixel to which are the voxels with minimum and maximum values are
                                // projected
                    // pxx_min = fmin(fmin(px00, px01), fmin(px10, px11));
                    // pxx_max = fmax(fmax(px00, px01), fmax(px10, px11));
                    REAL3* V_ccw[4]; // Point in which minimum is achieved and counter clock wise
                                     // points
                    // from the minimum voxel
                    REAL* PX_ccw[4]; // Point in which minimum is achieved and counter clock wise
                                     // points
                    // from the minimum voxel
                    if(offAxisPosition[0])
                    {
                        if(positiveShift[0].s0 > 0)
                        {
                            if(positiveShift[0].s1 > 0)
                            {
                                pxx_min = px11;
                                pxx_max = px00;
                                V_ccw[0] = &vx11;
                                V_ccw[1] = &vx10;
                                V_ccw[2] = &vx00;
                                V_ccw[3] = &vx01;
                                PX_ccw[0] = &px11;
                                PX_ccw[1] = &px10;
                                PX_ccw[2] = &px00;
                                PX_ccw[3] = &px01;
                            } else
                            {
                                pxx_min = px01;
                                pxx_max = px10;
                                V_ccw[0] = &vx01;
                                V_ccw[1] = &vx11;
                                V_ccw[2] = &vx10;
                                V_ccw[3] = &vx00;
                                PX_ccw[0] = &px01;
                                PX_ccw[1] = &px11;
                                PX_ccw[2] = &px10;
                                PX_ccw[3] = &px00;
                            }
                        } else
                        {
                            if(positiveShift[0].s1 > 0)
                            {
                                pxx_min = px10;
                                pxx_max = px01;
                                V_ccw[0] = &vx10;
                                V_ccw[1] = &vx00;
                                V_ccw[2] = &vx01;
                                V_ccw[3] = &vx11;
                                PX_ccw[0] = &px10;
                                PX_ccw[1] = &px00;
                                PX_ccw[2] = &px01;
                                PX_ccw[3] = &px11;
                            } else
                            {
                                // 00
                                pxx_min = px00;
                                V_ccw[0] = &vx00;
                                V_ccw[1] = &vx01;
                                V_ccw[2] = &vx11;
                                V_ccw[3] = &vx10;
                                PX_ccw[0] = &px00;
                                PX_ccw[1] = &px01;
                                PX_ccw[2] = &px11;
                                PX_ccw[3] = &px10;
                                pxx_max = px11;
                            }
                        }
                    } else
                    {
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
                    }
                    min_PX = convert_int_rtn(pxx_min + zeroPrecisionTolerance + HALF);
                    max_PX = convert_int_rtn(pxx_max - zeroPrecisionTolerance + HALF);
                    if(max_PX >= 0 && min_PX < Lpdims.x)
                    {
                        REAL3 vd1 = (*V_ccw[1]) - (*V_ccw[0]);
                        REAL3 vd3 = (*V_ccw[3]) - (*V_ccw[0]);
                        if(max_PX <= min_PX) // These indices are in the admissible range
                        {
#ifdef RELAXED
                            min_PX = convert_int_rtn(0.5f * (pxx_min + pxx_max) + 0.5f);
                            exactEdgeValuesF0(&projection[projectionOffset], CMF,
                                              (vx00 + vx11) / 2.0f, min_PX, value, voxelSizesF,
                                              pdims);
#else
                            min_PX = convert_int_rtn(0.5 * (pxx_min + pxx_max) + 0.5);
                            exactEdgeValues0(&projection[projectionOffset], CM, (vx00 + vx11) / 2.0,
                                             min_PX, value, voxelSizes, pdims);
#endif
                        } else
                        {
#ifdef RELAXED
                            float lastSectionSize, nextSectionSize, polygonSize;
                            float3 lastInt, nextInt, Int;
                            float factor;
#else
                            double lastSectionSize, nextSectionSize, polygonSize;
                            double3 lastInt, nextInt, Int;
                            double factor;
#endif
                            int I = max(-1, min_PX);
                            int I_STOP = min(max_PX, pdims.x);
                            // Section of the square that corresponds to the indices < i
                            // CCW and CW coordinates of the last intersection on the lines
                            // specified by the points in V_ccw
#ifdef RELAXED
                            lastSectionSize = exactIntersectionPointsF0_extended(
                                ((float)I) + 0.5f, V_ccw[0], V_ccw[1], V_ccw[2], V_ccw[3], vd1, vd3,
                                PX_ccw[0], PX_ccw[1], PX_ccw[2], PX_ccw[3], CMF, &lastInt);
#else
                            lastSectionSize = exactIntersectionPoints0_extended(
                                ((double)I) + 0.5, V_ccw[0], V_ccw[1], V_ccw[2], V_ccw[3], vd1, vd3,
                                PX_ccw[0], PX_ccw[1], PX_ccw[2], PX_ccw[3], CM, &lastInt);
#endif
                            if(I >= 0)
                            {
                                factor = value * lastSectionSize;
                                // insertEdgeValues(&projection[projectionOffset], CM, lastInt, I,
                                // factor, voxelSizes, pdims);
#ifdef RELAXED
                                exactEdgeValuesF0(&projection[projectionOffset], CMF, lastInt, I,
                                                  factor, voxelSizesF, pdims);
#else
                                exactEdgeValues0(&projection[projectionOffset], CM, lastInt, I,
                                                 factor, voxelSizes, pdims);
#endif
                            }
                            for(I = I + 1; I < I_STOP; I++)
                            {
#ifdef RELAXED
                                nextSectionSize = exactIntersectionPointsF0_extended(
                                    ((float)I) + 0.5f, V_ccw[0], V_ccw[1], V_ccw[2], V_ccw[3], vd1,
                                    vd3, PX_ccw[0], PX_ccw[1], PX_ccw[2], PX_ccw[3], CMF, &nextInt);
#else
                                nextSectionSize = exactIntersectionPoints0_extended(
                                    ((double)I) + 0.5, V_ccw[0], V_ccw[1], V_ccw[2], V_ccw[3], vd1,
                                    vd3, PX_ccw[0], PX_ccw[1], PX_ccw[2], PX_ccw[3], CM, &nextInt);
#endif
                                polygonSize = nextSectionSize - lastSectionSize;
                                Int = (nextSectionSize * nextInt - lastSectionSize * lastInt)
                                    / polygonSize;
                                factor = value * polygonSize;
#ifdef RELAXED
                                exactEdgeValuesF0(&projection[projectionOffset], CMF, Int, I,
                                                  factor, voxelSizesF, pdims);
#else
                                exactEdgeValues0(&projection[projectionOffset], CM, Int, I, factor,
                                                 voxelSizes, pdims);
#endif
                                lastSectionSize = nextSectionSize;
                                lastInt = nextInt;
                            }
                            if(I_STOP < pdims.x)
                            {
#ifdef RELAXED
                                polygonSize = 1.0f - lastSectionSize;
                                Int = ((*V_ccw[0] + *V_ccw[2]) * 0.5f - lastSectionSize * lastInt)
                                    / polygonSize;
                                factor = value * polygonSize;
                                exactEdgeValuesF0(&projection[projectionOffset], CMF, Int, I,
                                                  factor, voxelSizesF, pdims);
#else
                                polygonSize = 1.0 - lastSectionSize;
                                Int = ((*V_ccw[0] + *V_ccw[2]) * 0.5 - lastSectionSize * lastInt)
                                    / polygonSize;
                                factor = value * polygonSize;
                                localEdgeValues0(localProjection, CML[0], Int, I, factor,
                                                 voxelSizes, Lpdims);
#endif
                            }
                        }
                    }
                }
                // End CVP
            }
        if(offAxisPosition[0] == false) // Needed because its not solved otherwise
            if(cornerWorkItem)
            {
                projectorLocalRange[6] += projectorLocalRange[4];
            }
        barrier(CLK_LOCAL_MEM_FENCE); // Local to global copy
        if(offAxisPosition[0] == false) // Needed because its not solved otherwise
            if(cornerWorkItem)
            {
                CML[0].s0123 = CML[0].s0123 + projectorLocalRange[4] * CMF.s89ab;
                if(!fullyOffProjectorPosition[0])
                {
                    uint globalIndex = projectorLocalRange[6] * pdims.y + projectorLocalRange[2];
                    uint globalIndexIncrement = pdims.y - projectorLocalRange[5];
                    uint localIndex = 0;
                    for(int LI = 0; LI != projectorLocalRange[4]; LI++)
                    {
                        for(int LJ = 0; LJ != projectorLocalRange[5]; LJ++)
                        {
                            //                    localIndex = LI * projectorLocalRange[5] + LJ;
                            //                    globalIndex
                            //                        = (projectorLocalRange[6] + LI) * pdims.y +
                            //                        projectorLocalRange[2] + LJ;
                            AtomicAdd_g_f(projection + globalIndex, localProjection[localIndex]);
                            localIndex++;
                            globalIndex++;
                        }
                        globalIndex += globalIndexIncrement;
                    }
                    if(projectorLocalRange[6] < projectorLocalRange[1])
                    {
                        if(projectorLocalRange[1] - projectorLocalRange[6] < ILocalRange)
                        {
                            ILocalRange = projectorLocalRange[1] - projectorLocalRange[6];
                            projectorLocalRange[4] = ILocalRange;
                        }
                        mappedLocalRange = ILocalRange * Jrange;
                        for(int i = 0; i != mappedLocalRange; i++)
                        {
                            localProjection[i] = 0.0f;
                        }
                    }
                }
            }

    } while(projectorLocalRange[6] < projectorLocalRange[1]);
}
//==============================END projector_cvp_barrier.cl=====================================
