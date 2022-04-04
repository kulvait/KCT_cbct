//==============================include.cl=====================================
#ifndef zeroPrecisionTolerance
#define zeroPrecisionTolerance 1e-10
#endif

// Atomic operations
/** Atomic float addition.
 *
 * Function from
 * https://streamhpc.com/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved/.
 *
 *
 * @param source Pointer to the memory to perform atomic operation on.
 * @param operand Float to add.
 */
inline void AtomicAdd_g_f(volatile __global float* adr, const float v)
{
    union
    {
        unsigned int u32;
        float f32;
    } tmp, adrcatch;
    tmp.f32 = *adr;
    do
    {
        adrcatch.f32 = tmp.f32;
        tmp.f32 += v;
        tmp.u32 = atomic_cmpxchg((volatile __global unsigned int*)adr, adrcatch.u32, tmp.u32);
    } while(tmp.u32 != adrcatch.u32);
}

inline void AtomicAdd_l_f(volatile __local float* adr, const float v)
{
    union
    {
        unsigned int u32;
        float f32;
    } tmp, adrcatch;
    tmp.f32 = *adr;
    do
    {
        adrcatch.f32 = tmp.f32;
        tmp.f32 += v;
        tmp.u32 = atomic_cmpxchg((volatile __local unsigned int*)adr, adrcatch.u32, tmp.u32);
    } while(tmp.u32 != adrcatch.u32);
}

/** Atomic float minimum.
 *
 * Function from
 * https://streamhpc.com/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved/.
 *
 *
 * @param source Pointer to the memory to perform atomic operation on.
 * @param operand Float to add.
 */
inline void AtomicMin_g_f(volatile __global float* adr, const float v)
{
    union
    {
        unsigned int u32;
        float f32;
    } tmp, adrcatch;
    tmp.f32 = *adr;
    do
    {
        adrcatch.f32 = tmp.f32;
        tmp.f32 = min(v, tmp.f32);
        tmp.u32 = atomic_cmpxchg((volatile __global unsigned int*)adr, adrcatch.u32, tmp.u32);
    } while(tmp.u32 != adrcatch.u32);
}

// CVP Projector routines
//#define LOCALARRAYSIZE Theoretical maximum of 65536 bytes AMD, 49152 NVIDIA, 32768 Intel

#define DROPCENTEROFFPROJECTORVOXELS
// DROPINCOMPLETEVOXELS is not implemented the same in barrier implementation
#ifdef RELAXED
typedef float REAL;
typedef float2 REAL2;
typedef float3 REAL3;
typedef float8 REAL8;
typedef float16 REAL16;
//__constant float ONE=1.0f;
//__constant float HALF=0.5f;
//__constant float ZERO=0.0f;
#define ZERO 0.0f
#define ONESIXTH 0.16666667f
#define ONETHIRD 0.33333333f
#define HALF 0.5f
#define QUARTER 0.25f
#define TWOTHIRDS 0.66666666f
#define ONE 1.0f
#define THREE 3.0f
#define convert_REAL3(x) convert_float3(x)
#define LENGTH(x) fast_length(x)
#else
typedef double REAL;
typedef double2 REAL2;
typedef double3 REAL3;
typedef double8 REAL8;
typedef double16 REAL16;
//__constant double ONE=1.0;
//__constant double HALF=0.5;
//__constant double ZERO=0.0f;
#define ZERO 0.0
#define ONESIXTH 0.16666666666666667
#define ONETHIRD 0.3333333333333333
#define HALF 0.5
#define QUARTER 0.25
#define TWOTHIRDS 0.6666666666666666
#define ONE 1.0
#define THREE 3.0
#define convert_REAL3(x) convert_double3(x)
#define LENGTH(x) length(x)
#endif

#define PROJECTX0(CM, v0) dot(v0, CM.s012) / dot(v0, CM.s89a)

#define PROJECTY0(CM, v0) dot(v0, CM.s456) / dot(v0, CM.s89a)

#define PBPROJECTX(CM, v) dot(v, CM.s012) + CM.s3

#define PBPROJECTY(CM, v) dot(v, CM.s456) + CM.s7

#define INDEX(f) convert_int_rtn(f + HALF)

// For extracting edge values
#define EDGEMINMAX(PJ_min, PJ_max, v_min, v_min_minus_v_max_y)                                     \
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
        AtomicAdd_g_f(&projection[J], (lambda - lastLambda) * value);                              \
        lastLambda = lambda;                                                                       \
    }                                                                                              \
    AtomicAdd_g_f(&projection[PJ_max], (leastLambda - lastLambda) * value);

/** Projection of a volume point v0 onto X coordinate on projector.
 * No checks for boundaries.
 *
 * @param CM Projection camera matrix
 * @param v0 Volume point in source based coordinates
 * @param PX_out Output
 */
inline double projectX0(private const double16 CM, private const double3 v0)
{
    return dot(v0, CM.s012) / dot(v0, CM.s89a);
}

/** Projection of a volume point v0 onto Y coordinate on projector.
 * No checks for boundaries.
 *
 * @param CM Projection camera matrix
 * @param v0 Volume point in source based coordinates
 * @param PY_out Output
 */
inline double projectY0(private const double16 CM, private const double3 v0)
{
    return dot(v0, CM.s456) / dot(v0, CM.s89a);
}

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

inline int projectionIndex(private double16 CM, private double3 v, int2 pdims)
{
    double3 coord;
    int2 ind;
    coord.x = dot(v, CM.s012);
    coord.y = dot(v, CM.s456);
    coord.z = dot(v, CM.s89a);
    coord += CM.s37b;
    coord.x /= coord.z;
    coord.y /= coord.z;
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

inline int projectionIndex0(private const double16 CM, private const double3 v0, int2 const pdims)
{
    double PRZ;
    double2 PRJ;
    int2 IND;
    PRZ = dot(v0, CM.s89a);
    PRJ.x = dot(v0, CM.s012);
    PRJ.y = dot(v0, CM.s456);
    PRJ /= PRZ; // Scalar widening
                // https://www.informit.com/articles/article.aspx?p=1732873&seqNum=10
    IND.x = convert_int_rtn(PRJ.x + 0.5);
    IND.y = convert_int_rtn(PRJ.y + 0.5);
    if(IND.x >= 0 && IND.y >= 0 && IND.x < pdims.x && IND.y < pdims.y)
    {
        return IND.x * pdims.y + IND.y;
    } else
    {
        return -1;
    }
}

inline int projectionIndexF0(private const float16 CMF, private const float3 v0, int2 const pdims)
{
    float PRZ;
    float2 PRJ;
    int2 IND;
    PRZ = dot(v0, CMF.s89a);
    PRJ.x = dot(v0, CMF.s012);
    PRJ.y = dot(v0, CMF.s456);
    PRJ /= PRZ; // Scalar widening
                // https://www.informit.com/articles/article.aspx?p=1732873&seqNum=10
    IND.x = convert_int_rtn(PRJ.x + 0.5);
    IND.y = convert_int_rtn(PRJ.y + 0.5);
    if(IND.x >= 0 && IND.y >= 0 && IND.x < pdims.x && IND.y < pdims.y)
    {
        return IND.x * pdims.y + IND.y;
    } else
    {
        return -1;
    }
}

#ifdef ELEVATIONCORRECTION

/**
 * This shall be correct implementation, is here just for reference, because its slow and probably
 * faster what is below.
 */
void inline exactEdgeValues0ElevationCorrectionFORK(
    global float* projection,
    private REAL16 CM,
    private REAL3 v,
    private int PX,
    private REAL value,
    private REAL3 voxelSizes,
    private int2 pdims,
    private REAL corlambda) // corlambda is scaled to the size of lambda
{
    projection = projection + PX * pdims.y;
    const REAL3 distanceToEdge = (REAL3)(ZERO, ZERO, HALF * voxelSizes.s2);
    const REAL3 v_plus = v + distanceToEdge;
    const REAL3 v_minus = v - distanceToEdge;
    const REAL PY_plus = PROJECTY0(CM, v_plus);
    const REAL PY_minus = PROJECTY0(CM, v_minus);
    // const REAL3 v_diff = v_down - v_up;
    int PJ_min, PJ_max;
    int J;
    REAL3 v_min;
    REAL v_min_minus_v_max_y;
    if(PY_plus < PY_minus) // Classical geometry setup
    {
        PJ_max = convert_int_rtn(PY_minus + HALF);
        PJ_min = convert_int_rtn(PY_plus + HALF);
        v_min = v_plus;
        v_min_minus_v_max_y = voxelSizes.s2;
    } else
    {
        PJ_max = convert_int_rtn(PY_plus + HALF);
        PJ_min = convert_int_rtn(PY_minus + HALF);
        v_min = v_minus;
        v_min_minus_v_max_y = -voxelSizes.s2;
    }
    REAL lambda;
    // To model v_min + lambda (v_max - v_min)
    REAL lastLambda = ZERO;
    REAL leastLambda;
    REAL3 Fvector;
    // We will correct just in the regions [-corlambda, corlambda] and [1-corlambda, 1+corlambda]
    // PY = (dot(CM.s456, v_min) - lambda * CM.s6 * v_min_minus_v_max_y)/(dot(CM.s89a, v_min) -
    // lambda *CM.sa * v_min_minus_v_max_y)  PY = (A + lambda * B) / (C + lamdba * D), where
    REAL A = dot(CM.s456, v_min);
    REAL B = -CM.s6 * v_min_minus_v_max_y;
    REAL C = dot(CM.s89a, v_min);
    REAL D = -CM.sa * v_min_minus_v_max_y;
    REAL PY_min_cor_min = (A - corlambda * B) / (C - corlambda * D);
    int PJ_min_cor_min = convert_int_rtn(PY_min_cor_min + HALF);
    REAL PY_min_cor_max = (A + corlambda * B) / (C + corlambda * D);
    int PJ_min_cor_max = convert_int_rtn(PY_min_cor_max + HALF);
    REAL PY_max_cor_min = (A + (ONE - corlambda) * B) / (C - (ONE - corlambda) * D);
    int PJ_max_cor_min = convert_int_rtn(PY_max_cor_min + HALF);
    REAL PY_max_cor_max = (A + (ONE + corlambda) * B) / (C + (ONE + corlambda) * D);
    int PJ_max_cor_max = convert_int_rtn(PY_max_cor_max + HALF);
    if(PJ_max >= pdims.y) // Here usually no correction since it will be compensated
    {
        PJ_max = pdims.y - 1;
        Fvector = CM.s456 - (PJ_max + HALF) * CM.s89a;
        leastLambda = dot(v_min, Fvector) / (v_min_minus_v_max_y * Fvector.s2);
    } else
    {
        leastLambda = ONE;
    }
    if(PJ_min < 0)
    {
        J = 0;
        Fvector = CM.s456 + HALF * CM.s89a;
        lastLambda = dot(v_min, Fvector) / (v_min_minus_v_max_y * Fvector.s2);
    } else
    {
        J = PJ_min;
        Fvector = CM.s456 - (J - HALF) * CM.s89a;
    }
    REAL corQuarterMultiplier = QUARTER / corlambda;
    REAL corFactor;
    bool correctMin = (PJ_min_cor_min != PJ_min_cor_max && PJ_min_cor_max >= 0);
    bool correctMax = (PJ_max_cor_min != PJ_max_cor_max && PJ_max_cor_min < pdims.y);
    if(!correctMin && !correctMax) // Do not correct if they
                                   // map to single pixel or
                                   // outside detector as the
                                   // standard compensation
                                   // mechanism is in place
    {
        for(; J < PJ_max; J++)
        {
            Fvector -= CM.s89a;
            lambda = dot(v_min, Fvector) / (v_min_minus_v_max_y * Fvector.s2);
            AtomicAdd_g_f(&projection[J], (lambda - lastLambda) * value);
            lastLambda = lambda;
        }
        AtomicAdd_g_f(&projection[PJ_max], (leastLambda - lastLambda) * value);
    } else if(!correctMax) // correctMin
    {
        REAL3 Qvector;
        REAL lastCorLambda, leastCorlambda, lambdaCor;
        if(PJ_min_cor_min < 0)
        {
            PJ_min_cor_min = 0;
            Qvector = CM.s456 + HALF * CM.s89a;
            lastCorLambda = dot(v_min, Qvector) / (v_min_minus_v_max_y * Qvector.s2);
        } else
        {
            lastCorLambda = -corlambda;
            Qvector = CM.s456 - (PJ_min_cor_min - HALF) * CM.s89a;
        }
        for(; PJ_min_cor_min < J; PJ_min_cor_min++)
        {
            Qvector -= CM.s89a;
            lambda = dot(v_min, Qvector)
                / (v_min_minus_v_max_y * Qvector.s2); // Shall be negative here as we are before J
            corFactor = HALF * (lambda - lastCorLambda)
                + corQuarterMultiplier * (lambda * lambda - lastCorLambda * lastCorLambda);
            AtomicAdd_g_f(projection + PJ_min_cor_min, corFactor * value);
            lastCorLambda = lambda;
        }
        for(; J < PJ_max; J++)
        {
            Fvector -= CM.s89a;
            lambda = dot(v_min, Fvector) / (v_min_minus_v_max_y * Fvector.s2);
            if(lastLambda > corlambda)
            {
                AtomicAdd_g_f(&projection[J], (lambda - lastLambda) * value);
                lastLambda = lambda;
            } else
            {
                if(lambda < corlambda)
                {
                    corFactor = HALF * (lambda - lastCorLambda)
                        + corQuarterMultiplier * (lambda * lambda - lastCorLambda * lastCorLambda);
                    lastCorLambda = lambda;
                } else
                {
                    corFactor = HALF * (corlambda - lastCorLambda) + (lambda - corlambda)
                        + corQuarterMultiplier
                            * (corlambda * corlambda - lastCorLambda * lastCorLambda);
                }
                AtomicAdd_g_f(&projection[J], value * corFactor);
                lastLambda = lambda;
            }
        }
        if(lastLambda > corlambda)
        {
            AtomicAdd_g_f(&projection[PJ_max], (leastLambda - lastLambda) * value);
        } else
        {
            if(leastLambda
               < corlambda) // Highly unprobable and I will not correct further in this situation
            {
                corFactor = HALF * (leastLambda - lastCorLambda)
                    + corQuarterMultiplier
                        * (leastLambda * leastLambda - lastCorLambda * lastCorLambda);
            } else
            {
                corFactor = HALF * (corlambda - lastCorLambda) + (leastLambda - corlambda)
                    + corQuarterMultiplier
                        * (corlambda * corlambda - lastCorLambda * lastCorLambda);
            }
            AtomicAdd_g_f(&projection[PJ_max], corFactor * value);
        }
    } else if(!correctMin) // correctmax
    {
        REAL3 Qvector;
        REAL lastCorMaxLambdaShifted, leastCorMaxLambdaShifted, corMaxLambdaShifted;
        REAL corFactor;
        REAL lambdaShifted, lastLambdaShifted;
        if(PJ_max_cor_max >= pdims.y)
        {
            PJ_max_cor_max = pdims.y - 1;
            Qvector = CM.s456 - (PJ_max_cor_max + HALF) * CM.s89a;
            leastCorMaxLambdaShifted
                = (dot(v_min, Qvector) / (v_min_minus_v_max_y * Qvector.s2)) - ONE;
        } else
        {
            leastCorMaxLambdaShifted = corlambda;
        }
        corMaxLambdaShifted = -corlambda;
        for(; J < PJ_max_cor_max; J++)
        {
            Fvector -= CM.s89a;
            lambda = dot(v_min, Fvector) / (v_min_minus_v_max_y * Fvector.s2);
            lambdaShifted = lambda - ONE;
            lastLambdaShifted = lastLambda - ONE;
            if(lambdaShifted > -corlambda)
            {
                if(lastLambdaShifted > -corlambda)
                {
                    corFactor = HALF * (lambda - lastLambda)
                        + corQuarterMultiplier
                            * (lastLambdaShifted * lastLambdaShifted
                               - lambdaShifted * lambdaShifted);
                    corMaxLambdaShifted = lambdaShifted;
                } else
                {
                    corFactor = (-corlambda - lastLambdaShifted)
                        + HALF * (lambdaShifted + corlambda)
                        + corQuarterMultiplier
                            * (corlambda * corlambda - lambdaShifted * lambdaShifted);
                    corMaxLambdaShifted = lambdaShifted;
                }
            } else
            {
                corFactor = lambda - lastLambda;
            }
            AtomicAdd_g_f(&projection[J], corFactor * value);
            lastLambda = lambda;
        }
        lastLambdaShifted = lastLambda - ONE;
        if(lastLambdaShifted > -corlambda)
        {
            corFactor = HALF * (leastCorMaxLambdaShifted - lastLambdaShifted)
                + corQuarterMultiplier
                    * (lastLambdaShifted * lastLambdaShifted
                       - leastCorMaxLambdaShifted * leastCorMaxLambdaShifted);
        } else
        {
            corFactor = (-corlambda - lastLambdaShifted)
                + HALF * (leastCorMaxLambdaShifted + corlambda)
                + corQuarterMultiplier
                    * (corlambda * corlambda - leastCorMaxLambdaShifted * leastCorMaxLambdaShifted);
        }
        AtomicAdd_g_f(projection + PJ_max_cor_max, corFactor * value);
    } else // correctMin and correctMax
    {
        REAL lambdaShifted, lastLambdaShifted, lastCorMaxLambdaShifted, leastCorMaxLambdaShifted;
        if(PJ_max_cor_max >= pdims.y) // Here usually no correction since it will be compensated
        {
            PJ_max = pdims.y - 1;
            Fvector = CM.s456 - (PJ_max + HALF) * CM.s89a;
            leastLambda = dot(v_min, Fvector) / (v_min_minus_v_max_y * Fvector.s2);
            leastCorMaxLambdaShifted = leastLambda - ONE;
        } else
        {
            leastLambda = ONE + corlambda;
            leastCorMaxLambdaShifted = corlambda;
        }
        if(PJ_min_cor_min < 0)
        {
            J = 0;
            Fvector = CM.s456 + HALF * CM.s89a;
            lastLambda = dot(v_min, Fvector) / (v_min_minus_v_max_y * Fvector.s2);
        } else
        {
            J = PJ_min_cor_min;
            Fvector = CM.s456 - (J - HALF) * CM.s89a;
            lastLambda = -corlambda;
        }
        for(; J < PJ_max_cor_max; J++)
        {
            Fvector -= CM.s89a;
            lambda = dot(v_min, Fvector) / (v_min_minus_v_max_y * Fvector.s2);
            lambdaShifted = lambda - ONE;
            if(lastLambda > corlambda && lambdaShifted < -corlambda)
            {
                corFactor = lambda - lastLambda;
            } else if(lambdaShifted > -corlambda)
            {
                if(lastLambdaShifted > -corlambda)
                {
                    corFactor = HALF * (lambda - lastLambda)
                        + corQuarterMultiplier
                            * (lastLambdaShifted * lastLambdaShifted
                               - lambdaShifted * lambdaShifted);
                    lastCorMaxLambdaShifted = lambdaShifted;
                } else
                {
                    corFactor = (-corlambda - lastLambdaShifted)
                        + HALF * (lambdaShifted + corlambda)
                        + corQuarterMultiplier
                            * (corlambda * corlambda - lambdaShifted * lambdaShifted);
                    lastCorMaxLambdaShifted = lambdaShifted;
                }
            } else // lastLambda < corlambda
            {
                if(lambda < corlambda)
                {
                    corFactor = HALF * (lambda - lastLambda)
                        + corQuarterMultiplier * (lambda * lambda - lastLambda * lastLambda);
                } else
                {
                    corFactor = HALF * (corlambda - lastLambda) + (lambda - corlambda)
                        + corQuarterMultiplier * (corlambda * corlambda - lastLambda * lastLambda);
                }
            }
            AtomicAdd_g_f(&projection[J], corFactor * value);
            lastLambda = lambda;
            lastLambdaShifted = lastLambda - ONE;
        }
        if(lastLambdaShifted > -corlambda)
        {
            corFactor = HALF * (leastCorMaxLambdaShifted - lastLambdaShifted)
                + corQuarterMultiplier
                    * (lastLambdaShifted * lastLambdaShifted
                       - leastCorMaxLambdaShifted * leastCorMaxLambdaShifted);
        } else
        {
            corFactor = (-corlambda - lastLambdaShifted)
                + HALF * (leastCorMaxLambdaShifted + corlambda)
                + corQuarterMultiplier
                    * (corlambda * corlambda - leastCorMaxLambdaShifted * leastCorMaxLambdaShifted);
        }
        AtomicAdd_g_f(projection + PJ_max_cor_max, corFactor * value);
    }
}

// printf routines for debugging are in bbdb59
void inline exactEdgeValues0ElevationCorrection(
    global float* projection,
    private REAL16 CM,
    private REAL3 v,
    private int PX,
    private REAL value,
    private REAL3 voxelSizes,
    private int2 pdims,
    private REAL corLength) // corLength is scaled to the size of lambda
{
    projection = projection + PX * pdims.y;
    const REAL3 distanceToEdge = (REAL3)(ZERO, ZERO, HALF * voxelSizes.s2);
    const REAL3 v_plus = v + distanceToEdge;
    const REAL3 v_minus = v - distanceToEdge;
    const REAL PY_plus = PROJECTY0(CM, v_plus);
    const REAL PY_minus = PROJECTY0(CM, v_minus);
    // const REAL3 v_diff = v_down - v_up;
    int PJ_min, PJ_max;
    int J;
    REAL3 v_min;
    REAL v_min_minus_v_max_y;
    if(PY_plus < PY_minus) // Classical geometry setup
    {
        PJ_max = convert_int_rtn(PY_minus + HALF);
        PJ_min = convert_int_rtn(PY_plus + HALF);
        v_min = v_plus;
        v_min_minus_v_max_y = voxelSizes.s2;
    } else
    {
        PJ_max = convert_int_rtn(PY_plus + HALF);
        PJ_min = convert_int_rtn(PY_minus + HALF);
        v_min = v_minus;
        v_min_minus_v_max_y = -voxelSizes.s2;
    }
    // We will correct just in the regions [-corLength, corLength] and [1-corLength, 1+corLength]
    REAL corQuarterMultiplier;
    if(corLength < zeroPrecisionTolerance)
    {
        corQuarterMultiplier = ZERO;
        corLength = ZERO;
    } else
    {
        corQuarterMultiplier = QUARTER / corLength;
    }
    // PY = (dot(CM.s456, v_min) - lambda * CM.s6 * v_min_minus_v_max_y)/(dot(CM.s89a, v_min) -
    // lambda *CM.sa * v_min_minus_v_max_y)
    // PY = (A + lambda * B) / (C + lamdba * D), where
    REAL A = dot(CM.s456, v_min);
    REAL B = -CM.s6 * v_min_minus_v_max_y;
    REAL C = dot(CM.s89a, v_min);
    REAL D = -CM.sa * v_min_minus_v_max_y;
    REAL PY_min_cor_min = (A - corLength * B) / (C - corLength * D);
    int PJ_min_cor_min = convert_int_rtn(PY_min_cor_min + HALF);
    // REAL PY_min_cor_max = (A + corLength * B) / (C + corLength * D);
    // int PJ_min_cor_max = convert_int_rtn(PY_min_cor_max + HALF);
    // REAL PY_max_cor_min = (A + (ONE - corLength) * B) / (C - (ONE - corLength) * D);
    // int PJ_max_cor_min = convert_int_rtn(PY_max_cor_min + HALF);
    REAL PY_max_cor_max = (A + (ONE + corLength) * B) / (C + (ONE + corLength) * D);
    int PJ_max_cor_max = convert_int_rtn(PY_max_cor_max + HALF);

    if(PJ_max_cor_max > 0 && PJ_min_cor_min < pdims.y)
    {
        REAL lambda;
        // To model v_min + lambda (v_max - v_min)
        REAL lastLambda = ZERO;
        REAL leastLambda;
        REAL3 Fvector;
        REAL corFactor;
        REAL lambdaShifted, lastLambdaShifted, lastCorMaxLambdaShifted, leastCorMaxLambdaShifted;
        if(PJ_max_cor_max >= pdims.y)
        {
            PJ_max = pdims.y - 1;
            Fvector = CM.s456 - (PJ_max + HALF) * CM.s89a;
            leastLambda = dot(v_min, Fvector) / (v_min_minus_v_max_y * Fvector.s2);
            leastCorMaxLambdaShifted = leastLambda - ONE;
        } else
        {
            leastLambda = ONE + corLength;
            leastCorMaxLambdaShifted = corLength;
            PJ_max = PJ_max_cor_max;
        }
        if(PJ_min_cor_min < 0)
        {
            J = 0;
            Fvector = CM.s456 + HALF * CM.s89a;
            lastLambda = dot(v_min, Fvector) / (v_min_minus_v_max_y * Fvector.s2);
            lastLambdaShifted = lastLambda - ONE;
        } else
        {
            J = PJ_min_cor_min;
            Fvector = CM.s456 - (J - HALF) * CM.s89a;
            lastLambda = -corLength;
            lastLambdaShifted = lastLambda - ONE;
        }
        for(; J < PJ_max; J++)
        {
            Fvector -= CM.s89a;
            lambda = dot(v_min, Fvector) / (v_min_minus_v_max_y * Fvector.s2);
            lambdaShifted = lambda - ONE;
            if(lastLambda >= corLength && lambdaShifted <= -corLength)
            {
                corFactor = lambda - lastLambda;
            } else if(lambdaShifted > -corLength)
            {
                if(lastLambdaShifted > -corLength)
                {
                    corFactor = HALF * (lambda - lastLambda)
                        + corQuarterMultiplier
                            * (lastLambdaShifted * lastLambdaShifted
                               - lambdaShifted * lambdaShifted);
                    lastCorMaxLambdaShifted = lambdaShifted;
                } else
                {
                    corFactor = (-corLength - lastLambdaShifted)
                        + HALF * (lambdaShifted + corLength)
                        + corQuarterMultiplier
                            * (corLength * corLength - lambdaShifted * lambdaShifted);
                    lastCorMaxLambdaShifted = lambdaShifted;
                }
            } else // lastLambda < corLength
            {
                if(lambda < corLength)
                {
                    corFactor = HALF * (lambda - lastLambda)
                        + corQuarterMultiplier * (lambda * lambda - lastLambda * lastLambda);
                } else
                {
                    corFactor = HALF * (corLength - lastLambda) + (lambda - corLength)
                        + corQuarterMultiplier * (corLength * corLength - lastLambda * lastLambda);
                }
            }
            AtomicAdd_g_f(&projection[J], corFactor * value);
            lastLambda = lambda;
            lastLambdaShifted = lastLambda - ONE;
        }
        if(corLength == ZERO || leastCorMaxLambdaShifted < -corLength)
        {
            corFactor = leastCorMaxLambdaShifted - lastLambdaShifted;
        } else if(lastLambdaShifted > -corLength)
        {
            corFactor = HALF * (leastCorMaxLambdaShifted - lastLambdaShifted)
                + corQuarterMultiplier
                    * (lastLambdaShifted * lastLambdaShifted
                       - leastCorMaxLambdaShifted * leastCorMaxLambdaShifted);
        } else
        {
            corFactor = (-corLength - lastLambdaShifted)
                + HALF * (leastCorMaxLambdaShifted + corLength)
                + corQuarterMultiplier
                    * (corLength * corLength - leastCorMaxLambdaShifted * leastCorMaxLambdaShifted);
        }
        AtomicAdd_g_f(projection + PJ_max, corFactor * value);
    }
}
#endif

void inline exactEdgeValues0(global float* projection,
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
    // const REAL3 v_diff = v_down - v_up;
    int PJ_up = convert_int_rtn(PY_up + HALF);
    int PJ_down = convert_int_rtn(PY_down + HALF);

    int J;
    REAL lambda;
    REAL lastLambda = ZERO;
    REAL leastLambda;
    REAL3 Fvector;
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

// clang-format off
// const REAL vd1 = v1->x - v0->x; Nonzero x part
// const REAL vd3 = v3->y - v0->y; Nonzero y part
// polygon center of mass https://www.efunda.com/math/areas/Trapezoid.cfm to see
// when I have polygon with p base and q top, then Tx=(p*p+p*q+q*q)/(3*(p+q)) Ty=(p+2q)/(3*(p+q))
// when I take w=q/(3*(p+q)) I will have
// Tx = p/3 + q*w Ty= 1/3 + w
// Computation of CENTROID of rectangle with triangle cutout with p, q sides
// NAREA_complement=(p*q)/2
// NAREA = 1-NAREA_complement
// In cutout corner
// CENTROID = (0.5-p*NAREA_complement/3)/NAREA = 0.5 + (NAREA_COMPLEMENT/NAREA)(0.5 -(p/3)) = 0.5/NAREA - p*(ONETHIRD*NAREA_complement/NAREA)
// Outside cutout corner
// CENTROID = 0.5 - 0.5 * NAREA_COMPLEMENT/NAREA + (p/3)* NAREA_COMPLEMENT/NAREA
// example when p_complement in oposite x corner and y coordinates of the corner are the same
// w = HALF / NAREA;
// w_complement = ONETHIRD * NAREA_complement / NAREA;
// CENTROID = (REAL2)(vd1 * (ONE - w + p_complement * w_complement), vd3 * (w - q * w_complement));
// When trying to find lambda(PX) tak, že PROJ(lambda(PX)) = PX on line v0 + lambda d
// lambda = dot(v, Fvector)/-dot(d, Fvector)
// clang-format on
// When FproductVD1NEG or FproductVD3NEG are zero then basically all the segment is projected to
// given index but we need to achieve p \in [0,1] and q\in[0,1] so that we have to solve it
// individually
// I put it int the way to greedy select the biggest area possible
inline REAL exactIntersectionPolygons0(const REAL PX,
                                       const REAL vd1,
                                       const REAL vd3,
                                       const REAL3* v0,
                                       const REAL* PX_xyx0,
                                       const REAL* PX_xyx1,
                                       const REAL* PX_xyx2,
                                       const REAL* PX_xyx3,
                                       const REAL16 CM,
                                       REAL3 voxelSizes,
                                       REAL2* centroid,
                                       REAL* llength)
{
    const REAL3 Fvector = CM.s012 - PX * CM.s89a;
    REAL Fproduct = dot(*v0, Fvector);
    REAL FproductVD1NEG = -vd1 * Fvector.x;
    REAL FproductVD3NEG = -vd3 * Fvector.y;

    REAL p, q, p_complement;
    REAL w, wcomplement, NAREA_complement;
    REAL LINELENGTH, NAREA;
    REAL2 CENTROID;
    if(PX < (*PX_xyx1))
    {
        if(FproductVD1NEG)
            p = Fproduct / FproductVD1NEG; // From v0 to v1
        else
            p = ONE;
        if(PX < (*PX_xyx3))
        {
            if(FproductVD3NEG)
                q = Fproduct / FproductVD3NEG; // From v0 to v3
            else
                q = ONE;
            CENTROID = (REAL2)(p * vd1, q * vd3);
            LINELENGTH = LENGTH(CENTROID);
            (*llength) = LINELENGTH;
            CENTROID *= ONETHIRD;
            NAREA = HALF * p * q;
            (*centroid) = v0->s01 + CENTROID;
            return NAREA;
        } else if(PX < (*PX_xyx2))
        {
            if(FproductVD1NEG)
                q = (Fproduct - FproductVD3NEG) / FproductVD1NEG; // From v3 to v2
            else
                q = ONE;
            CENTROID = (REAL2)((q - p) * vd1, voxelSizes.y);
            LINELENGTH = LENGTH(CENTROID);
            (*llength) = LINELENGTH;
            NAREA = THREE * (p + q);
            if(NAREA > ZERO)
            {
                w = q / NAREA;
                CENTROID = (REAL2)(vd1 * (ONETHIRD * p + q * w), vd3 * (ONETHIRD + w));
            } else
            {
                CENTROID = (REAL2)(ZERO, vd3 * HALF);
            }
            (*centroid) = v0->s01 + CENTROID;
            NAREA = ONESIXTH * NAREA;
            return NAREA;
        } else
        {
            p_complement = ONE - p;
            if(FproductVD3NEG)
                q = (Fproduct - FproductVD1NEG) / FproductVD3NEG; // From v1 to v2
            else
                q = ZERO;
            CENTROID = (REAL2)(p_complement * vd1, q * vd3);
            LINELENGTH = LENGTH(CENTROID);
            (*llength) = LINELENGTH;
            NAREA_complement = HALF * p_complement * q;
            NAREA = ONE - NAREA_complement;
            w = HALF / NAREA;
            wcomplement = ONETHIRD * NAREA_complement / NAREA;
            CENTROID = (REAL2)(vd1 * (ONE - w + p_complement * wcomplement),
                               vd3 * (w - q * wcomplement));
            (*centroid) = v0->s01 + CENTROID;
            return NAREA;
        }
    } else if(PX < (*PX_xyx2))
    {
        if(FproductVD3NEG)
            p = (Fproduct - FproductVD1NEG - FproductVD3NEG) / -FproductVD3NEG; // v2 to v1
        else
            p = ZERO;
        if(PX < (*PX_xyx3))
        {
            p = ONE - p; // v1 to v2
            if(FproductVD3NEG)
                q = Fproduct / FproductVD3NEG; // v0 to v3
            else
                q = ONE;
            CENTROID = (REAL2)(voxelSizes.x, (q - p) * vd3);
            LINELENGTH = LENGTH(CENTROID);
            (*llength) = LINELENGTH;
            NAREA = THREE * (p + q);
            if(NAREA > ZERO)
            {
                w = p / NAREA;
                CENTROID = (REAL2)(vd1 * (ONETHIRD + w), vd3 * (p * w + ONETHIRD * q));
            } else
            {
                CENTROID = (REAL2)(vd1 * HALF, ZERO);
            }
            (*centroid) = v0->s01 + CENTROID;
            NAREA = ONESIXTH * NAREA;
            return NAREA;
        } else
        {
            if(FproductVD1NEG)
                q = (Fproduct - FproductVD1NEG - FproductVD3NEG) / -FproductVD1NEG; // v2 to v3
            else
                q = ZERO;
            CENTROID = (REAL2)(q * vd1, p * vd3);
            LINELENGTH = LENGTH(CENTROID);
            (*llength) = LINELENGTH;
            NAREA_complement = HALF * p * q;
            NAREA = ONE - NAREA_complement;
            w = HALF / NAREA;
            wcomplement = ONETHIRD * NAREA_complement / NAREA;
            CENTROID
                = (REAL2)(vd1 * (ONE - w + q * wcomplement), vd3 * (ONE - w + p * wcomplement));
            (*centroid) = v0->s01 + CENTROID;
            return NAREA;
        }
    } else if(PX >= *PX_xyx3)
    {
        NAREA = ONE;
        LINELENGTH = ZERO;
        (*llength) = LINELENGTH;
        CENTROID = (REAL2)(HALF * vd1, HALF * vd3);
        (*centroid) = v0->s01 + CENTROID;
        return ONE;

    } else
    {
        if(FproductVD3NEG)
            p = (Fproduct - FproductVD3NEG) / -FproductVD3NEG; // v3 to v0
        else
            p = ZERO;
        if(FproductVD1NEG)
            q = (Fproduct - FproductVD3NEG) / FproductVD1NEG; // v3 to v2
        else
            q = ZERO;
        CENTROID = (REAL2)(p * vd3, q * vd1);
        LINELENGTH = LENGTH(CENTROID);
        (*llength) = LINELENGTH;
        NAREA_complement = HALF * p * q;
        NAREA = ONE - NAREA_complement;
        w = HALF / NAREA;
        wcomplement = ONETHIRD * NAREA_complement / NAREA;
        CENTROID = (REAL2)(vd1 * (w - q * wcomplement), vd3 * (ONE - w + p * wcomplement));
        (*centroid) = v0->s01 + CENTROID;
        return NAREA;
    }
}

inline REAL exactIntersectionPoints0_extended(const REAL PX,
                                              const REAL3* v0,
                                              const REAL3* v1,
                                              const REAL3* v2,
                                              const REAL3* v3,
                                              const REAL vd1,
                                              const REAL vd3,
                                              const REAL* PX_xyx0,
                                              const REAL* PX_xyx1,
                                              const REAL* PX_xyx2,
                                              const REAL* PX_xyx3,
                                              const REAL16 CM,
                                              REAL3* centroid)
{
    const REAL3 Fvector = CM.s012 - PX * CM.s89a;
    // const REAL3 vd1 = (*v1) - (*v0);//Nonzero x part
    // const REAL3 vd3 = (*v3) - (*v0);//Nonzero y part
    REAL Fproduct, FproductVD;
    REAL p, q;
    REAL A, w, wcomplement;
    if(PX < (*PX_xyx1))
    {
        Fproduct = -dot(*v0, Fvector);
        FproductVD = vd1 * Fvector.x; // VD1
        p = Fproduct / FproductVD; // v0+p*(v1-v0)
        if(PX < (*PX_xyx3))
        {
            q = Fproduct / (vd3 * Fvector.y);
            (*centroid) = (*v0) + (REAL3)(ONETHIRD * p * vd1, ONETHIRD * q * vd3, ZERO);
            return HALF * p * q;
        } else if(PX < (*PX_xyx2))
        {
            q = -dot(*v3, Fvector) / FproductVD;
            A = THREE * (p + q);
            (*centroid) = (*v0);
            if(A > ZERO) // Due to rounding errors equality might happen producing nan
            {
                // w = p / A;
                //(*centroid) = (*v0)
                //    + mad(p, mad(-ONE / 6.0, w, 2.0 / 3.0), mad(-q, w, q) / 3.0) * (vd1)
                //    + mad(-ONE / 6.0, w, 2.0 / 3.0) * (vd3);
                //(*centroid) = (*v0) + (p * (2.0 / 3.0 - w / 6.0) + q * (1 - w) / 3.0) * (vd1)
                //    + (2.0 / 3.0 - w / 6.0) * (vd3);
                // wcomplement = TWOTHIRDS - ONESIXTH * w;
                // centroid->s01 += (REAL2)((p * wcomplement + ONETHIRD * q * (ONE - w)) * vd1,
                //                         wcomplement * vd3);
                //(*centroid) = (*v0) + (p * wcomplement + ONETHIRD * q * (ONE - w)) * (vd1)
                //    + wcomplement * (vd3);
                // See https://www.efunda.com/math/areas/Trapezoid.cfm to see that these formulas
                // are correct
                w = q / A;
                centroid->s01 += (REAL2)(vd1 * (ONETHIRD * p + q * w), vd3 * (ONETHIRD + w));
            } else
            {
                centroid->s01 += (REAL2)(ZERO, vd3 * HALF);
            }
            return ONESIXTH * A;
        } else
        {
            p = ONE - p;
            q = -dot(*v1, Fvector) / (vd3 * Fvector.y);
            A = ONE - HALF * p * q;
            // w = ONE / A;
            //(*centroid) = (*v0) - mad(HALF, w, mad(p, -w, p) / 3.0) * vd1
            //    + mad(HALF, w, mad(q, -w, q) / 3.0) * vd3;
            //(*centroid) = (*v1) - (HALF * w + (p * (1 - w)) / 3.0) * vd1
            //    + (HALF * w + (q * (1 - w)) / 3.0) * vd3;
            w = HALF / A;
            wcomplement = TWOTHIRDS * (HALF - w); //=-ONETHIRD*(ONE-A)/A
            (*centroid) = (*v1);
            centroid->s01 += (REAL2)((w + p * wcomplement) * -vd1, (w + q * wcomplement) * vd3);
            return A;
        }
    } else if(PX < (*PX_xyx2))
    {
        Fproduct = dot(*v2, Fvector);
        FproductVD = vd3 * Fvector.y;
        p = Fproduct / FproductVD; // V2 + p * (V1-V2)
        if(PX < (*PX_xyx3))
        {
            (*centroid) = (*v0);
            p = ONE - p; // V1 + p * (V2-V1)
            q = -dot(*v0, Fvector) / FproductVD; // V0 + q (V3-V0)
            A = THREE * (p + q);
            if(A > ZERO) // Due to rounding errors equality might happen producing nan
            {
                // w = q / A;
                //(*centroid) = (*v0)
                //    + mad(q, mad(-ONE / 6.0, w, 2.0 / 3.0), mad(-p, w, p) / 3.0) * (vd3)
                //    + mad(-ONE / 6.0, w, 2.0 / 3.0) * (vd1);
                //(*centroid) = (*v0) + (q * (2.0 / 3.0 - w / 6.0) + p * (1 - w) / 3.0) * (vd3)
                //    + (2.0 / 3.0 - w / 6.0) * (vd1);
                // wcomplement = TWOTHIRDS - ONESIXTH * w;
                //  (*centroid) = (*v0) + (q * wcomplement + ONETHIRD * p * (ONE - w)) * (vd3)
                //      + wcomplement * (vd1);
                w = p / A;
                centroid->s01 += (REAL2)(vd1 * (ONETHIRD + w), vd3 * (p * w + ONETHIRD * q));
            } else
            {
                centroid->s01 += (REAL2)(vd1 * HALF, ZERO);
            }
            return ONESIXTH * A;
        } else
        {
            q = Fproduct / (vd1 * Fvector.x); // v2+q(v3-v2)
            A = ONE - HALF * p * q;
            // w = ONE / A;
            //(*centroid) = (*v2) - mad(HALF, w, mad(q, -w, q) / 3.0) * vd1
            //    - mad(HALF, w, mad(p, -w, p) / 3.0) * vd3;
            //(*centroid) = (*v2) - (HALF * w + (q * (1 - w)) / 3.0) * vd1
            //    - (HALF * w + (p * (1 - w)) / 3.0) * vd3;
            w = HALF / A;
            wcomplement = TWOTHIRDS * (HALF - w);
            (*centroid)
                = (*v2) + (REAL3)(-vd1 * (w + q * wcomplement), -vd3 * (w + p * wcomplement), ZERO);
            // (*centroid) = (*v2) - (w + q * wcomplement) * vd1 - (w + p * wcomplement) * vd3;
            return A;
        }
    } else if(PX >= *PX_xyx3)
    {
        (*centroid) = HALF * ((*v0) + (*v2));
        return ONE;

    } else
    {
        Fproduct = dot(*v3, Fvector);
        p = Fproduct / (vd3 * Fvector.y);
        q = -Fproduct / (vd1 * Fvector.x);
        A = ONE - HALF * p * q;
        // w = ONE / A;
        //(*centroid) = (*v3) + mad(HALF, w, mad(q, -w, q) / 3.0) * vd1
        //    - mad(HALF, w, mad(p, -w, p) / 3.0) * vd3;
        //(*centroid)
        //    = (*v3) + (HALF * w + (p * (1 - w)) / 3.0) * vd1 - (HALF * w + (q * (1 - w)) / 3.0) *
        //    vd3;
        w = HALF / A;
        wcomplement = TWOTHIRDS * (HALF - w);
        //(*centroid) = (*v3) + (w + p * wcomplement) * vd1 - (w + q * wcomplement) * vd3;
        (*centroid)
            = (*v3) + (REAL3)(vd1 * (w + q * wcomplement), -vd3 * (w + p * wcomplement), ZERO);
        return A;
    }
}

inline uint voxelIndex(uint i, uint j, uint k, int3 vdims)
{
    return i + j * vdims.x + k * vdims.x * vdims.y;
}

/**
 * @brief Finds ranges of corners when center voxel v is given
 *
 * @param v Center of given voxel
 * @param voxelSizes Voxel dimensions
 * @param CM Projection matrix
 * @param I_min OUT i range
 * @param I_max OUT i range inclusive
 * @param J_min OUT j range
 * @param J_max OUT j range inclusive
 */
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

// Parallel beam geometry specific includes

/**
 * Used in FLOAT_pbct_cutting_voxel_project to traverse along PY
 *
 * @param projection
 * @param CM
 * @param v
 * @param PX
 * @param value
 * @param voxelSizes
 * @param pdims
 */
void inline PBexactEdgeValues(global float* projection,
                              private REAL8 CM,
                              private REAL3 v,
                              private int PX,
                              private REAL value,
                              private REAL3 voxelSizes,
                              private int2 pdims)
{
    projection = projection + PX * pdims.y;
    const REAL PY_diff = CM.s6 * voxelSizes.s2;
    REAL3 v_min;
    REAL F, PY_min, PY_max; // PY_min + alpha F = PY for given PY
    // for alpha=1 PY_min + F = PY_max
    if(PY_diff > 0)
    {
        v_min = (REAL3)(v.x, v.y, v.z - HALF * voxelSizes.s2);
        PY_min = PBPROJECTY(CM, v_min);
        PY_max = PY_min + PY_diff;
        F = PY_diff;
    } else
    {
        v_min = (REAL3)(v.x, v.y, v.z + HALF * voxelSizes.s2);
        PY_min = PBPROJECTY(CM, v_min);
        PY_max = PY_min - PY_diff;
        F = -PY_diff;
    }
    int PJ_min = convert_int_rtn(PY_min + HALF);
    int PJ_max = convert_int_rtn(PY_max + HALF);
#ifdef DROPINCOMPLETEVOXELS
    if(PJ_min >= 0 && PJ_max < pdims.y)
#else
    if(PJ_max >= 0 && PJ_min < pdims.y)
#endif
    {
        int J;
        REAL lambda, leastLambda, lastLambda;
        if(PJ_max >= pdims.y)
        {
            PJ_max = pdims.y - 1;
            leastLambda = ((PJ_max + HALF) - PY_min) / F;
        } else
        {
            leastLambda = ONE;
        }
        if(PJ_min < 0)
        {
            J = 0;
            lastLambda = (-HALF - PY_min) / F;
        } else
        {
            J = PJ_min;
            lastLambda = ZERO;
        }
        for(; J < PJ_max; J++)
        {
            lambda = (J + HALF - PY_min) / F;
            AtomicAdd_g_f(&projection[J], (lambda - lastLambda) * value);
            lastLambda = lambda;
        }
        AtomicAdd_g_f(&projection[PJ_max], (leastLambda - lastLambda) * value);
    }
}

/**
 * Used in FLOAT_pbct_cutting_voxel_backproject to traverse along PY
 *
 * @param projection
 * @param CM
 * @param v
 * @param PX
 * @param voxelSizes
 * @param pdims
 *
 * @return
 */
float inline PBbackprojectExactEdgeValues(global const float* projection,
                                          private REAL8 CM,
                                          private REAL3 v,
                                          private int PX,
                                          private REAL3 voxelSizes,
                                          private int2 pdims)
{
    projection = projection + PX * pdims.y;
    const REAL PY_diff = CM.s6 * voxelSizes.s2;
    REAL3 v_min;
    REAL F, PY_min, PY_max; // PY_min + alpha F = PY for given PY
    // for alpha=1 PY_min + F = PY_max
    if(PY_diff > 0)
    {
        v_min = (REAL3)(v.x, v.y, v.z - HALF * voxelSizes.s2);
        PY_min = PBPROJECTY(CM, v_min);
        PY_max = PY_min + PY_diff;
        F = PY_diff;
    } else
    {
        v_min = (REAL3)(v.x, v.y, v.z + HALF * voxelSizes.s2);
        PY_min = PBPROJECTY(CM, v_min);
        PY_max = PY_min - PY_diff;
        F = -PY_diff;
    }
    int PJ_min = convert_int_rtn(PY_min + HALF);
    int PJ_max = convert_int_rtn(PY_max + HALF);
    float ADD = 0.0f;
#ifdef DROPINCOMPLETEVOXELS
    if(PJ_min >= 0 && PJ_max < pdims.y)
#else
    if(PJ_max >= 0 && PJ_min < pdims.y)
#endif
    {
        int J;
        REAL lambda, leastLambda, lastLambda;
        if(PJ_max >= pdims.y)
        {
            PJ_max = pdims.y - 1;
            leastLambda = ((PJ_max + HALF) - PY_min) / F;
        } else
        {
            leastLambda = ONE;
        }
        if(PJ_min < 0)
        {
            J = 0;
            lastLambda = (-HALF - PY_min) / F;
        } else
        {
            J = PJ_min;
            lastLambda = ZERO;
        }
        for(; J < PJ_max; J++)
        {
            lambda = (J + HALF - PY_min) / F;
            ADD += projection[J] * (lambda - lastLambda);
            lastLambda = lambda;
        }
        ADD += projection[PJ_max] * (leastLambda - lastLambda);
    }
    return ADD; // Scaling by value is performed at the end
}

// clang-format off
// const REAL vd1 = v1->x - v0->x; Nonzero x part
// const REAL vd3 = v3->y - v0->y; Nonzero y part
// polygon center of mass https://www.efunda.com/math/areas/Trapezoid.cfm to see
// when I have polygon with p base and q top, then Tx=(p*p+p*q+q*q)/(3*(p+q)) Ty=(p+2q)/(3*(p+q))
// when I take w=q/(3*(p+q)) I will have
// Tx = p/3 + q*w Ty= 1/3 + w
// Computation of CENTROID of rectangle with triangle cutout with p, q sides
// NAREA_complement=(p*q)/2
// NAREA = 1-NAREA_complement
// In cutout corner
// CENTROID = (0.5-p*NAREA_complement/3)/NAREA = 0.5 + (NAREA_COMPLEMENT/NAREA)(0.5 -(p/3)) = 0.5/NAREA - p*(ONETHIRD*NAREA_complement/NAREA)
// Outside cutout corner
// CENTROID = 0.5 - 0.5 * NAREA_COMPLEMENT/NAREA + (p/3)* NAREA_COMPLEMENT/NAREA
// example when p_complement in oposite x corner and y coordinates of the corner are the same
// w = HALF / NAREA;
// w_complement = ONETHIRD * NAREA_complement / NAREA;
// CENTROID = (REAL2)(vd1 * (ONE - w + p_complement * w_complement), vd3 * (w - q * w_complement));
// When trying to find lambda(PX) tak, že PROJ(lambda(PX)) = PX on line v0 + lambda d
// lambda = dot(v, Fvector)/-dot(d, Fvector)
// clang-format on
// When FproductVD1NEG or FproductVD3NEG are zero then basically all the segment is projected to
// given index but we need to achieve p \in [0,1] and q\in[0,1] so that we have to solve it
// individually
// I put it int the way to greedy select the biggest area possible
inline REAL PBexactIntersectionPolygons(const REAL PX,
                                        const REAL vd1,
                                        const REAL vd3,
                                        const REAL3* v0,
                                        const REAL* PX_xyx0,
                                        const REAL* PX_xyx1,
                                        const REAL* PX_xyx2,
                                        const REAL* PX_xyx3,
                                        const REAL8 CM,
                                        REAL3 voxelSizes,
                                        REAL2* centroid)
{
    REAL FX = vd1 * CM.s0;
    REAL FY = vd3 * CM.s1;
    REAL DST;
    REAL p, q, p_complement;
    REAL w, wcomplement, NAREA_complement;
    REAL NAREA;
    REAL2 CENTROID;
    if(PX < (*PX_xyx1))
    {
        DST = PX - (*PX_xyx0);
        if(FX)
        {
            p = DST / FX; // From v0 to v1
        } else
        {
            p = ONE;
        }
        if(PX < (*PX_xyx3))
        {
            if(FY)
                q = DST / FY; // From v0 to v3
            else
                q = ONE;
            CENTROID = (REAL2)(p * vd1, q * vd3);
            CENTROID *= ONETHIRD;
            (*centroid) = v0->s01 + CENTROID;
            NAREA = HALF * p * q;
            return NAREA;
        } else if(PX < (*PX_xyx2))
        {
            DST = PX - (*PX_xyx3);
            if(FX)
                q = DST / FX; // From v3 to v2
            else
                q = ONE;
            CENTROID = (REAL2)((q - p) * vd1, voxelSizes.y);
            NAREA = THREE * (p + q);
            if(NAREA > ZERO)
            {
                w = q / NAREA;
                CENTROID = (REAL2)(vd1 * (ONETHIRD * p + q * w), vd3 * (ONETHIRD + w));
            } else
            {
                CENTROID = (REAL2)(ZERO, vd3 * HALF);
            }
            (*centroid) = v0->s01 + CENTROID;
            NAREA = ONESIXTH * NAREA;
            return NAREA;
        } else
        {
            NAREA = ONE;
            CENTROID = (REAL2)(HALF * vd1, HALF * vd3);
            (*centroid) = v0->s01 + CENTROID;
            return ONE;
        }
    } else if(PX < (*PX_xyx2))
    {
        DST = PX - (*PX_xyx2);
        if(FY)
            p = DST / -FY; // v2 to v1
        else
            p = ZERO;
        if(PX < (*PX_xyx3))
        {
            p = ONE - p; // v1 to v2
            DST = PX - (*PX_xyx0);
            if(FY)
                q = DST / FY; // v0 to v3
            else
                q = ONE;
            CENTROID = (REAL2)(voxelSizes.x, (q - p) * vd3);
            NAREA = THREE * (p + q);
            if(NAREA > ZERO)
            {
                w = p / NAREA;
                CENTROID = (REAL2)(vd1 * (ONETHIRD + w), vd3 * (p * w + ONETHIRD * q));
            } else
            {
                CENTROID = (REAL2)(vd1 * HALF, ZERO);
            }
            (*centroid) = v0->s01 + CENTROID;
            NAREA = ONESIXTH * NAREA;
            return NAREA;
        } else
        {
            if(FX)
                q = DST / -FX; // v2 to v3
            else
                q = ZERO;
            CENTROID = (REAL2)(q * vd1, p * vd3);
            NAREA_complement = HALF * p * q;
            NAREA = ONE - NAREA_complement;
            w = HALF / NAREA;
            wcomplement = ONETHIRD * NAREA_complement / NAREA;
            CENTROID
                = (REAL2)(vd1 * (ONE - w + q * wcomplement), vd3 * (ONE - w + p * wcomplement));
            (*centroid) = v0->s01 + CENTROID;
            return NAREA;
        }
    } else
    {
        NAREA = ONE;
        CENTROID = (REAL2)(HALF * vd1, HALF * vd3);
        (*centroid) = v0->s01 + CENTROID;
        return ONE;
    }
}
//==============================END include.cl=====================================
