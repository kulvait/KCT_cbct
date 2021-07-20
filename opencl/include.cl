//==============================include.cl=====================================

//Atomic operations
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


//CVP Projector routines
//#define LOCALARRAYSIZE Theoretical maximum of 65536 bytes AMD, 49152 NVIDIA, 32768 Intel

#define DROPCENTEROFFPROJECTORVOXELS
//DROPINCOMPLETEVOXELS is not implemented the same in barrier implementation
#ifdef RELAXED
typedef float REAL;
typedef float3 REAL3;
typedef float16 REAL16;
//__constant float ONE=1.0f;
//__constant float HALF=0.5f;
//__constant float ZERO=0.0f;
#define ZERO 0.0f
#define ONESIXTH 0.16666667f
#define ONETHIRD 0.33333333f
#define HALF 0.5f
#define TWOTHIRDS 0.66666666f
#define ONE 1.0f
#define convert_REAL3(x) convert_float3(x)
#else
typedef double REAL;
typedef double3 REAL3;
typedef double16 REAL16;
//__constant double ONE=1.0;
//__constant double HALF=0.5;
//__constant double ZERO=0.0f;
#define ZERO 0.0
#define ONESIXTH 0.16666666666666667
#define ONETHIRD 0.3333333333333333
#define HALF 0.5
#define TWOTHIRDS 0.6666666666666666
#define ONE 1.0
#define convert_REAL3(x) convert_double3(x)
#endif

#define PROJECTX0(CM, v0) dot(v0, CM.s012) / dot(v0, CM.s89a)

#define PROJECTY0(CM, v0) dot(v0, CM.s456) / dot(v0, CM.s89a)

#define INDEX(f) convert_int_rtn(f + HALF)

//For extracting edge values
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


inline REAL exactIntersectionPoints0_extended(const REAL PX,
                                                const REAL3* v0,
                                                const REAL3* v1,
                                                const REAL3* v2,
                                                const REAL3* v3,
                                                const REAL3 vd1,
                                                const REAL3 vd3,
                                                const REAL* PX_ccw0,
                                                const REAL* PX_ccw1,
                                                const REAL* PX_ccw2,
                                                const REAL* PX_ccw3,
                                                const REAL16 CM,
                                                REAL3* centroid)
{
    const REAL3 Fvector = CM.s012 - PX * CM.s89a;
    // const REAL3 vd1 = (*v1) - (*v0);
    // const REAL3 vd3 = (*v3) - (*v0);
    REAL Fproduct, FproductVD;
    REAL p, q;
    REAL A, w, wcomplement;
    if(PX < (*PX_ccw1))
    {
        Fproduct = -dot(*v0, Fvector);
        FproductVD = dot(vd1, Fvector); // VD1
        p = Fproduct / FproductVD; // v0+p*(v1-v0)
        if(PX < (*PX_ccw3))
        {
            q = Fproduct / dot(vd3, Fvector);
            (*centroid) = (*v0) + (ONETHIRD * p) * vd1 + (ONETHIRD * q) * vd3;
            return HALF * p * q;
        } else if(PX < (*PX_ccw2))
        {
            q = -dot(*v3, Fvector) / FproductVD;
            A = HALF * (p + q);
            if(A != ZERO) // Due to rounding errors equality might happen producing nan
            {
                w = p / A;
                //(*centroid) = (*v0)
                //    + mad(p, mad(-ONE / 6.0, w, 2.0 / 3.0), mad(-q, w, q) / 3.0) * (vd1)
                //    + mad(-ONE / 6.0, w, 2.0 / 3.0) * (vd3);
                //(*centroid) = (*v0) + (p * (2.0 / 3.0 - w / 6.0) + q * (1 - w) / 3.0) * (vd1)
                //    + (2.0 / 3.0 - w / 6.0) * (vd3);
                wcomplement = TWOTHIRDS - ONESIXTH * w;
                (*centroid) = (*v0) + (p * wcomplement + ONETHIRD * q * (ONE - w)) * (vd1)
                    + wcomplement * (vd3);
            } else
            {
                (*centroid) = (*v0);
            }
            return A;
        } else
        {
            p = ONE - p;
            q = -dot(*v1, Fvector) / dot(vd3, Fvector);
            A = ONE - HALF * p * q;
            // w = ONE / A;
            //(*centroid) = (*v0) - mad(HALF, w, mad(p, -w, p) / 3.0) * vd1
            //    + mad(HALF, w, mad(q, -w, q) / 3.0) * vd3;
            //(*centroid) = (*v1) - (HALF * w + (p * (1 - w)) / 3.0) * vd1
            //    + (HALF * w + (q * (1 - w)) / 3.0) * vd3;
            w = HALF / A;
            wcomplement = TWOTHIRDS * (HALF - w);
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
            p = ONE - p; // V1 + p * (V2-V1)
            q = -dot(*v0, Fvector) / FproductVD; // V0 + q (V3-V0)
            A = HALF * (p + q);
            if(A != ZERO) // Due to rounding errors equality might happen producing nan
            {
                w = q / A;
                //(*centroid) = (*v0)
                //    + mad(q, mad(-ONE / 6.0, w, 2.0 / 3.0), mad(-p, w, p) / 3.0) * (vd3)
                //    + mad(-ONE / 6.0, w, 2.0 / 3.0) * (vd1);
                //(*centroid) = (*v0) + (q * (2.0 / 3.0 - w / 6.0) + p * (1 - w) / 3.0) * (vd3)
                //    + (2.0 / 3.0 - w / 6.0) * (vd1);
                wcomplement = TWOTHIRDS - ONESIXTH * w;
                (*centroid) = (*v0) + (q * wcomplement + ONETHIRD * p * (ONE - w)) * (vd3)
                    + wcomplement * (vd1);
            } else
            {
                (*centroid) = (*v0);
            }
            return A;
        } else
        {
            q = Fproduct / dot(vd1, Fvector); // v2+q(v3-v2)
            A = ONE - HALF * p * q;
            // w = ONE / A;
            //(*centroid) = (*v2) - mad(HALF, w, mad(q, -w, q) / 3.0) * vd1
            //    - mad(HALF, w, mad(p, -w, p) / 3.0) * vd3;
            //(*centroid) = (*v2) - (HALF * w + (q * (1 - w)) / 3.0) * vd1
            //    - (HALF * w + (p * (1 - w)) / 3.0) * vd3;
            w = HALF / A;
            wcomplement = TWOTHIRDS * (HALF - w);
            (*centroid) = (*v2) - (w + q * wcomplement) * vd1 - (w + p * wcomplement) * vd3;
            return A;
        }
    } else if(PX >= *PX_ccw3)
    {
        (*centroid) = HALF * ((*v0) + (*v2));
        return ONE;

    } else
    {
        Fproduct = dot(*v3, Fvector);
        p = Fproduct / dot(vd3, Fvector);
        q = -Fproduct / dot(vd1, Fvector);
        A = ONE - HALF * p * q;
        // w = ONE / A;
        //(*centroid) = (*v3) + mad(HALF, w, mad(q, -w, q) / 3.0) * vd1
        //    - mad(HALF, w, mad(p, -w, p) / 3.0) * vd3;
        //(*centroid)
        //    = (*v3) + (HALF * w + (p * (1 - w)) / 3.0) * vd1 - (HALF * w + (q * (1 - w)) / 3.0) *
        //    vd3;
        w = HALF / A;
        wcomplement = TWOTHIRDS * (HALF - w);
        (*centroid) = (*v3) + (w + p * wcomplement) * vd1 - (w + q * wcomplement) * vd3;
        return A;
    }
}

inline uint voxelIndex(uint i, uint j, uint k, int3 vdims)
{
    return i + j * vdims.x + k * vdims.x * vdims.y;
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

//==============================END include.cl=====================================
