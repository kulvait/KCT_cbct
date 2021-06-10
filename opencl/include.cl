//==============================include.cl=====================================
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


//==============================END include.cl=====================================
