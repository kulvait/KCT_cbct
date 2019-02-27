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

/** Atomic float addition.
 *
 * Function from
 * https://streamhpc.com/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved/.
 *
 *
 * @param source Pointer to the memory to perform atomic operation on.
 * @param operand Float to add.
 */
inline void AtomicAdd_l_f(volatile __global float* adr, const float v)
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
void kernel FLOATvector_NormSquarePartial(global float* x,
                                          global float* normSquare,
                                          private int frameLen)
{
    int gid = get_global_id(0);
    float sum = 0;
    float val;
    int start = gid * frameLen;
    int end = start + frameLen;
    for(int i = start; i < end; i++)
    {
        val = x[i];
        sum += val * val;
    }
    normSquare[gid] = sum;
}

void kernel FLOATvector_SumPartial(global float* x, global float* sumPartial, private int frameLen)
{
    int gid = get_global_id(0);
    float sum = 0;
    float val;
    int start = gid * frameLen;
    int end = start + frameLen;
    for(int i = start; i < end; i++)
    {
        val = x[i];
        sum += val;
    }
    sumPartial[gid] = sum;
}

// Code based on
// https://www.fz-juelich.de/SharedDocs/Downloads/IAS/JSC/EN/slides/opencl/opencl-05-reduction.pdf?__blob=publicationFile
// Evidently gs must be multiple of ls and for this code to work ls must be 2^n
void kernel FLOATvector_NormSquarePartial_barier(global float* x,
                                                 global float* normSquare,
                                                 local float* localx,
                                                 private uint vecLength)
{
    int gid = get_global_id(0);
    int gs = get_global_size(0);
    int lid = get_local_id(0);
    int ls = get_local_size(0);
    float val;
    if(gs < vecLength)
    {
        val = x[gid];
    } else
    {
        val = 0.0;
    }
    localx[lid] = val * val;

    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint stride = ls / 2; stride > 1; stride >>= 1) // Does the same as /=2
    {
        if(lid < stride)
        {
            localx[lid] += localx[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lid == 0)
    {
        normSquare[get_group_id(0)] = localx[0] + localx[1];
    }
}
