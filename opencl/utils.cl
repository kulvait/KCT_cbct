//#pragma OPENCL EXTENSION cl_amd_printf : enable
//==============================utils.cl=====================================
void kernel FLOATvector_NormSquarePartial(global const float* restrict x,
                                          global float* restrict normSquare,
                                          private uint frameLen)
{
    uint gid = get_global_id(0);
    uint start = gid * frameLen;
    uint end = start + frameLen;
    float sum = 0.0f;
    float val;
    for(int i = start; i < end; i++)
    {
        val = x[i];
        sum += val * val;
    }
    normSquare[gid] = sum;
}

void kernel FLOATvector_SumPartial(global const float* restrict x,
                                   global float* restrict sumPartial,
                                   private uint frameLen)
{
    uint gid = get_global_id(0);
    uint start = gid * frameLen;
    uint end = start + frameLen;
    float sum = 0.0f;
    float val;
    for(int i = start; i < end; i++)
    {
        val = x[i];
        sum += val;
    }
    sumPartial[gid] = sum;
}

void kernel FLOATvector_MaxPartial(global const float* restrict x,
                                   global float* restrict partialResults,
                                   private uint frameLen)
{
    uint gid = get_global_id(0);
    uint start = gid * frameLen;
    uint end = start + frameLen;
    float maxVal = x[start];
    float val;
    for(int i = start; i < end; i++)
    {
        val = x[i];
        maxVal = max(val, maxVal);
    }
    partialResults[gid] = maxVal;
}

// Code based on
// https://www.fz-juelich.de/SharedDocs/Downloads/IAS/JSC/EN/slides/opencl/opencl-05-reduction.pdf?__blob=publicationFile
// Evidently gs must be multiple of ls and for this code to work ls must be 2^n
void kernel FLOATvector_NormSquarePartial_barrier(global const float* restrict x,
                                                  global float* restrict normSquare,
                                                  local float* localx,
                                                  private uint vecLength)
{
    uint gid = get_global_id(0);
    uint gs = get_global_size(0);
    uint lid = get_local_id(0);
    uint ls = get_local_size(0);
    float val;
    if(gid < vecLength)
    {
        val = x[gid];
    } else
    {
        val = 0.0f;
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
        gid = get_group_id(0);
        normSquare[gid] = localx[0] + localx[1];
    }
}

// Code based on
// https://www.fz-juelich.de/SharedDocs/Downloads/IAS/JSC/EN/slides/opencl/opencl-05-reduction.pdf?__blob=publicationFile
// Evidently gs must be multiple of ls and for this code to work ls must be 2^n
void kernel FLOATvector_SumPartial_barrier(global const float* restrict x,
                                           global float* restrict partialSum,
                                           local float* loc,
                                           private uint vecLength)
{
    uint gid = get_global_id(0);
    uint gs = get_global_size(0);
    uint lid = get_local_id(0);
    uint ls = get_local_size(0);
    float val;
    if(gid < vecLength)
    {
        val = x[gid];
    } else
    {
        val = 0.0;
    }
    loc[lid] = val;

    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint stride = ls / 2; stride > 1; stride >>= 1) // Does the same as /=2
    {
        if(lid < stride)
        {
            loc[lid] += loc[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lid == 0)
    {
        gid = get_group_id(0);
        partialSum[gid] = loc[0] + loc[1];
    }
}

// Code based on
// https://www.fz-juelich.de/SharedDocs/Downloads/IAS/JSC/EN/slides/opencl/opencl-05-reduction.pdf?__blob=publicationFile
// Evidently gs must be multiple of ls and for this code to work ls must be 2^n
void kernel FLOATvector_MaxPartial_barrier(global const float* restrict x,
                                           global float* restrict partialResult,
                                           local float* localx,
                                           private uint vecLength)
{
    uint gid = get_global_id(0);
    uint gs = get_global_size(0);
    uint lid = get_local_id(0);
    uint ls = get_local_size(0);
    float val;
    if(gid < vecLength)
    {
        val = x[gid];
    } else
    {
        val = 0.0f;
    }
    localx[lid] = val;

    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint stride = ls / 2; stride > 1; stride >>= 1) // Does the same as /=2
    {
        if(lid < stride)
        {
            localx[lid] = max(localx[lid], localx[lid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lid == 0)
    {
        gid = get_group_id(0);
        partialResult[gid] = max(localx[0], localx[1]);
    }
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
void kernel vector_NormSquarePartial(global const float* restrict x,
                                     global double* restrict normSquare,
                                     private uint frameLen)
{
    int gid = get_global_id(0);
    int start = gid * frameLen;
    int end = start + frameLen;
    double sum = 0.0;
    double val;
    for(int i = start; i < end; i++)
    {
        val = x[i];
        sum += val * val;
    }
    normSquare[gid] = sum;
}

/**
 * Second line function that accepts doubles
 *
 * @param x
 * @param sumPartial
 * @param frameLen
 *
 * @return
 */
void kernel vector_SumPartial(global const double* restrict x,
                              global double* restrict sumPartial,
                              private uint frameLen)
{
    uint gid = get_global_id(0);
    uint start = gid * frameLen;
    uint end = start + frameLen;
    double sum = 0.0;
    for(uint i = start; i < end; i++)
    {
        sum += x[i];
    }
    sumPartial[gid] = sum;
}

// Code based on
// https://www.fz-juelich.de/SharedDocs/Downloads/IAS/JSC/EN/slides/opencl/opencl-05-reduction.pdf?__blob=publicationFile
// Evidently gs must be multiple of ls and for this code to work ls must be 2^n
void kernel vector_NormSquarePartial_barrier(global const float* restrict x,
                                             global double* restrict normSquare,
                                             local double* localx,
                                             private uint vecLength)
{
    uint gid = get_global_id(0);
    uint gs = get_global_size(0);
    uint lid = get_local_id(0);
    uint ls = get_local_size(0);
    double val;
    if(gid < vecLength)
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
        gid = get_group_id(0);
        normSquare[gid] = localx[0] + localx[1];
    }
}

// Want to use it in second line where I am loading doubles into it
// https://www.fz-juelich.de/SharedDocs/Downloads/IAS/JSC/EN/slides/opencl/opencl-05-reduction.pdf?__blob=publicationFile
// Evidently gs must be multiple of ls and for this code to work ls must be 2^n
void kernel vector_SumPartial_barrier(global const double* restrict x,
                                      global double* restrict partialSum,
                                      local double* loc,
                                      private uint vecLength)
{
    uint gid = get_global_id(0);
    uint gs = get_global_size(0);
    uint lid = get_local_id(0);
    uint ls = get_local_size(0);
    double val;
    if(gid < vecLength)
    {
        val = x[gid];
    } else
    {
        val = 0.0;
    }
    loc[lid] = val;

    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint stride = ls / 2; stride > 1; stride >>= 1) // Does the same as /=2
    {
        if(lid < stride)
        {
            loc[lid] += loc[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lid == 0)
    {
        gid = get_group_id(0);
        partialSum[gid] = loc[0] + loc[1];
    }
}

void kernel vector_ScalarProductPartial_barrier(global const float* restrict a,
                                                global const float* restrict b,
                                                global double* restrict product,
                                                local double* localx,
                                                private uint vecLength)
{
    uint gid = get_global_id(0);
    uint gs = get_global_size(0);
    uint lid = get_local_id(0);
    uint ls = get_local_size(0);
    double val;
    if(gid < vecLength)
    {
        val = a[gid] * b[gid];
    } else
    {
        val = 0.0;
    }
    localx[lid] = val;

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
        gid = get_group_id(0);
        product[gid] = localx[0] + localx[1];
    }
}

void kernel FLOATvector_zero(global float* a) { a[get_global_id(0)] = 0.0f; }

void kernel FLOATvector_zero_infinite_values(global float* X)
{
    uint gid = get_global_id(0);
    float val = X[gid];
    if(isinf(val))
    {
        X[gid] = 0.0f;
    }
}

void kernel FLOATvector_scale(global float* v, private float f)
{
    int gid = get_global_id(0);
    v[gid] = v[gid] * f;
}

void kernel FLOATvector_sqrt(global float* v)
{
    uint gid = get_global_id(0);
    v[gid] = sqrt(v[gid]);
}

void kernel FLOATvector_invert(global float* v)
{
    uint gid = get_global_id(0);
    v[gid] = 1.0 / v[gid];
}

void kernel FLOATvector_invert_except_zero(global float* X)
{
    uint gid = get_global_id(0);
    float val = X[gid];
    if(val != 0.0f)
    {
        X[gid] = 1 / val;
    }
}

void kernel FLOATvector_substitute_greater_than(global float* X,
                                                const float minValue,
                                                const float substitution)
{
    uint gid = get_global_id(0);
    float val = X[gid];
    if(val > minValue)
    {
        X[gid] = substitution;
    }
}

void kernel FLOATvector_substitute_lower_than(global float* X,
                                              const float maxValue,
                                              const float substitution)
{
    uint gid = get_global_id(0);
    float val = X[gid];
    if(val < maxValue)
    {
        X[gid] = substitution;
    }
}

void kernel FLOATvector_copy(global const float* restrict A, global float* restrict B)
{
    const size_t gid = get_global_id(0);
    B[gid] = A[gid];
}

void kernel FLOATvector_copy_offset(global const float* restrict A,
                                    global float* restrict B,
                                    private uint offset)
{
    const size_t index = get_global_id(0) + offset;
    B[index] = A[index];
}

void kernel FLOATvector_copy_offsets(global const float* restrict A,
                                     private uint oA,
                                     global float* restrict B,
                                     private uint oB)
{
    const size_t gid = get_global_id(0);
    B[gid + oB] = A[gid + oA];
}

void kernel FLOATvector_A_equals_cB(global float* restrict A,
                                    global const float* restrict B,
                                    const private float c)
{
    const size_t gid = get_global_id(0);
    A[gid] = c * B[gid];
}

void kernel FLOATvector_A_equals_A_plus_cB(global float* restrict A,
                                           global const float* restrict B,
                                           const private float c)
{
    const size_t gid = get_global_id(0);
    A[gid] += c * B[gid];
}

void kernel FLOATvector_A_equals_Ac_plus_B(global float* restrict A,
                                           global const float* restrict B,
                                           private float c)
{
    const size_t gid = get_global_id(0);
    A[gid] = A[gid] * c + B[gid];
}

void kernel FLOATvector_A_equals_A_times_B(global float* restrict A, global const float* restrict B)
{
    const size_t gid = get_global_id(0);
    A[gid] = A[gid] * B[gid];
}

void kernel FLOATvector_C_equals_A_times_B(global const float* restrict A,
                                           global const float* restrict B,
                                           global float* restrict C)
{
    const size_t gid = get_global_id(0);
    C[gid] = A[gid] * B[gid];
}

void kernel FLOATvector_A_equals_A_plus_cB_offset(global float* restrict A,
                                                  global const float* restrict B,
                                                  private float c,
                                                  private uint offset)
{
    const size_t index = get_global_id(0) + offset;
    A[index] += c * B[index];
}

void kernel FLOATvector_B_equals_A_plus_B_offsets(global const float* restrict A,
                                                  const uint oA,
                                                  global float* restrict B,
                                                  const uint oB)
{
    const size_t gid = get_global_id(0);
    B[gid + oB] += A[gid + oA];
}

void kernel FLOATvector_A_equals_A_plus_cB_offsets(global float* restrict A,
                                                   private const uint oA,
                                                   global const float* restrict B,
                                                   private const uint oB,
                                                   private float c)
{
    const size_t gid = get_global_id(0);
    A[gid + oA] += c * B[gid + oB];
}

//==============================END utils.cl=====================================
