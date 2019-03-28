void kernel FLOATvector_NormSquarePartial(global float* x,
                                          global float* normSquare,
                                          private uint frameLen)
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

void kernel FLOATvector_SumPartial(global float* x, global float* sumPartial, private uint frameLen)
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

// Code based on
// https://www.fz-juelich.de/SharedDocs/Downloads/IAS/JSC/EN/slides/opencl/opencl-05-reduction.pdf?__blob=publicationFile
// Evidently gs must be multiple of ls and for this code to work ls must be 2^n
void kernel FLOATvector_SumPartial_barier(global float* x,
                                          global float* partialSum,
                                          local float* loc,
                                          private uint vecLength)
{
    int gid = get_global_id(0);
    int gs = get_global_size(0);
    int lid = get_local_id(0);
    int ls = get_local_size(0);
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
void kernel vector_NormSquarePartial(global float* x,
                                     global double* normSquare,
                                     private uint frameLen)
{
    int gid = get_global_id(0);
    double sum = 0;
    double val;
    int start = gid * frameLen;
    int end = start + frameLen;
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
void kernel vector_SumPartial(global double* x, global double* sumPartial, private uint frameLen)
{
    uint gid = get_global_id(0);
    double sum = 0;
    uint start = gid * frameLen;
    uint end = start + frameLen;
    for(uint i = start; i < end; i++)
    {
        sum += x[i];
    }
    sumPartial[gid] = sum;
}

// Code based on
// https://www.fz-juelich.de/SharedDocs/Downloads/IAS/JSC/EN/slides/opencl/opencl-05-reduction.pdf?__blob=publicationFile
// Evidently gs must be multiple of ls and for this code to work ls must be 2^n
void kernel vector_NormSquarePartial_barier(global float* x,
                                            global double* normSquare,
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

void kernel vector_ScalarProductPartial_barier(global float* a,
                                               global float* b,
                                               global double* product,
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

// Want to use it in second line where I am loading doubles into it
// https://www.fz-juelich.de/SharedDocs/Downloads/IAS/JSC/EN/slides/opencl/opencl-05-reduction.pdf?__blob=publicationFile
// Evidently gs must be multiple of ls and for this code to work ls must be 2^n
void kernel vector_SumPartial_barier(global double* x,
                                     global double* partialSum,
                                     local double* loc,
                                     private uint vecLength)
{
    int gid = get_global_id(0);
    int gs = get_global_size(0);
    int lid = get_local_id(0);
    int ls = get_local_size(0);
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

void kernel FLOAT_copy_vector(global float* from, global float* to)
{
    int gid = get_global_id(0);
    to[gid] = from[gid];
}

void kernel FLOAT_scale_vector(global float* v, private float f)
{
    int gid = get_global_id(0);
    v[gid] = v[gid] * f;
}

void kernel FLOAT_add_into_first_vector_second_vector_scaled(global float* a,
                                                             global float* b,
                                                             private float f)
{
    int gid = get_global_id(0);
    float val = a[gid] + f * b[gid];
    a[gid] = val;
}

void kernel FLOAT_add_into_first_vector_scaled_second_vector(global float* a,
                                                             global float* b,
                                                             private float f)
{
    int gid = get_global_id(0);
    float val = f * a[gid] + b[gid];
    a[gid] = val;
}
