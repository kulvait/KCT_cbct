
/** Atomic float addition. Less effective implementation to have it here.
 *
 * Function from
 * https://streamhpc.com/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved/.
 *
 *
 * @param source Pointer to the memory to perform atomic operation on.
 * @param operand Float to add.
 */
inline void AtomicAdd_g_f(volatile __global float* source, const float operand)
{
    union
    {   
        unsigned int intVal;
        float floatVal;
    } newVal;
    union
    {   
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do  
    {   
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while(atomic_cmpxchg((volatile __global unsigned int*)source, prevVal.intVal, newVal.intVal)
            != prevVal.intVal);
}

int2 projectionIndex(private double16 P, private double4 vdim, int2 pdims)
{
    double4 coord;
    coord.x = vdim.x * P[0] + vdim.y * P[1] + vdim.z * P[2] + P[3];
    coord.y = vdim.x * P[4] + vdim.y * P[5] + vdim.z * P[6] + P[7];
    coord.z = vdim.x * P[8] + vdim.y * P[9] + vdim.z * P[10] + P[11];
    coord.x /= coord.z;
    coord.y /= coord.z;
    int2 ind;
    ind.x = convert_int_rtp(coord.x + 0.5);
    ind.y = convert_int_rtp(coord.y + 0.5);
    if(ind.x >= 0 && ind.y >= 0 && ind.x < pdims.x && ind.y < pdims.y)
    {
        return ind;
    } else
    {
        return pdims;
    }
}

int volIndex(int* i, int* j, int* k, int4* vdims)
{
    return (*i) + (*j) * vdims->x + (*k) * (vdims->x * vdims->y);
}

/** Project given volume using cutting voxel projector.
 *
 *
 * @param volume Volume to project.
 * @param projection Projection to construct.
 * @param PM Projection matrix. This projection matrix is constructed in the way that (i,j,k) =
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
void kernel FLOATcenter_voxel_project(global float* volume,
                                       global float* projection,
                                       private double16 PM,
                                       private double4 sourcePosition,
                                       private double4 normalToDetector,
                                       private int4 vdims,
                                       private double4 voxelSizes,
                                       private int2 pdims,
                                       private float scalingFactor)
{
    int i = get_global_id(2);
    int j = get_global_id(1);
    int k = get_global_id(0); // This is more effective from the perspective of atomic colisions
    const double4 IND_ijk = { (double)(i), (double)(j), (double)(k), 0.0 };
    const double4 zerocorner_xyz = -convert_double4(vdims) / 2.0;
    const double4 voxelcenter_xyz = zerocorner_xyz
        + ((IND_ijk + 0.5) * voxelSizes); // Using widening and vector multiplication operations
    int2 p_ab = projectionIndex(PM, voxelcenter_xyz, pdims);
    if(p_ab.x != pdims.x && p_ab.y != pdims.y)
    {
	int VINDEX = volIndex(&i, &j, &k, &vdims);
        float voxelValue = volume[VINDEX];
        double4 sourceToVoxel_xyz = voxelcenter_xyz - sourcePosition;
        double sourceToVoxel_xyz_norm = length(sourceToVoxel_xyz);
        double cosine = dot(normalToDetector, sourceToVoxel_xyz) / sourceToVoxel_xyz_norm;
        double cosPowThree = cosine * cosine * cosine;
        float value = voxelValue * scalingFactor
            / (sourceToVoxel_xyz_norm * sourceToVoxel_xyz_norm * cosPowThree);
        int ind = p_ab.y * pdims.x + p_ab.x;
        AtomicAdd_g_f(&projection[ind], value); // Atomic version of projection[ind] += value;
    }
}
