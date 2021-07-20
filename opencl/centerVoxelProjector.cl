//==============================centerVoxelProjector.cl=====================================
int volIndex(int* i, int* j, int* k, int3* vdims)
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
void kernel FLOATcenter_voxel_project(global const float* restrict volume,
                                      global float* restrict projection,
                                      private uint projectionOffset,
                                      private double16 PM,
                                      private double3 sourcePosition,
                                      private double3 normalToDetector,
                                      private int3 vdims,
                                      private double3 voxelSizes,
                                      private double3 volumeCenter,
                                      private int2 pdims,
                                      private float scalingFactor)
{
    int i = get_global_id(2);
    int j = get_global_id(1);
    int k = get_global_id(0); // This is more effective from the perspective of atomic colisions
    const double3 IND_ijk = { (double)(i), (double)(j), (double)(k), 0.0 };
    const double3 zerocorner_xyz = { volumeCenter.x - 0.5 * (double)vdims.x * voxelSizes.x,
                                     volumeCenter.y - 0.5 * (double)vdims.y * voxelSizes.y,
                                     volumeCenter.z - 0.5 * (double)vdims.z * voxelSizes.z };
    const double3 voxelcenter_xyz = zerocorner_xyz
        + ((IND_ijk + 0.5) * voxelSizes); // Using widening and vector multiplication operations
    int ind = projectionIndex(PM, voxelcenter_xyz, pdims);
    if(ind != -1)
    {
        int VINDEX = volIndex(&i, &j, &k, &vdims);
        float voxelValue = volume[VINDEX];
        double3 sourceToVoxel_xyz = voxelcenter_xyz - sourcePosition;
        double sourceToVoxel_xyz_norm = length(sourceToVoxel_xyz);
        double cosine = dot(normalToDetector, sourceToVoxel_xyz) / sourceToVoxel_xyz_norm;
        double cosPowThree = cosine * cosine * cosine;
        float value = voxelValue * scalingFactor
            / (sourceToVoxel_xyz_norm * sourceToVoxel_xyz_norm * cosPowThree);
        AtomicAdd_g_f(&projection[projectionOffset + ind],
                      value); // Atomic version of projection[ind] += value;
    }
}
//==============================END centerVoxelProjector.cl=====================================
