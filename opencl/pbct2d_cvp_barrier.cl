//==============================pbct2d_cvp.cl=====================================
void inline PB2DInsertVerticalValuesBarrier(global const float* restrict volumePtr,
                                            private uint volumeStride,
                                            local float* restrict projectionPtr,
                                            private uint numValues,
                                            private float factor)
{
    float val;
    for(uint i = 0; i != numValues; i++)
    {
        val = volumePtr[i * volumeStride];
        val *= factor;
        if(val != 0.0f)
        {
            AtomicAdd_l_f(projectionPtr + i, val);
        }
    }
}

inline REAL PB2DExactPolygonPartBarrierOriginal(const REAL PQ,
                                         const REAL PX_inc1,
                                         const REAL PX_inc2,
                                         const REAL PX_inc3)
{
    if(PQ <= ZERO)
    {
        return ZERO;
    } else if(PQ < PX_inc1)
    {
        return HALF * PQ * PQ / (PX_inc1 * PX_inc2);
    } else if(PQ < PX_inc2)
    {
        return PQ - HALF * PX_inc1 / PX_inc2;
    } else if(PQ < PX_inc3)
    {
        if(PX_inc1 == 0)
        {
            return PQ / PX_inc2;
        } else
        {
            return ONE - HALF * (PX_inc3 - PQ) * (PX_inc3 - PQ) / (PX_inc1 * PX_inc2);
        }
    } else
    {
        return ONE;
    }
}

// clang-format off
// PQ is PX - PX_min for given coordinate to test PX
// PX_inc1 <= PX_inc2 <= PX_inc3 are values so that
// PX_min + PX_inc1 is in one corner
// PX_min + PX_inc2 is in another corner
// PX_min + PX_inc3 is in maximal corner
// clang-format on
// I don't need first test because I have assured PX_inc1!=0 when calling this procedure
inline REAL PB2DExactPolygonPartBarrier(const REAL PQ,
                                        const REAL PX_inc1,
                                        const REAL PX_inc2,
                                        const REAL PX_inc3)
{
    if(PQ < PX_inc1)
    {
        return HALF * PQ * PQ / (PX_inc1 * PX_inc2);
    } else if(PQ < PX_inc2)
    {
        return PQ - HALF * PX_inc1 / PX_inc2;
    } else if(PQ < PX_inc3)
    {
        return ONE - HALF * (PX_inc3 - PQ) * (PX_inc3 - PQ) / (PX_inc1 * PX_inc2);
    } else
    {
        return ONE;
    }
}

void inline FLOAT_pbct2d_cutting_voxel_project_barrier_local_pxinc1iszero(
    global const float* restrict voxelPointer,
    local float* restrict projection,
    private int2 pdims,
    private float voxelVolumeTimesScalingFactor,
    private int PI_min,
    private int PI_max,
    private float PX_min,
    private float PX_inc2,
    private int volumeStride)
{
    if(PI_max >= 0 && PI_min < pdims.x)
    {
        local float* restrict pixelPointer;
        if(PI_max <= PI_min) // These indices are in the admissible range
        {
            PI_min = convert_int_rtn(PX_min + HALF * PX_inc2 + HALF);
            pixelPointer = projection + PI_min * pdims.y;
            PB2DInsertVerticalValuesBarrier(voxelPointer, volumeStride, pixelPointer, pdims.y,
                                            voxelVolumeTimesScalingFactor);
        } else
        {
            REAL sectionSize_prev, sectionSize_cur, polygonSize;
            REAL factor;
            REAL PQ;
            int I = max(-1, PI_min);
            int I_STOP = min(PI_max, pdims.x);
            // Section of the square that corresponds to the indices < i
            PQ = ((REAL)I) + HALF - PX_min;
            sectionSize_prev = PQ / PX_inc2;
            if(I >= 0)
            {
                factor = voxelVolumeTimesScalingFactor * sectionSize_prev;
                pixelPointer = projection + I * pdims.y;
                PB2DInsertVerticalValuesBarrier(voxelPointer, volumeStride, pixelPointer, pdims.y,
                                                factor);
            }
            for(I = I + 1; I < I_STOP; I++)
            {
                PQ = ((REAL)I) + HALF - PX_min;
                sectionSize_cur = PQ / PX_inc2;
                polygonSize = sectionSize_cur - sectionSize_prev;
                factor = voxelVolumeTimesScalingFactor * polygonSize;
                pixelPointer = projection + I * pdims.y;
                PB2DInsertVerticalValuesBarrier(voxelPointer, volumeStride, pixelPointer, pdims.y,
                                                factor);
                sectionSize_prev = sectionSize_cur;
            }
            polygonSize = ONE - sectionSize_prev;
            // Without second test polygonsize==0 triggers division by zero
            if(I_STOP < pdims.x && polygonSize > zeroPrecisionTolerance)
            {
                factor = voxelVolumeTimesScalingFactor * polygonSize;
                pixelPointer = projection + I * pdims.y;
                PB2DInsertVerticalValuesBarrier(voxelPointer, volumeStride, pixelPointer, pdims.y,
                                                factor);
            }
        }
    }
}

void inline FLOAT_pbct2d_cutting_voxel_project_barrier_local(
    global const float* restrict voxelPointer,
    local float* restrict projection,
    private int2 pdims,
    private float voxelVolumeTimesScalingFactor,
    private int PI_min,
    private int PI_max,
    private float PX_min,
    private float PX_inc1,
    private float PX_inc2,
    private float PX_inc3,
    private int volumeStride)
{
    if(PI_max >= 0 && PI_min < pdims.x)
    {
        local float* restrict pixelPointer;
        if(PI_max <= PI_min) // These indices are in the admissible range
        {
            PI_min = convert_int_rtn(PX_min + HALF * PX_inc3 + HALF);
            pixelPointer = projection + PI_min * pdims.y;
            PB2DInsertVerticalValuesBarrier(voxelPointer, volumeStride, pixelPointer, pdims.y,
                                            voxelVolumeTimesScalingFactor);
        } else
        {
            REAL sectionSize_prev, sectionSize_cur, polygonSize;
            REAL factor;
            REAL PQ;
            int I = max(-1, PI_min);
            int I_STOP = min(PI_max, pdims.x);
            // Section of the square that corresponds to the indices < i
            PQ = ((REAL)I) + HALF - PX_min;
            sectionSize_prev = PB2DExactPolygonPartBarrier(PQ, PX_inc1, PX_inc2, PX_inc3);
            if(I >= 0)
            {
                factor = voxelVolumeTimesScalingFactor * sectionSize_prev;
                pixelPointer = projection + I * pdims.y;
                PB2DInsertVerticalValuesBarrier(voxelPointer, volumeStride, pixelPointer, pdims.y,
                                                factor);
            }
            for(I = I + 1; I < I_STOP; I++)
            {
                PQ = ((REAL)I) + HALF - PX_min;
                sectionSize_cur = PB2DExactPolygonPartBarrier(PQ, PX_inc1, PX_inc2, PX_inc3);
                polygonSize = sectionSize_cur - sectionSize_prev;
                factor = voxelVolumeTimesScalingFactor * polygonSize;
                pixelPointer = projection + I * pdims.y;
                PB2DInsertVerticalValuesBarrier(voxelPointer, volumeStride, pixelPointer, pdims.y,
                                                factor);
                sectionSize_prev = sectionSize_cur;
            }
            polygonSize = ONE - sectionSize_prev;
            // Without second test polygonsize==0 triggers division by zero
            if(I_STOP < pdims.x && polygonSize > zeroPrecisionTolerance)
            {
                factor = voxelVolumeTimesScalingFactor * polygonSize;
                pixelPointer = projection + I * pdims.y;
                PB2DInsertVerticalValuesBarrier(voxelPointer, volumeStride, pixelPointer, pdims.y,
                                                factor);
            }
        }
    }
}

/** Project given volume using cutting voxel projector and parallel rays geometry.
 *
 *
 * @param volume Volume to project.
 * @param projection Projection to construct.
 * @param CM Projection matrix. This projection matrix is constructed in the way that (i,j,k) =
 * (0,0,0,1) is projected to the center of the voxel with given coordinates.
 * @param sourcePosition Source position in the xyz space.
 * @param vdims Dimensions of the volume.
 * @param voxelSizes Lengths of the voxel edges, including third dimension.
 * @param pdims Dimensions of the projection.
 * @param scalingFactor Scale the results by this factor.
 *
 */
void kernel FLOAT_pbct2d_cutting_voxel_project_barrier(global const float* restrict volume,
                                                       global float* restrict projection,
                                                       local float* restrict localProjection,
                                                       private ulong projectionOffset,
                                                       private double3 _CM,
                                                       private int3 vdims,
                                                       private double3 _voxelSizes,
                                                       private double2 _volumeCenter,
                                                       private int2 pdims,
                                                       private float scalingFactor,
                                                       private int k_from,
                                                       private int k_count)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int li = get_local_id(0);
    int lj = get_local_id(1);
    int lis = get_local_size(0);
    int ljs = get_local_size(1);
    uint LOCSIZE = lis * ljs;
    uint LID = li * ljs + lj;
    projection += projectionOffset;
#ifdef RELAXED
    const float3 CM = convert_float3(_CM);
    const float3 voxelSizes = convert_float3(_voxelSizes);
    const float2 volumeCenter = convert_float2(_volumeCenter);
#else
#define CM _CM
#define voxelSizes _voxelSizes
#define volumeCenter _volumeCenter
#endif
    const REAL2 halfVoxelSizes = HALF * voxelSizes.s01;
    const REAL2 volumeCenter_voxelcenter_offset
        = (REAL2)(2 * i + 1 - vdims.x, 2 * j + 1 - vdims.y) * halfVoxelSizes;
    const REAL2 voxelcenter_xy = volumeCenter + volumeCenter_voxelcenter_offset;
    const ulong IND = voxelIndex(i, j, k_from, vdims);
    const global float* restrict voxelPointer = volume + IND;
    const int volumeStride = vdims.x * vdims.y;
    const REAL voxelVolumeTimesScalingFactor
        = voxelSizes.x * voxelSizes.y * voxelSizes.z * scalingFactor;
    // Projected voxel center and analysis important values
    const REAL PX0 = PB2DPROJECT(CM, voxelcenter_xy);
    const REAL2 PXinc = fabs(CM.s01) * halfVoxelSizes.s01; // Increments for half voxel shifts
    const REAL PXincTotal = PXinc.x + PXinc.y;
    const REAL PX_min = PX0 - PXincTotal; // At minimum
    const REAL PX_max = PX0 + PXincTotal; // maximum
    const REAL PX_inc1 = TWO * fmin(PXinc.x, PXinc.y);
    const REAL PX_inc2 = TWO * fmax(PXinc.x, PXinc.y);
    const REAL PX_inc3 = TWO * PXincTotal;
    const int PI_min = INDEX(PX_min + zeroPrecisionTolerance);
    const int PI_max = INDEX(PX_max - zeroPrecisionTolerance);

    local int LocalMemory_PIStart_LOC;
    int LocalMemory_PIStart;
    local int LocalMemory_PICount_LOC;
    local uint LocalMemory_Size_LOC;
    local int LocalFootprint_PIUpperBound_LOC;
    local int2 Lpdims_LOC;
    local bool stopNextIteration_LOC;
    bool stopThisIteration;
    if(LID == 0) // Get dimension
    {
        const REAL2 volumeCenter_localVoxelcenter_offset
            = (REAL2)(2 * i + lis - vdims.x, 2 * j + ljs - vdims.y) * halfVoxelSizes;
        const REAL2 voxelcenter_local_xy = volumeCenter + volumeCenter_localVoxelcenter_offset;
        const REAL2 halfLocalSizes = { HALF * lis * voxelSizes.x, HALF * ljs * voxelSizes.y };
        REAL LPX0 = PB2DPROJECT(CM, voxelcenter_local_xy);
        REAL LPXincTotal = dot(fabs(CM.s01), halfLocalSizes.s01);
        REAL LPX_min = LPX0 - LPXincTotal; // At minimum
        REAL LPX_max = LPX0 + LPXincTotal; // maximum
        LocalMemory_PIStart_LOC = INDEX(LPX_min + zeroPrecisionTolerance);
        LocalFootprint_PIUpperBound_LOC = INDEX(LPX_max - zeroPrecisionTolerance) + 1;
        if(LocalFootprint_PIUpperBound_LOC <= 0 || LocalMemory_PIStart_LOC >= pdims.x)
        {
            stopNextIteration_LOC = true;
            LocalMemory_PICount_LOC = 0;
            LocalFootprint_PIUpperBound_LOC = 0;
            LocalMemory_PIStart_LOC = 0;
        } else
        {
            if(LocalMemory_PIStart_LOC < 0)
            {
                LocalMemory_PIStart_LOC = 0;
            }
            if(LocalFootprint_PIUpperBound_LOC > pdims.x)
            {
                LocalFootprint_PIUpperBound_LOC = pdims.x;
            }
            // Prepare local memory
            LocalMemory_PICount_LOC = LocalFootprint_PIUpperBound_LOC - LocalMemory_PIStart_LOC;
            // How many columns fits to local memory
            uint LocalFootprint_size = LocalMemory_PICount_LOC * k_count;
            if(LocalFootprint_size <= LOCALARRAYSIZE)
            {
                stopNextIteration_LOC = true;
            } else
            {
                LocalMemory_PICount_LOC
                    = LOCALARRAYSIZE / k_count; // How many columns fits to local memory
                stopNextIteration_LOC = false;
            }
        }
        Lpdims_LOC = (int2)(LocalMemory_PICount_LOC, k_count);
        LocalMemory_Size_LOC = LocalMemory_PICount_LOC * k_count;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(Lpdims_LOC.x <= 0)
        return;
    int2 Lpdims;
    int PI_min_local;
    int PI_max_local;
    float PX_min_local;
    uint fillStart, fillStop, LIDRANGE;
    LIDRANGE = (LocalMemory_Size_LOC + LOCSIZE - 1) / LOCSIZE;
    fillStart = min(LID * LIDRANGE, LocalMemory_Size_LOC);
    fillStop = min(fillStart + LIDRANGE, LocalMemory_Size_LOC);
    uint LI, LJ, globalIndex, globalOffset;
    float val;
    // Do not care overfilling last array
    do
    {
        Lpdims = Lpdims_LOC;
        LocalMemory_PIStart = LocalMemory_PIStart_LOC;
        PI_min_local = PI_min - LocalMemory_PIStart;
        PI_max_local = PI_max - LocalMemory_PIStart;
        PX_min_local = PX_min - LocalMemory_PIStart;
        for(uint IND = fillStart; IND != fillStop; IND++)
        {
            localProjection[IND] = 0.0f;
        }
        stopThisIteration = stopNextIteration_LOC;
        barrier(CLK_LOCAL_MEM_FENCE);
        if(PX_inc1 == ZERO)
        {
            FLOAT_pbct2d_cutting_voxel_project_barrier_local_pxinc1iszero(
                voxelPointer, localProjection, Lpdims, voxelVolumeTimesScalingFactor, PI_min_local,
                PI_max_local, PX_min_local, PX_inc2, volumeStride);
        } else
        {
            FLOAT_pbct2d_cutting_voxel_project_barrier_local(
                voxelPointer, localProjection, Lpdims, voxelVolumeTimesScalingFactor, PI_min_local,
                PI_max_local, PX_min_local, PX_inc1, PX_inc2, PX_inc3, volumeStride);
        }
        barrier(CLK_LOCAL_MEM_FENCE); // Local to global copy
        globalOffset = LocalMemory_PIStart * pdims.y + k_from;
        for(uint IND = fillStart; IND != fillStop; IND++)
        // for(int IND = LID; IND < mappedLocalRange; IND+=LIDRANGE)
        {
            LI = IND / Lpdims.y;
            LJ = IND % Lpdims.y;
            globalIndex = globalOffset + LI * pdims.y + LJ;
            val = localProjection[IND];
            AtomicAdd_g_f(projection + globalIndex, val);
        }

        if(!stopThisIteration)
        {
            if(LID == 0)
            {
                LocalMemory_PIStart_LOC += LocalMemory_PICount_LOC;
                // if(LocalMemory_PIStart_LOC < LocalFootprint_PIUpperBound_LOC) .. shall be allways
                // true because stopThisIteration == false
                //{
                if(LocalFootprint_PIUpperBound_LOC - LocalMemory_PIStart_LOC
                   <= LocalMemory_PICount_LOC)
                {
                    LocalMemory_PICount_LOC
                        = LocalFootprint_PIUpperBound_LOC - LocalMemory_PIStart_LOC;
                    stopNextIteration_LOC = true;
                    Lpdims_LOC.x = LocalMemory_PICount_LOC;
                    LocalMemory_Size_LOC = LocalMemory_PICount_LOC * k_count;
                }
                //}
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            fillStart = min(fillStart, LocalMemory_Size_LOC);
            fillStop = min(fillStop, LocalMemory_Size_LOC);
        }
    } while(!stopThisIteration);
}
