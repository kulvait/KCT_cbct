//==============================pbct_cvp_barrier.cl=====================================
#ifndef zeroPrecisionTolerance
#define zeroPrecisionTolerance 1e-10
#endif

/** Project given volume using cutting voxel projector and parallel rays geometry, barrier
 * implementation.
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
void kernel FLOAT_pbct_cutting_voxel_project_barrier(global const float* restrict volume,
                                                     global float* restrict projection,
                                                     local float* restrict localProjection,
                                                     private ulong projectionOffset,
                                                     private double8 _CM,
                                                     private int3 vdims,
                                                     private double3 _voxelSizes,
                                                     private double3 _volumeCenter,
                                                     private int2 pdims,
                                                     private float scalingFactor)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);
    int li = get_local_id(0);
    int lj = get_local_id(1);
    int lk = get_local_id(2);
    int lis = get_local_size(0);
    int ljs = get_local_size(1);
    int lks = get_local_size(2);
    // Not uint for correct int subtraction
    // Shift projection array by offset
    projection += projectionOffset;
    uint LOCSIZE = lis * ljs * lks;
    uint LID = li * ljs * lks + lj * lks + lk;
    uint LIDRANGE, fillStart, fillStop;
    int2 Lpdims;
    uint mappedLocalRange, Jrange, ILocalRange; // Memory used only in cornerWorkItem
    local bool partlyOffProjectorPosition; // If true, some vertices of the local cuboid are
                                           // projected outside the projector
    local bool fullyOffProjectorPosition; // If true, shall end the execution
    local bool stopNextIteration_local;
    bool stopNextIteration;
    local REAL8 CML;
    local int projectorLocalRange[7]; //
    int PILocalMin, PILocalMax, PJLocalMin, PJLocalMax;
#ifdef RELAXED
    const float8 CM = convert_float8(_CM);
    const float3 voxelSizes = convert_float3(_voxelSizes);
    const float3 volumeCenter = convert_float3(_volumeCenter);
#else
#define CM _CM
#define voxelSizes _voxelSizes
#define volumeCenter _volumeCenter
#endif
    const REAL3 halfVoxelSizes = HALF * voxelSizes;
    const REAL3 volumeCenter_voxelcenter_offset
        = (REAL3)(2 * i + 1 - vdims.x, 2 * j + 1 - vdims.y, 2 * k + 1 - vdims.z) * halfVoxelSizes;
    const REAL3 voxelcenter_xyz = volumeCenter + volumeCenter_voxelcenter_offset;
    if(LID == 0) // Get dimension
    {
        // LID==0
        const REAL3 volumeCenter_localVoxelcenter_offset
            = (REAL3)(2 * i + lis - vdims.x, 2 * j + ljs - vdims.y, 2 * k + lks - vdims.z)
            * halfVoxelSizes;
        /*
            // LID==LIX
            const REAL3 volumeCenter_localVoxelcenter_offset = (REAL3)(2 * i + 2 - lis -
                        vdims.x, 2 * j + 2 - ljs - vdims.y, 2 * k + 2 - lks - vdims.z) *
                                        halfVoxelSizes;
        */
        const REAL3 voxelcenter_local_xyz = volumeCenter + volumeCenter_localVoxelcenter_offset;
        const REAL3 halfLocalSizes
            = { HALF * lis * voxelSizes.x, HALF * ljs * voxelSizes.y, HALF * lks * voxelSizes.z };
        REAL LPX = PBPROJECTX(CM, voxelcenter_local_xyz);
        REAL LPY = PBPROJECTY(CM, voxelcenter_local_xyz);
        REAL LPXinc = dot(fabs(CM.s012), halfLocalSizes);
        REAL LPYinc = dot(fabs(CM.s456), halfLocalSizes);
        PILocalMin = INDEX(LPX - LPXinc);
        PILocalMax = INDEX(LPX + LPXinc);
        PJLocalMin = INDEX(LPY - LPYinc);
        PJLocalMax = INDEX(LPY + LPYinc);
        if(PILocalMax < 0 || PILocalMin >= pdims.x || PJLocalMax < 0 || PJLocalMin >= pdims.y)
        {
            fullyOffProjectorPosition = true;
            partlyOffProjectorPosition = true;
        } else if(PILocalMin < 0 || PILocalMax >= pdims.x || PJLocalMin < 0
                  || PJLocalMax >= pdims.y)
        {
            fullyOffProjectorPosition = false;
            partlyOffProjectorPosition = true;
            projectorLocalRange[0] = max(0, PILocalMin);
            projectorLocalRange[1] = min(pdims.x, PILocalMax + 1);
            projectorLocalRange[2] = max(0, PJLocalMin);
            projectorLocalRange[3] = min(pdims.y, PJLocalMax + 1);
        } else
        {
            fullyOffProjectorPosition = false;
            partlyOffProjectorPosition = false;
            projectorLocalRange[0] = PILocalMin;
            projectorLocalRange[1] = PILocalMax + 1;
            projectorLocalRange[2] = PJLocalMin;
            projectorLocalRange[3] = PJLocalMax + 1;
        }
        // Prepare local memory
        if(!fullyOffProjectorPosition)
        {
            uint Irange = projectorLocalRange[1] - projectorLocalRange[0];
            uint Jrange = projectorLocalRange[3] - projectorLocalRange[2];
            uint FullLocalRange = Irange * Jrange;
            if(FullLocalRange <= LOCALARRAYSIZE)
            {
                ILocalRange = Irange; // How many columns fits to local memory
                stopNextIteration_local = true;
            } else
            {
                ILocalRange = LOCALARRAYSIZE / Jrange;
                stopNextIteration_local = false;
                // printf("%zu %zu %zu", i, j, k);
                // printf("FullLocalRange=%zu exceeds range\n", FullLocalRange);
            }
            projectorLocalRange[4] = ILocalRange;
            projectorLocalRange[5] = Jrange;
            projectorLocalRange[6]
                = projectorLocalRange[0]; // Where current local array has start IRange

            // 0..PIMIN, 1..PIMAX, 2..PJMIN, 3..PJMAX, 4 .. ILocalRange, 5 .. PJMAX-PJMIN, 6
            // CurrentPISTART
            CML = CM;
            CML.s3 -= projectorLocalRange[0];
            CML.s7 -= projectorLocalRange[2];
        } else
        {
            projectorLocalRange[1] = -1;
            projectorLocalRange[2] = 0;
            projectorLocalRange[3] = 0;
            projectorLocalRange[4] = 0;
            projectorLocalRange[5] = 0;
            projectorLocalRange[6] = 0;
            stopNextIteration_local = true;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE); // Cutting voxel projector
                                  // 0..PIMIN, 1..PIMAX, 2..PJMIN, 3..PJMAX, 4 .. ILocalRange, 5
                                  // .. PJMAX-PJMIN, 6 CurrentPISTART
    if(fullyOffProjectorPosition)
        return;
    Lpdims = (int2)(projectorLocalRange[4], projectorLocalRange[5]);
    mappedLocalRange = Lpdims.x * Lpdims.y;
    LIDRANGE = (mappedLocalRange + LOCSIZE - 1) / LOCSIZE;
    fillStart = min(LID * LIDRANGE, mappedLocalRange);
    // minimum for LIDRANGE=1 and LID >= mappedLocalRange
    fillStop = min(fillStart + LIDRANGE, mappedLocalRange);
    for(uint IND = fillStart; IND != fillStop; IND++)
    {
        localProjection[IND] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint prevStartRange;
    const uint IND = voxelIndex(i, j, k, vdims);
    const float voxelValue = volume[IND];
    bool dropVoxel = false;
    if(voxelValue == 0.0f)
    {
        dropVoxel = true;
    }
#ifdef DROPINCOMPLETEVOXELS
    int xindex = INDEX(PBPROJECTX(CM, voxelcenter_xyz));
    int yindex = INDEX(PBPROJECTY(CM, voxelcenter_xyz));
    if(xindex < 0 || yindex < 0 || xindex >= pdims.x || yindex >= pdims.y)
    {
        dropVoxel = true;
    } // more do further
#elif defined DROPCENTEROFFPROJECTORVOXELS // Here I need to do less, previous code sufficient
    int xindex = INDEX(PBPROJECTX(CM, voxelcenter_xyz));
    int yindex = INDEX(PBPROJECTY(CM, voxelcenter_xyz));
    if(xindex < 0 || yindex < 0 || xindex >= pdims.x || yindex >= pdims.y)
    {
        dropVoxel = true;
    }
#endif
    const REAL voxelVolume = voxelSizes.x * voxelSizes.y * voxelSizes.z;
#ifdef RELAXED
    float value = (voxelValue * voxelVolume * scalingFactor);
#else
    float value = (float)(voxelValue * voxelVolume * scalingFactor);
#endif
    do
    {
        stopNextIteration = stopNextIteration_local;
        prevStartRange = projectorLocalRange[6];
        // printf("i,j,k=%d,%d,%d", i, j, k);
        // Start CVP
        if(!dropVoxel)
        {
            REAL PXL = PBPROJECTX(CML, voxelcenter_xyz);
            REAL PYL = PBPROJECTY(CML, voxelcenter_xyz); // Here I assume that PYL is the same when
                                                         // changing x and y throughout the voxel
            REAL2 PXinc
                = fabs(CM.s01) * halfVoxelSizes.s01; // Investigate minmax only in XY directions
            REAL PXincTotal = PXinc.x + PXinc.y;
            int PIL_min = INDEX(PXL - PXincTotal + zeroPrecisionTolerance);
            int PIL_max = INDEX(PXL + PXincTotal - zeroPrecisionTolerance);
            local float* localProjectionColumn;
            const REAL PYinc = fabs(CM.s6) * halfVoxelSizes.s2;
            REAL PYL_min = PYL - PYinc;
            REAL PYL_max = PYL + PYinc;
            int PJL_min = INDEX(PYL_min + zeroPrecisionTolerance);
            int PJL_max = INDEX(PYL_max - zeroPrecisionTolerance);
            int J;
            if(PIL_max >= 0 && PIL_min < Lpdims.x && PJL_max >= 0 && PJL_min < Lpdims.y)
            {
                int PJL_min_eff, PJL_max_eff;
                float lambdaIncrement, firstLambdaIncrement, leastLambdaIncrement;
                lambdaIncrement = HALF / PYinc;
                if(PJL_max >= Lpdims.y)
                {
                    PJL_max_eff = Lpdims.y - 1;
                    leastLambdaIncrement = lambdaIncrement;
                } else
                {
                    PJL_max_eff = PJL_max;
                    leastLambdaIncrement = (PYL_max - (PJL_max - HALF)) / (TWO * PYinc);
                }
                if(PJL_min < 0)
                {
                    PJL_min_eff = 0;
                    firstLambdaIncrement = lambdaIncrement;
                } else
                {
                    PJL_min_eff = PJL_min;
                    firstLambdaIncrement = (PJL_min + HALF - PYL_min) / (TWO * PYinc);
                }
                if(PIL_max
                   <= PIL_min) // These indices are in the admissible range, effectivelly same index
                {
                    PIL_min = INDEX(PXL);
                    // Unfolded localEdgeValues
                    localProjectionColumn = localProjection + PIL_min * Lpdims.y;
                    if(PJL_max <= PJL_min) // These indices are in the admissible range,
                                           // effectivelly same index
                    {
                        PJL_min = INDEX(PYL);
                        AtomicAdd_l_f(localProjectionColumn + PJL_min, value);
                    } else
                    {
                        J = PJL_min_eff;
                        AtomicAdd_l_f(localProjectionColumn + J, firstLambdaIncrement * value);
                        for(J = J + 1; J < PJL_max_eff; J++)
                        {
                            AtomicAdd_l_f(localProjectionColumn + J, lambdaIncrement * value);
                        }
                        AtomicAdd_l_f(localProjectionColumn + PJL_max_eff,
                                      leastLambdaIncrement * value);
                    }
                } else
                {
                    REAL vd1, vd3;
                    REAL PX_xyx0 = PXL - PXincTotal; // At minimum
                    REAL PX_xyx1 = PXL - PXinc.y + PXinc.x; // Minimum plus xshift
                    REAL PX_xyx2 = PXL + PXincTotal; // maximum
                    REAL PX_xyx3 = PXL + PXinc.y - PXinc.x; // Minimum plus yshift
                    REAL3 V0;
                    if(CM.s0 < 0)
                    {
                        if(CM.s1 < 0)
                        {
                            V0 = voxelcenter_xyz + halfVoxelSizes * (REAL3)(ONE, ONE, ZERO);
                            vd1 = -voxelSizes.x;
                            vd3 = -voxelSizes.y;
                        } else
                        {
                            V0 = voxelcenter_xyz + halfVoxelSizes * (REAL3)(ONE, -ONE, ZERO);
                            vd1 = -voxelSizes.x;
                            vd3 = voxelSizes.y;
                        }
                    } else
                    {
                        if(CM.s1 < 0)
                        {
                            V0 = voxelcenter_xyz + halfVoxelSizes * (REAL3)(-ONE, ONE, ZERO);
                            vd1 = voxelSizes.x;
                            vd3 = -voxelSizes.y;
                        } else
                        {
                            V0 = voxelcenter_xyz + halfVoxelSizes * (REAL3)(-ONE, -ONE, ZERO);
                            vd1 = voxelSizes.x;
                            vd3 = voxelSizes.y;
                        }
                    }
                    REAL sectionSize_prev, sectionSize_cur, polygonSize;
                    REAL3 Int;
                    REAL factor;
                    REAL2 CENTROID, CENTROID_cur, CENTROID_prev;
                    int I = max(-1, PIL_min);
                    int I_STOP = min(PIL_max, Lpdims.x);
                    // Section of the square that corresponds to the indices < i
                    sectionSize_prev = PBexactIntersectionPolygons(
                        ((REAL)I) + HALF, vd1, vd3, &V0, &PX_xyx0, &PX_xyx1, &PX_xyx2, &PX_xyx3,
                        CML, voxelSizes, &CENTROID_prev);
                    if(I >= 0)
                    {
                        factor = value * sectionSize_prev;
                        // Unfolded localEdgeValues
                        localProjectionColumn = localProjection + I * Lpdims.y;
                        if(PJL_max <= PJL_min) // These indices are in the admissible range,
                                               // effectivelly same index
                        {
                            PJL_min = INDEX(PYL);
                            AtomicAdd_l_f(localProjectionColumn + PJL_min, factor);
                        } else
                        {
                            J = PJL_min_eff;
                            AtomicAdd_l_f(localProjectionColumn + J, firstLambdaIncrement * factor);
                            for(J = J + 1; J < PJL_max_eff; J++)
                            {
                                AtomicAdd_l_f(localProjectionColumn + J, lambdaIncrement * factor);
                            }
                            AtomicAdd_l_f(localProjectionColumn + PJL_max_eff,
                                          leastLambdaIncrement * factor);
                        }
                    }
                    for(I = I + 1; I < I_STOP; I++)
                    {
                        sectionSize_cur = PBexactIntersectionPolygons(
                            ((REAL)I) + HALF, vd1, vd3, &V0, &PX_xyx0, &PX_xyx1, &PX_xyx2, &PX_xyx3,
                            CML, voxelSizes, &CENTROID_cur);
                        polygonSize = sectionSize_cur - sectionSize_prev;
                        factor = value * polygonSize;
                        // Unfolded localEdgeValues
                        localProjectionColumn = localProjection + I * Lpdims.y;
                        if(PJL_max <= PJL_min) // These indices are in the admissible range,
                                               // effectivelly same index
                        {
                            PJL_min = INDEX(PYL);
                            AtomicAdd_l_f(localProjectionColumn + PJL_min, factor);
                        } else
                        {
                            J = PJL_min_eff;
                            AtomicAdd_l_f(localProjectionColumn + J, firstLambdaIncrement * factor);
                            for(J = J + 1; J < PJL_max_eff; J++)
                            {
                                AtomicAdd_l_f(localProjectionColumn + J, lambdaIncrement * factor);
                            }
                            AtomicAdd_l_f(localProjectionColumn + PJL_max_eff,
                                          leastLambdaIncrement * factor);
                        }
                        sectionSize_prev = sectionSize_cur;
                    }
                    polygonSize = ONE - sectionSize_prev;
                    // Without second test polygonsize==0 triggers division by zero
                    if(I_STOP < Lpdims.x && polygonSize > zeroPrecisionTolerance)
                    {
                        factor = value * polygonSize;
                        // Unfolded localEdgeValues
                        localProjectionColumn = localProjection + PIL_max * Lpdims.y;
                        if(PJL_max <= PJL_min) // These indices are in the admissible range,
                                               // effectivelly same index
                        {
                            PJL_min = INDEX(PYL);
                            AtomicAdd_l_f(localProjectionColumn + PJL_min, factor);
                        } else
                        {
                            J = PJL_min_eff;
                            AtomicAdd_l_f(localProjectionColumn + J, firstLambdaIncrement * factor);
                            for(J = J + 1; J < PJL_max_eff; J++)
                            {
                                AtomicAdd_l_f(localProjectionColumn + J, lambdaIncrement * factor);
                            }
                            AtomicAdd_l_f(localProjectionColumn + PJL_max_eff,
                                          leastLambdaIncrement * factor);
                        }
                    }
                }
            }
        }
        // End CVP
        // 0..PIMIN, 1..PIMAX, 2..PJMIN, 3..PJMAX, 4 .. ILocalRange, 5 .. PJMAX-PJMIN, 6
        // CurrentPISTART
        barrier(CLK_LOCAL_MEM_FENCE); // Local to global copy
        uint LI, LJ, globalIndex;
        uint globalOffset = prevStartRange * pdims.y + projectorLocalRange[2];
        for(int IND = fillStart; IND != fillStop; IND++)
        // for(int IND = LID; IND < mappedLocalRange; IND+=LIDRANGE)
        {
            LI = IND / Lpdims.y;
            LJ = IND % Lpdims.y;
            globalIndex = globalOffset + LI * pdims.y + LJ;
            AtomicAdd_g_f(projection + globalIndex, localProjection[IND]);
        }
        if(LID == 0)
        {
            CML.s3 -= projectorLocalRange[4];
            // 0..PIMIN, 1..PIMAX, 2..PJMIN, 3..PJMAX, 4 .. PIMAX-PIMIN, 5 .. PJMAX-PJMIN, 6
            // CurrentPISTART
            projectorLocalRange[6] += projectorLocalRange[4];
            if(projectorLocalRange[6] < projectorLocalRange[1])
            {
                if(projectorLocalRange[1] - projectorLocalRange[6] <= projectorLocalRange[4])
                {
                    projectorLocalRange[4] = projectorLocalRange[1] - projectorLocalRange[6];
                    stopNextIteration_local = true;
                }
            }
        }

        if(!stopNextIteration)
        {
            barrier(CLK_LOCAL_MEM_FENCE);
            Lpdims = (int2)(projectorLocalRange[4], projectorLocalRange[5]);
            mappedLocalRange = Lpdims.x * Lpdims.y;
            fillStart = min(fillStart, mappedLocalRange);
            fillStop = min(fillStop, mappedLocalRange);
            // When LIDRANGE=1 there might be processes not to fill at all
            for(int IND = fillStart; IND != fillStop; IND++)
            {
                localProjection[IND] = 0.0f;
            }
            barrier(CLK_LOCAL_MEM_FENCE); // Cutting voxel projector
                                          // 0..PIMIN, 1..PIMAX, 2..PJMIN, 3..PJMAX, 4 ..
                                          // ILocalRange, 5
                                          // .. PJMAX-PJMIN, 6 CurrentPISTART

            // printf("Next %d %d %d. \n", i, j, k);
        }
    } while(!stopNextIteration);
}

//==============================END pbct_cvp_barrier.cl=====================================
