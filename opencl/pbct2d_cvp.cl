//==============================pbct2d_cvp.cl=====================================
void inline PB2DInsertVerticalValues(global const float* restrict volumePtr,
                                     private uint volumeStride,
                                     global float* restrict projectionPtr,
                                     private uint numValues,
                                     private float factor)
{
    float val;
    for(uint i = 0; i != numValues; i++)
    {
        val = volumePtr[i * volumeStride];
        AtomicAdd_g_f(projectionPtr + i, factor * val);
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
inline REAL PB2DExactPolygonPart(const REAL PX,
                                 const REAL vd1,
                                 const REAL vd3,
                                 const REAL PX_xyx0,
                                 const REAL PX_xyx1,
                                 const REAL PX_xyx2,
                                 const REAL PX_xyx3,
                                 const REAL3 CM)
{
    const REAL FX = vd1 * CM.s0;
    const REAL FY = vd3 * CM.s1;
    REAL DST;
    REAL p, q;
    REAL NAREA;
    if(PX < (PX_xyx1))
    {
        DST = PX - (PX_xyx0);
        if(FX)
        {
            p = DST / FX; // From v0 to v1
        } else
        {
            p = ONE;
        }
        if(PX < (PX_xyx3))
        {
            if(FY)
                q = DST / FY; // From v0 to v3
            else
                q = ONE;
            NAREA = HALF * p * q;
            return NAREA;
        } else if(PX < (PX_xyx2))
        {
            DST = PX - (PX_xyx3);
            if(FX)
                q = DST / FX; // From v3 to v2
            else
                q = ONE;
            NAREA = HALF * (p + q);
            return NAREA;
        } else
        {
            return ONE;
        }
    } else if(PX < (PX_xyx2))
    {
        DST = PX - (PX_xyx2);
        if(FY)
            p = DST / -FY; // v2 to v1
        else
            p = ZERO;
        if(PX < (PX_xyx3))
        {
            p = ONE - p; // v1 to v2
            DST = PX - (PX_xyx0);
            if(FY)
                q = DST / FY; // v0 to v3
            else
                q = ONE;
            NAREA = HALF * (p + q);
            return NAREA;
        } else
        {
            if(FX)
                q = DST / -FX; // v2 to v3
            else
                q = ZERO;
            // NAREA_complement = HALF * p * q;
            NAREA = ONE - HALF * p * q;
            return NAREA;
        }
    } else
    {
        return ONE;
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
void kernel FLOAT_pbct2d_cutting_voxel_project(global const float* restrict volume,
                                               global float* restrict projection,
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
    projection += projectionOffset;
//_normalToDetector is not used in this implementation
#ifdef RELAXED
    const float3 CM = convert_float3(_CM);
    const float3 voxelSizes = convert_float2(_voxelSizes);
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
    const ulong IND = voxelIndex(i, j, 0, vdims);
    const global float* voxelPointer = volume + IND;
    const int volumeStride = vdims.x * vdims.y;
    // Projected voxel center
    const REAL PX0 = PB2DPROJECT(CM, voxelcenter_xy);
    const int PINDEX = INDEX(PX0);
#ifdef DROPINCOMPLETEVOXELS
    if(PINDEX < 0 || PINDEX >= pdims.x)
    {
        return;
    } // More do further
#elif defined DROPCENTEROFFPROJECTORVOXELS // Here I need to do less, previous code sufficient
    if(PINDEX < 0 || PINDEX >= pdims.x)
    {
        return;
    } // More do further
#endif

    const REAL voxelVolumeTimesScalingFactor
        = voxelSizes.x * voxelSizes.y * voxelSizes.z * scalingFactor;
    // Let V0 be the vertex with minimal projection
    // More formally we count V0, V1, V2, V3 as vertices so that VX1 is in the
    // corner that can be traversed in V0.xy plane by changing V0.x by voxelSizes.x so
    // that we are still on the voxel boundary
    // In the same manner are other points definned
    // V0, V1=V0+xshift, V2=V0+xshift+yshift, V3=V0+yshift
    // then we set up two distances
    // vd1 = V1->x-V0->x
    // vd3 = V3->x-V0->x
    // We do not define points V1, V2, V3 but just those differences
    // We are interested in projected coordinates
    // [ PX_xyx0 =  PX_min,  PX_xyx0,  PX_xyx0,  PX_xyx0=PX_max] and
    // [PI_min, PI_max]
    REAL2 PXinc = fabs(CM.s01) * halfVoxelSizes.s01; // Increments for half voxel shifts
    REAL PXincTotal = PXinc.x + PXinc.y;
    REAL PX_xyx0 = PX0 - PXincTotal; // At minimum
    REAL PX_xyx1 = PX0 - PXinc.y + PXinc.x; // Minimum plus xshift
    REAL PX_xyx2 = PX0 + PXincTotal; // maximum
    REAL PX_xyx3 = PX0 + PXinc.y - PXinc.x; // Minimum plus yshift
    REAL PX_min = PX_xyx0;
    REAL PX_max = PX_xyx2;
    int PI_min = INDEX(PX_min + zeroPrecisionTolerance);
    int PI_max = INDEX(PX_max - zeroPrecisionTolerance);
    REAL vd1 = (CM.s0 > 0 ? voxelSizes.s0 : -voxelSizes.s0);
    REAL vd3 = (CM.s1 > 0 ? voxelSizes.s1 : -voxelSizes.s1);
    global float* restrict pixelPointer;
    if(PI_max >= 0 && PI_min < pdims.x)
    {
        if(PI_max <= PI_min) // These indices are in the admissible range
        {
            PI_min = convert_int_rtn(HALF * (PX_min + PX_max) + HALF);
            pixelPointer = projection + PI_min * pdims.y + k_from;
            PB2DInsertVerticalValues(voxelPointer, volumeStride, pixelPointer, k_count,
                                     voxelVolumeTimesScalingFactor);
            if(voxelVolumeTimesScalingFactor < 0.0)
            {
                printf("SINGLE PI (i,j)=(%d,%d) voxelVolumeTimesScalingFactor=%f\n", i, j,
                       voxelVolumeTimesScalingFactor);
            }
        } else
#ifdef DROPINCOMPLETEVOXELS
            if(PI_min < 0 || PI_max >= pdims.x)
        {
            return;
        } else
#endif
        {
            REAL sectionSize_prev, sectionSize_cur, polygonSize;
            REAL factor;
            int I = max(-1, PI_min);
            int I_STOP = min(PI_max, pdims.x);
            // Section of the square that corresponds to the indices < i
            sectionSize_prev = PB2DExactPolygonPart(((REAL)I) + HALF, vd1, vd3, PX_xyx0, PX_xyx1,
                                                    PX_xyx2, PX_xyx3, CM);
            if(I >= 0)
            {
                factor = voxelVolumeTimesScalingFactor * sectionSize_prev;
                pixelPointer = projection + I * pdims.y + k_from;
                PB2DInsertVerticalValues(voxelPointer, volumeStride, pixelPointer, k_count, factor);
                if(factor < 0.0)
                {
                    printf("IF (i,j)=(%d,%d) voxelVolumeTimesScalingFactor=%f\n", i, j, factor);
                    printf("voxelVolumeTimesScalingFactor=%f voxelSizes=(%f, %f, %f) scalingFactor=%f\n", voxelVolumeTimesScalingFactor, voxelSizes.x, voxelSizes.y, voxelSizes.z, scalingFactor);
                    printf("sectionSize_prev=%f PX_min=%f PX_max=%f PI_min=%d I=%d CM=(%f, %f, %f)\n", sectionSize_prev, PX_min, PX_max, PI_min, I, CM.s0, CM.s1, CM.s2);
                }
            }
            for(I = I + 1; I < I_STOP; I++)
            {
                sectionSize_cur = PB2DExactPolygonPart(((REAL)I) + HALF, vd1, vd3, PX_xyx0, PX_xyx1,
                                                       PX_xyx2, PX_xyx3, CM);
                polygonSize = sectionSize_cur - sectionSize_prev;
                factor = voxelVolumeTimesScalingFactor * polygonSize;
                pixelPointer = projection + I * pdims.y + k_from;
                PB2DInsertVerticalValues(voxelPointer, volumeStride, pixelPointer, k_count, factor);
                sectionSize_prev = sectionSize_cur;
                if(factor < 0.0)
                {
                    printf("FOR (i,j)=(%d,%d) voxelVolumeTimesScalingFactor=%f\n", i, j, factor);
                }
            }
            polygonSize = ONE - sectionSize_prev;
            // Without second test polygonsize==0 triggers division by zero
            if(I_STOP < pdims.x && polygonSize > zeroPrecisionTolerance)
            {
                factor = voxelVolumeTimesScalingFactor * polygonSize;
                pixelPointer = projection + I * pdims.y + k_from;
                PB2DInsertVerticalValues(voxelPointer, volumeStride, pixelPointer, k_count, factor);
                if(factor < 0.0)
                {
                    printf("END (i,j)=(%d,%d) voxelVolumeTimesScalingFactor=%f\n", i, j, factor);
                }
            }
        }
    }
}

// Default size of the array to store ADD values
#define KCOUNT 10
/** Adjoint operator to the FLOAT_pbct_cutting_voxel_project to perform backprojection.
 *
 *
 * @param volume Volume to project.
 * @param projection Projection to construct.
 * @param CM Projection matrix. This projection matrix is constructed in the way that (i,j,k) =
 * (0,0,0,1) is projected to the center of the voxel with given coordinates.
 * @param vdims Dimensions of the volume.
 * @param voxelSizes Lengths of the voxel edges, including third dimension.
 * @param pdims Dimensions of the projection.
 * @param scalingFactor Scale the results by this factor.
 *
 */
void kernel FLOAT_pbct2d_cutting_voxel_backproject(global float* restrict volume,
                                                   global const float* restrict projection,
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
    projection += projectionOffset;
//_normalToDetector is not used in this implementation
#ifdef RELAXED
    const float3 CM = convert_float3(_CM);
    const float3 voxelSizes = convert_float2(_voxelSizes);
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
    const ulong IND = voxelIndex(i, j, 0, vdims);
    const global float* voxelPointer = volume + IND;
    const int volumeStride = vdims.x * vdims.y;
    // Projected voxel center
    const REAL PX0 = PB2DPROJECT(CM, voxelcenter_xy);
    const int PINDEX = INDEX(PX0);
#ifdef DROPINCOMPLETEVOXELS
    if(PINDEX < 0 || PINDEX >= pdims.x)
    {
        return;
    } // More do further
#elif defined DROPCENTEROFFPROJECTORVOXELS // Here I need to do less, previous code sufficient
    if(PINDEX < 0 || PINDEX >= pdims.x)
    {
        return;
    } // More do further
#endif

    const REAL voxelVolumeTimesScalingFactor
        = voxelSizes.x * voxelSizes.y * voxelSizes.z * scalingFactor;
    // Let V0 be the vertex with minimal projection
    // More formally we count V0, V1, V2, V3 as vertices so that VX1 is in the
    // corner that can be traversed in V0.xy plane by changing V0.x by voxelSizes.x so
    // that we are still on the voxel boundary
    // In the same manner are other points definned
    // V0, V1=V0+xshift, V2=V0+xshift+yshift, V3=V0+yshift
    // then we set up two distances
    // vd1 = V1->x-V0->x
    // vd3 = V3->x-V0->x
    // We do not define points V1, V2, V3 but just those differences
    // We are interested in projected coordinates
    // [ PX_xyx0 =  PX_min,  PX_xyx0,  PX_xyx0,  PX_xyx0=PX_max] and
    // [PI_min, PI_max]
    REAL2 PXinc = fabs(CM.s01) * halfVoxelSizes.s01; // Increments for half voxel shifts
    REAL PXincTotal = PXinc.x + PXinc.y;
    REAL PX_xyx0 = PX0 - PXincTotal; // At minimum
    REAL PX_xyx1 = PX0 - PXinc.y + PXinc.x; // Minimum plus xshift
    REAL PX_xyx2 = PX0 + PXincTotal; // maximum
    REAL PX_xyx3 = PX0 + PXinc.y - PXinc.x; // Minimum plus yshift
    REAL PX_min = PX_xyx0;
    REAL PX_max = PX_xyx2;
    int PI_min = INDEX(PX_min + zeroPrecisionTolerance);
    int PI_max = INDEX(PX_max - zeroPrecisionTolerance);
    REAL vd1 = (CM.s0 > 0 ? voxelSizes.s0 : -voxelSizes.s0);
    REAL vd3 = (CM.s1 > 0 ? voxelSizes.s1 : -voxelSizes.s1);
    global float* pixelPointer;
    float ADD[KCOUNT]; // uninitialized by default
    int k_processed = 0;
    int k_to = k_from + k_count;
    if(PI_max >= 0 && PI_min < pdims.x)
    {
        // Handle k_count > 1

        int k_from_loc = k_from;
        int k_count_loc = min(KCOUNT, k_to - k_from_loc);
        while(k_from_loc < k_to)
        {
            for(int k = 0; k != k_count_loc; k++)
            {
                ADD[k] = 0.0f;
            }
            if(PI_max <= PI_min) // These indices are in the admissible range
            {
                PI_min = convert_int_rtn(HALF * (PX_min + PX_max) + HALF);
                pixelPointer = projection + PI_min * pdims.y + k_from_loc;
                for(int k = 0; k != k_count_loc; k++)
                {
                    ADD[k] = pixelPointer[k] * voxelVolumeTimesScalingFactor;
                }
            } else
#ifdef DROPINCOMPLETEVOXELS
                if(PI_min < 0 || PI_max >= pdims.x)
            {
                return;
            } else
#endif
            {
                REAL sectionSize_prev, sectionSize_cur, polygonSize;
                REAL factor;
                int I = max(-1, PI_min);
                int I_STOP = min(PI_max, pdims.x);
                // Section of the square that corresponds to the indices < i
                sectionSize_prev = PB2DExactPolygonPart(((REAL)I) + HALF, vd1, vd3, PX_xyx0,
                                                        PX_xyx1, PX_xyx2, PX_xyx3, CM);
                if(I >= 0)
                {
                    factor = voxelVolumeTimesScalingFactor * sectionSize_prev;
                    pixelPointer = projection + I * pdims.y + k_from_loc;
                    for(int k = 0; k != k_count_loc; k++)
                    {
                        ADD[k] += pixelPointer[k] * factor;
                    }
                }
                for(I = I + 1; I < I_STOP; I++)
                {
                    sectionSize_cur = PB2DExactPolygonPart(((REAL)I) + HALF, vd1, vd3, PX_xyx0,
                                                           PX_xyx1, PX_xyx2, PX_xyx3, CM);
                    polygonSize = sectionSize_cur - sectionSize_prev;
                    factor = voxelVolumeTimesScalingFactor * polygonSize;
                    pixelPointer = projection + I * pdims.y + k_from_loc;
                    for(int k = 0; k != k_count_loc; k++)
                    {
                        ADD[k] += pixelPointer[k] * factor;
                    }
                    sectionSize_prev = sectionSize_cur;
                }
                polygonSize = ONE - sectionSize_prev;
                // Without second test polygonsize==0 triggers division by zero
                if(I_STOP < pdims.x)
                {
                    factor = voxelVolumeTimesScalingFactor * polygonSize;
                    pixelPointer = projection + I * pdims.y + k_from_loc;
                    for(int k = 0; k != k_count_loc; k++)
                    {
                        ADD[k] += pixelPointer[k] * factor;
                    }
                }
            }
            for(int k = 0; k != k_count_loc; k++)
            {
                if(ADD[k] != 0)
                {
                    volume[IND + k * volumeStride] += ADD[k];
                }
            }
            k_from_loc += k_count_loc;
            k_count_loc = min(KCOUNT, k_to - k_from_loc);
        }
    }
}
//==============================END pbct2d_cvp.cl=====================================
