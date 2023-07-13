//==============================pbct2d_cvp.cl=====================================

#define PB2DPROJECT(CM, v) (dot(v, CM.s01) + CM.s2);

void inline PB2DInsertVerticalValues(global const float* restrict volumeBase,
                                     private uint volumeStride,
                                     global float* restrict projectionBase,
                                     private uint numValues,
                                     private float factor)
{
    float val;
    for(uint i = 0; i != numValues; i++)
    {
        val = volumeBase[i * volumeStride];
        AtomicAdd_g_f(projection + i, factor * val);
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
                                 const REAL* PX_xyx0,
                                 const REAL* PX_xyx1,
                                 const REAL* PX_xyx2,
                                 const REAL* PX_xyx3,
                                 const REAL3 CM)
{
    REAL FX = vd1 * CM.s0;
    REAL FY = vd3 * CM.s1;
    REAL DST;
    REAL p, q;
    REAL NAREA, narea_complement;
    if(PX < (*PX_xyx1))
    {
        DST = PX - (*PX_xyx0);
        if(FX)
        {
            p = DST / FX; // From v0 to v1
        } else
        {
            p = ONE;
        }
        if(PX < (*PX_xyx3))
        {
            if(FY)
                q = DST / FY; // From v0 to v3
            else
                q = ONE;
            NAREA = HALF * p * q;
            return NAREA;
        } else if(PX < (*PX_xyx2))
        {
            DST = PX - (*PX_xyx3);
            if(FX)
                q = DST / FX; // From v3 to v2
            else
                q = ONE;
            NAREA = ONEHALF * (p + q);
            return NAREA;
        } else
        {
            return ONE;
        }
    } else if(PX < (*PX_xyx2))
    {
        DST = PX - (*PX_xyx2);
        if(FY)
            p = DST / -FY; // v2 to v1
        else
            p = ZERO;
        if(PX < (*PX_xyx3))
        {
            p = ONE - p; // v1 to v2
            DST = PX - (*PX_xyx0);
            if(FY)
                q = DST / FY; // v0 to v3
            else
                q = ONE;
            NAREA = ONEHALF * (p + q);
            return NAREA;
        } else
        {
            if(FX)
                q = DST / -FX; // v2 to v3
            else
                q = ZERO;
            NAREA_complement = HALF * p * q;
            NAREA = ONE - NAREA_complement;
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
 * @param normalToDetector Normal to detector in the (i,j,k) space.
 * @param vdims Dimensions of the volume.
 * @param voxelSizes Lengths of the voxel edges.
 * @param pdims Dimensions of the projection.
 * @param scalingFactor Scale the results by this factor.
 *
 */
void kernel FLOAT_pbct2d_cutting_voxel_project(global const float* restrict volume,
                                               global float* restrict projection,
                                               private ulong projectionOffset,
                                               private double3 _CM,
                                               private int3 vdims,
                                               private double2 _voxelSizes,
                                               private double2 _volumeCenter,
                                               private int2 pdims,
                                               private float scalingFactor)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    projection += projectionOffset;
//_normalToDetector is not used in this implementation
#ifdef RELAXED
    const float3 CM = convert_float3(_CM);
    const float2 voxelSizes = convert_float3(_voxelSizes);
    const float2 volumeCenter = convert_float3(_volumeCenter);
#else
#define CM _CM
#define voxelSizes _voxelSizes
#define volumeCenter _volumeCenter
#endif
    const REAL2 halfVoxelSizes = HALF * voxelSizes;
    const REAL2 volumeCenter_voxelcenter_offset
        = (REAL2)(2 * i + 1 - vdims.x, 2 * j + 1 - vdims.y) * halfVoxelSizes;
    const REAL2 voxelcenter_xy = volumeCenter + volumeCenter_voxelcenter_offset;
    const ulong IND = voxelIndex(i, j, 0, vdims);
    float* voxelPointer = volume + IND;
    const int volumeStride = voxelIndex(0, 0, 1, vdims);
#ifdef DROPINCOMPLETEVOXELS
    int PINDEX = INDEX(PB2DPROJECTX(CM, voxelcenter_xy));
    if(PINDEX < 0 || PINDEX >= pdims.x)
    {
        return;
    } // More do further
#elif defined DROPCENTEROFFPROJECTORVOXELS // Here I need to do less, previous code sufficient
    int PINDEX = INDEX(PB2DPROJECTX(CM, voxelcenter_xy));
    if(PINDEX < 0 || PINDEX >= pdims.x)
    {
        return;
    } // More do further
#endif

    const REAL voxelVolume = voxelSizes.x * voxelSizes.y * scalingFactor;
    // If all the corners of given voxel points to a common coordinate, then compute the value
    // based on the center
    REAL px00, px10, px01, px11, xdiff, ydiff;
    REAL2 vx00, vx10, vx01, vx11; // Last is the voxel, where minimum PX is reached
    vx00 = voxelcenter_xyz + halfVoxelSizes * (REAL2)(-ONE, -ONE);
    vx10 = voxelcenter_xyz + halfVoxelSizes * (REAL2)(ONE, -ONE);
    vx01 = voxelcenter_xyz + halfVoxelSizes * (REAL2)(-ONE, ONE);
    vx11 = voxelcenter_xyz + halfVoxelSizes * (REAL2)(ONE, ONE);
    px00 = PB2DPROJECTX(CM, vx00);
    // These are PX increments
    xdiff = CM.s0 * voxelSizes.x;
    ydiff = CM.s1 * voxelSizes.y;
    // Linearity of PBCT
    px10 = px00 + xdiff;
    px01 = px00 + ydiff;
    px11 = px10 + ydiff;
    // We now figure out the vertex that projects to minimum and maximum px
    REAL pxx_min, pxx_max; // Minimum and maximum values of projector x coordinate
    int max_PX,
        min_PX; // Pixel to which are the voxels with minimum and maximum values are
                // projected
    // pxx_min = fmin(fmin(px00, px10), fmin(px01, px11));
    // pxx_max = fmax(fmax(px00, px10), fmax(px01, px11));
    REAL3* V0; // Point in which PX minimum is achieved
               // More formally we count V0, V1, V2, V3 as vertices so that VX1 is in the
               // corner that can be traversed in V0.xy plane by changing V0.x by voxelSizes.x so
               // that we are still on the voxel boundary
               // In the same manner are other points definned
               // V0, V1=V0+xshift, V2=V0+xshift+yshift, V3=V0+yshift
               // then we set up two distances
               // vd1 = V1->x-V0->x
               // vd3 = V3->x-V0->x
               // We do not define points V1, V2, V3 but just those differences
    REAL* PX_xyx[4]; // PX values in V0, V1, V2, V3
    REAL vd1, vd3;
    // REAL vd1 = V_xyx[1]->x - V_xyx[0]->x;
    // REAL vd3 = V_xyx[3]->y - V_xyx[0]->y;
    if(xdiff < 0)
    {
        if(ydiff < 0)
        {
            pxx_min = px11;
            pxx_max = px00;
            V0 = &vx11;
            vd1 = -voxelSizes.x;
            vd3 = -voxelSizes.y;
            PX_xyx[0] = &px11;
            PX_xyx[1] = &px01;
            PX_xyx[2] = &px00;
            PX_xyx[3] = &px10;
        } else
        {
            pxx_min = px10;
            pxx_max = px01;
            V0 = &vx10;
            vd1 = -voxelSizes.x;
            vd3 = voxelSizes.y;
            PX_xyx[0] = &px10;
            PX_xyx[1] = &px00;
            PX_xyx[2] = &px01;
            PX_xyx[3] = &px11;
        }
    } else
    {
        if(ydiff < 0)
        {
            pxx_min = px01;
            pxx_max = px10;
            V0 = &vx01;
            vd1 = voxelSizes.x;
            vd3 = -voxelSizes.y;
            PX_xyx[0] = &px01;
            PX_xyx[1] = &px11;
            PX_xyx[2] = &px10;
            PX_xyx[3] = &px00;
        } else
        {
            pxx_min = px00;
            pxx_max = px11;
            V0 = &vx00;
            vd1 = voxelSizes.x;
            vd3 = voxelSizes.y;
            PX_xyx[0] = &px00;
            PX_xyx[1] = &px10;
            PX_xyx[2] = &px11;
            PX_xyx[3] = &px01;
        }
    }

    min_PX = convert_int_rtn(pxx_min + zeroPrecisionTolerance + HALF);
    max_PX = convert_int_rtn(pxx_max - zeroPrecisionTolerance + HALF);
    float* pixelPointer;
    if(max_PX >= 0 && min_PX < pdims.x)
    {
        if(max_PX <= min_PX) // These indices are in the admissible range
        {
            min_PX = convert_int_rtn(HALF * (pxx_min + pxx_max) + HALF);
            pixelPointer = projection + min_PX * pdims.y;
            PB2DInsertVerticalValues(voxelPointer, volumeStride, pixelPointer, pdims.y,
                                     voxelVolume);
        } else
#ifdef DROPINCOMPLETEVOXELS
            if(min_PX < 0 || max_PX >= pdims.x)
        {
            return;
        } else
#endif
        {
            REAL sectionSize_prev, sectionSize_cur, polygonSize;
            REAL3 Int;
            REAL factor;
            int I = max(-1, min_PX);
            int I_STOP = min(max_PX, pdims.x);
            // Section of the square that corresponds to the indices < i
            sectionSize_prev = PB2DExactPolygonPart(((REAL)I) + HALF, vd1, vd3, PX_xyx[0],
                                                    PX_xyx[1], PX_xyx[2], PX_xyx[3], CM);
            if(I >= 0)
            {
                factor = voxelVolume * sectionSize_prev;
                pixelPointer = projection + I * pdims.y;
                PB2DInsertVerticalValues(voxelPointer, volumeStride, pixelPointer, pdims.y, factor);
            }
            for(I = I + 1; I < I_STOP; I++)
            {
                sectionSize_cur = PB2DExactPolygonPart(((REAL)I) + HALF, vd1, vd3, PX_xyx[0],
                                                       PX_xyx[1], PX_xyx[2], PX_xyx[3], CM);
                polygonSize = sectionSize_cur - sectionSize_prev;
                factor = voxelVolume * polygonSize;
                pixelPointer = projection + I * pdims.y;
                PB2DInsertVerticalValues(voxelPointer, volumeStride, pixelPointer, pdims.y, factor);
                sectionSize_prev = sectionSize_cur;
            }
            polygonSize = ONE - sectionSize_prev;
            // Without second test polygonsize==0 triggers division by zero
            if(I_STOP < pdims.x && polygonSize > zeroPrecisionTolerance)
            {
                factor = voxelVolume * polygonSize;
                pixelPointer = projection + I * pdims.y;
                PB2DInsertVerticalValues(voxelPointer, volumeStride, pixelPointer, pdims.y, factor);
            }
        }
    }
}

/** Adjoint operator to the FLOAT_pbct_cutting_voxel_project to perform backprojection.
 *
 *
 * @param volume Volume to project.
 * @param projection Projection to construct.
 * @param CM Projection matrix. This projection matrix is constructed in the way that (i,j,k) =
 * (0,0,0,1) is projected to the center of the voxel with given coordinates.
 * @param vdims Dimensions of the volume.
 * @param voxelSizes Lengths of the voxel edges.
 * @param pdims Dimensions of the projection.
 * @param scalingFactor Scale the results by this factor.
 *
 */
void kernel FLOAT_pbct2d_cutting_voxel_backproject(global float* restrict volume,
                                                   global const float* restrict projection,
                                                   private ulong projectionOffset,
                                                   private double3 _CM,
                                                   private int3 vdims,
                                                   private double2 _voxelSizes,
                                                   private double2 _volumeCenter,
                                                   private int2 pdims,
                                                   private float scalingFactor)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    // Not uint for correct int subtraction
    // Shift projection array by offset
    projection += projectionOffset;
//_normalToDetector is not used in this implementation
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
    const ulong IND = voxelIndex(i, j, k, vdims);
    float ADD = 0.0;

#ifdef DROPCENTEROFFPROJECTORVOXELS
    int xindex = INDEX(PBPROJECTX(CM, voxelcenter_xyz));
    int yindex = INDEX(PBPROJECTY(CM, voxelcenter_xyz));
    if(xindex < 0 || yindex < 0 || xindex >= pdims.x || yindex >= pdims.y)
    {
        return;
    }
#elif defined DROPINCOMPLETEVOXELS // Here I need to do more but when the previous code was executed
                                   // there is no need to do it twice
    int xindex = INDEX(PBPROJECTX(CM, voxelcenter_xyz));
    int yindex = INDEX(PBPROJECTY(CM, voxelcenter_xyz));
    if(xindex < 0 || yindex < 0 || xindex >= pdims.x || yindex >= pdims.y)
    {
        return;
    }
#endif

    const REAL voxelVolume = voxelSizes.x * voxelSizes.y * voxelSizes.z;
#ifdef RELAXED
    float value = voxelVolume * scalingFactor;
#else
    float value = (float)(voxelVolume * scalingFactor);
#endif
    REAL px00, px10, px01, px11, xdiff, ydiff;
    REAL3 vx00, vx10, vx01, vx11; // Last is the voxel, where minimum PX is reached
    vx00 = voxelcenter_xyz + halfVoxelSizes * (REAL3)(-ONE, -ONE, ZERO);
    vx10 = voxelcenter_xyz + halfVoxelSizes * (REAL3)(ONE, -ONE, ZERO);
    vx01 = voxelcenter_xyz + halfVoxelSizes * (REAL3)(-ONE, ONE, ZERO);
    vx11 = voxelcenter_xyz + halfVoxelSizes * (REAL3)(ONE, ONE, ZERO);
    px00 = PBPROJECTX(CM, vx00);
    // These are PX increments
    xdiff = CM.s0 * voxelSizes.x;
    ydiff = CM.s1 * voxelSizes.y;
    // Linearity of PBCT
    px10 = px00 + xdiff;
    px01 = px00 + ydiff;
    px11 = px10 + ydiff;
    // We now figure out the vertex that projects to minimum and maximum px
    REAL pxx_min, pxx_max; // Minimum and maximum values of projector x coordinate
    int max_PX,
        min_PX; // Pixel to which are the voxels with minimum and maximum values are
                // projected
    // pxx_min = fmin(fmin(px00, px10), fmin(px01, px11));
    // pxx_max = fmax(fmax(px00, px10), fmax(px01, px11));
    REAL3* V0; // Point in which PX minimum is achieved
               // More formally we count V0, V1, V2, V3 as vertices so that VX1 is in the
               // corner that can be traversed in V0.xy plane by changing V0.x by voxelSizes.x so
               // that we are still on the voxel boundary
               // In the same manner are other points definned
               // V0, V1=V0+xshift, V2=V0+xshift+yshift, V3=V0+yshift
               // then we set up two distances
               // vd1 = V1->x-V0->x
               // vd3 = V3->x-V0->x
               // We do not define points V1, V2, V3 but just those differences
    REAL* PX_xyx[4]; // PX values in V0, V1, V2, V3
    REAL vd1, vd3;
    // REAL vd1 = V_xyx[1]->x - V_xyx[0]->x;
    // REAL vd3 = V_xyx[3]->y - V_xyx[0]->y;
    if(xdiff < 0)
    {
        if(ydiff < 0)
        {
            pxx_min = px11;
            pxx_max = px00;
            V0 = &vx11;
            vd1 = -voxelSizes.x;
            vd3 = -voxelSizes.y;
            PX_xyx[0] = &px11;
            PX_xyx[1] = &px01;
            PX_xyx[2] = &px00;
            PX_xyx[3] = &px10;
        } else
        {
            pxx_min = px10;
            pxx_max = px01;
            V0 = &vx10;
            vd1 = -voxelSizes.x;
            vd3 = voxelSizes.y;
            PX_xyx[0] = &px10;
            PX_xyx[1] = &px00;
            PX_xyx[2] = &px01;
            PX_xyx[3] = &px11;
        }
    } else
    {
        if(ydiff < 0)
        {
            pxx_min = px01;
            pxx_max = px10;
            V0 = &vx01;
            vd1 = voxelSizes.x;
            vd3 = -voxelSizes.y;
            PX_xyx[0] = &px01;
            PX_xyx[1] = &px11;
            PX_xyx[2] = &px10;
            PX_xyx[3] = &px00;
        } else
        {
            pxx_min = px00;
            pxx_max = px11;
            V0 = &vx00;
            vd1 = voxelSizes.x;
            vd3 = voxelSizes.y;
            PX_xyx[0] = &px00;
            PX_xyx[1] = &px10;
            PX_xyx[2] = &px11;
            PX_xyx[3] = &px01;
        }
    }

    min_PX = convert_int_rtn(pxx_min + zeroPrecisionTolerance + HALF);
    max_PX = convert_int_rtn(pxx_max - zeroPrecisionTolerance + HALF);
    if(max_PX >= 0 && min_PX < pdims.x)
    {
#ifdef ELEVATIONCORRECTION
        REAL corlambda, corLenEstimate;
        // Typically voxelSizes.x == voxelSizes.y
        const REAL corLenLimit = HALF * (voxelSizes.x + voxelSizes.y);
#endif
        if(max_PX <= min_PX) // These indices are in the admissible range
        {
            min_PX = convert_int_rtn(HALF * (pxx_min + pxx_max) + HALF);
            ADD = PBbackprojectExactEdgeValues(projection, CM, HALF * (vx10 + vx01), min_PX,
                                               voxelSizes, pdims);
            volume[IND] += value * ADD;
        } else
#ifdef DROPINCOMPLETEVOXELS
            if(min_PX < 0 || max_PX >= pdims.x)
        {
            return;
        } else
#endif
        {
            REAL sectionSize_prev, sectionSize_cur, polygonSize;
            REAL3 Int;
            REAL factor;
            REAL2 CENTROID, CENTROID_cur, CENTROID_prev;
            int I = max(-1, min_PX);
            int I_STOP = min(max_PX, pdims.x);
            // Section of the square that corresponds to the indices < i
            sectionSize_prev
                = PBexactIntersectionPolygons(((REAL)I) + HALF, vd1, vd3, V0, PX_xyx[0], PX_xyx[1],
                                              PX_xyx[2], PX_xyx[3], CM, voxelSizes, &CENTROID_prev);
            if(I >= 0)
            {
                Int = (REAL3)(CENTROID_prev, vx00.z);
                factor = PBbackprojectExactEdgeValues(projection, CM, Int, I, voxelSizes, pdims);
                ADD += sectionSize_prev * factor;
            }
            for(I = I + 1; I < I_STOP; I++)
            {
                sectionSize_cur = PBexactIntersectionPolygons(
                    ((REAL)I) + HALF, vd1, vd3, V0, PX_xyx[0], PX_xyx[1], PX_xyx[2], PX_xyx[3], CM,
                    voxelSizes, &CENTROID_cur);
                polygonSize = sectionSize_cur - sectionSize_prev;
                CENTROID = (sectionSize_cur * CENTROID_cur - sectionSize_prev * CENTROID_prev)
                    / polygonSize;
                Int = (REAL3)(CENTROID, vx00.z);
                factor = PBbackprojectExactEdgeValues(projection, CM, Int, I, voxelSizes, pdims);
                ADD += polygonSize * factor;
                CENTROID_prev = CENTROID_cur;
                sectionSize_prev = sectionSize_cur;
            }
            polygonSize = ONE - sectionSize_prev;
            // Without second test polygonsize==0 triggers division by zero
            if(I_STOP < pdims.x && polygonSize > zeroPrecisionTolerance)
            {
                CENTROID_cur = V0->s01 + (REAL2)(HALF * vd1, HALF * vd3);
                CENTROID = (CENTROID_cur - sectionSize_prev * CENTROID_prev) / polygonSize;
                Int = (REAL3)(CENTROID, vx00.z);
                factor = PBbackprojectExactEdgeValues(projection, CM, Int, I, voxelSizes, pdims);
                ADD += polygonSize * factor;
            }
            volume[IND] += value * ADD;
        }
    }
}

//==============================END pbct2d_cvp.cl=====================================
