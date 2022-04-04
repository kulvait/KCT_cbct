//==============================pbct_cvp.cl=====================================
#ifndef zeroPrecisionTolerance
#define zeroPrecisionTolerance 1e-10
#endif

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
void kernel FLOAT_pbct_cutting_voxel_project(global const float* restrict volume,
                                             global float* restrict projection,
                                             private ulong projectionOffset,
                                             private double8 _CM,
                                             private int3 vdims,
                                             private double3 _voxelSizes,
                                             private double3 _volumeCenter,
                                             private int2 pdims,
                                             private float scalingFactor)
{
    int i = get_global_id(2);
    int j = get_global_id(1);
    int k = get_global_id(0); // This is more effective from the perspective of atomic colisions
    projection += projectionOffset;
//_normalToDetector is not used in this implementation
#ifdef RELAXED
    const float16 CM = convert_float8(_CM);
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
    const float voxelValue = volume[IND];
    if(voxelValue == 0.0f)
    {
        return;
    }
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
    float value = voxelValue * voxelVolume * scalingFactor;
#else
    float value = (float)(voxelValue * voxelVolume * scalingFactor);
#endif
    // EXPERIMENTAL ... reconstruct inner circle
    /*   const double3 pixcoords = zerocorner_xyz + voxelSizes * (IND_ijk + (double3)(0.5, 0.5,
       0.5)); if(sqrt(pixcoords.x * pixcoords.x + pixcoords.y * pixcoords.y) > 110.0)
       {
           return;
       }*/
    // EXPERIMENTAL ... reconstruct inner circle
    // If all the corners of given voxel points to a common coordinate, then compute the value
    // based on the center
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
        if(max_PX <= min_PX) // These indices are in the admissible range
        {
            min_PX = convert_int_rtn(HALF * (pxx_min + pxx_max) + HALF);
            PBexactEdgeValues(projection, CM, (vx00 + vx11) * HALF, min_PX, value, voxelSizes,
                              pdims);
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
                factor = value * sectionSize_prev;
                Int = (REAL3)(CENTROID_prev, vx00.z);
                PBexactEdgeValues(projection, CM, Int, I, factor, voxelSizes, pdims);
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
                factor = value * polygonSize;
                PBexactEdgeValues(projection, CM, Int, I, factor, voxelSizes, pdims);
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
                factor = value * polygonSize;
                PBexactEdgeValues(projection, CM, Int, I, factor, voxelSizes, pdims);
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
void kernel FLOAT_pbct_cutting_voxel_backproject(global float* restrict volume,
                                                 global const float* restrict projection,
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
    // Not uint for correct int subtraction
    // Shift projection array by offset
    projection += projectionOffset;
//_normalToDetector is not used in this implementation
#ifdef RELAXED
    const float16 CM = convert_float16(_CM);
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

//==============================END pbct_cvp.cl=====================================
