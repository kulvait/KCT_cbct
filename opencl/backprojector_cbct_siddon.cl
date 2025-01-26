//==============================backprojector_cbct_siddon.cl=====================================

void kernel FLOATsidon_backproject(global float* restrict volume,
                                   global const float* restrict projection,
                                   private uint projectionOffset,
                                   private double16 ICM,
                                   private double3 sourcePosition,
                                   private double3 normalToDetector,
                                   private int3 vdims,
                                   private double3 voxelSizes,
                                   private double3 volumeCenter,
                                   private int2 pdims,
                                   private float scalingFactor,
                                   private uint2 raysPerPixel)
{
    uint px = get_global_id(0);
    uint py = get_global_id(1);
    double totalProbes = (double)raysPerPixel.x * raysPerPixel.y;
    uint pin = px * pdims.y + py;
    float VAL = projection[projectionOffset + pin];
    double2 pixelCorner = (double2)((double)px, (double)py) - (double2)0.5;
    double2 pixelSamplingGap = (double2)(1.0) / convert_double2(raysPerPixel);
    double4 P = { 0.0, 0.0, 1.0, 0.0 },
            V; // V is the point that will be projected to P by extended CM
    const double3 zerocorner_xyz = { volumeCenter.x - 0.5 * (double)vdims.x * voxelSizes.x,
                                     volumeCenter.y - 0.5 * (double)vdims.y * voxelSizes.y,
                                     volumeCenter.z - 0.5 * (double)vdims.z * voxelSizes.z };
    const double3 maxcorner_xyz = -zerocorner_xyz;
    for(uint pi = 0; pi < raysPerPixel.x; pi++)
    {
        P.x = pixelCorner.x + (pi + 0.5) * pixelSamplingGap.x;
        for(uint pj = 0; pj < raysPerPixel.y; pj++)
        {
            P.y = pixelCorner.y + (pj + 0.5) * pixelSamplingGap.y;
            V.s0 = dot(ICM.s0123, P);
            V.s1 = dot(ICM.s4567, P);
            V.s2 = dot(ICM.s89ab, P);
            V.s3 = dot(ICM.scdef, P);
            if(fabs(V.s3) < 0.001) // This is legal operation since source is projected to (0,0,0)
            {
                V.s012 = V.s012 + sourcePosition;
                V.s3 = V.s3 + 1.0;
            }
            V.s0 = V.s0 / V.s3;
            V.s1 = V.s1 / V.s3;
            V.s2 = V.s2 / V.s3;
            double3 a = normalize(V.s012 - sourcePosition);
            // Direction from the source to a given detector position sourcePosition   + alpha * a
            double cosine = dot(a, -sourcePosition);
            if(cosine < 0.0)
            {
                a = -a;
            }
            double minalpha = 0.0;
            double maxalpha = DBL_MAX;
            double minalphai, maxalphai, tmp;
            double3 cornera_minus_s = zerocorner_xyz - sourcePosition;
            double3 cornerb_minus_s = maxcorner_xyz - sourcePosition;
            double3 sidonIncrement = 0.0;
            double minSidonIncrement = DBL_MAX;
            double3 alphasPrev; // Previous intersection with the plane in given direction
            int maximalAlphasIndex; // Pointer to the element in alphasPrev that at maxalphai is
                                    // on the boundary
            if(a.x != 0.0)
            {
                sidonIncrement.x = fabs(voxelSizes.x / a.x);
                minSidonIncrement = sidonIncrement.x;
                // As I know cornera_minus_s < cornerb_minus_s
                if(a.x > 0.0)
                {
                    minalphai = cornera_minus_s.x / a.x;
                    maxalphai = cornerb_minus_s.x / a.x;
                } else
                {
                    maxalphai = cornera_minus_s.x / a.x;
                    minalphai = cornerb_minus_s.x / a.x;
                }
                minalpha = minalphai;
                maxalpha = maxalphai;
                alphasPrev.x = minalphai;
                maximalAlphasIndex = 0;
            }
            if(a.y != 0.0)
            {
                sidonIncrement.y = fabs(voxelSizes.y / a.y);
                minSidonIncrement = fmin(minSidonIncrement, sidonIncrement.y);
                if(a.y > 0)
                {
                    minalphai = cornera_minus_s.y / a.y;
                    maxalphai = cornerb_minus_s.y / a.y;
                } else
                {
                    maxalphai = cornera_minus_s.y / a.y;
                    minalphai = cornerb_minus_s.y / a.y;
                }
                if(minalphai > minalpha)
                {
                    minalpha = minalphai;
                    alphasPrev.y = minalphai;
                    if(a.x != 0.0)
                    {
                        // Naive implementation
                        // while(alphasPrev.x + sidonIncrement.x < minalpha)
                        // {
                        //     alphasPrev.x += sidonIncrement.x;
                        // }
                        if(minalpha - alphasPrev.x >= sidonIncrement.x)
                        {
                            alphasPrev.x += sidonIncrement.x
                                * floor((minalpha - alphasPrev.x) / sidonIncrement.x);
                        }
                    }
                } else
                {
                    alphasPrev.y = minalphai;
                    // Naive
                    // while(alphasPrev.y + sidonIncrement.y < minalpha)
                    //{
                    //    alphasPrev.y += sidonIncrement.y;
                    //}
                    if(minalpha - alphasPrev.y >= sidonIncrement.y)
                    {
                        alphasPrev.y += sidonIncrement.y
                            * floor((minalpha - alphasPrev.y) / sidonIncrement.y);
                    }
                }
                if(maxalphai < maxalpha)
                {
                    maxalpha = maxalphai;
                    maximalAlphasIndex = 1;
                }
            }
            if(a.z != 0.0)
            {
                sidonIncrement.z = fabs(voxelSizes.z / a.z);
                minSidonIncrement = fmin(minSidonIncrement, sidonIncrement.z);
                if(a.z > 0)
                {
                    minalphai = cornera_minus_s.z / a.z;
                    maxalphai = cornerb_minus_s.z / a.z;
                } else
                {
                    maxalphai = cornera_minus_s.z / a.z;
                    minalphai = cornerb_minus_s.z / a.z;
                }
                if(minalphai > minalpha)
                {
                    minalpha = minalphai;
                    alphasPrev.z = minalphai;
                    if(a.x != 0.0)
                    {
                        // Naive implementation
                        // while(alphasPrev.x + sidonIncrement.x < minalpha)
                        // {
                        //     alphasPrev.x += sidonIncrement.x;
                        // }
                        if(minalpha - alphasPrev.x >= sidonIncrement.x)
                        {
                            alphasPrev.x += sidonIncrement.x
                                * floor((minalpha - alphasPrev.x) / sidonIncrement.x);
                        }
                    }
                    if(a.y != 0.0)
                    {
                        // Naive
                        //    while(alphasPrev.y + sidonIncrement.y < minalpha)
                        //    {
                        //        alphasPrev.y += sidonIncrement.y;
                        //    }
                        if(minalpha - alphasPrev.y >= sidonIncrement.y)
                        {
                            alphasPrev.y += sidonIncrement.y
                                * floor((minalpha - alphasPrev.y) / sidonIncrement.y);
                        }
                    }
                } else
                {
                    alphasPrev.z = minalphai;
                    // Naive
                    // while(alphasPrev.z + sidonIncrement.z < minalpha)
                    //{
                    //    alphasPrev.z += sidonIncrement.z;
                    //}
                    if(minalpha - alphasPrev.z >= sidonIncrement.z)
                    {
                        alphasPrev.z += sidonIncrement.z
                            * floor((minalpha - alphasPrev.z) / sidonIncrement.z);
                    }
                }
                if(maxalphai < maxalpha)
                {
                    maxalpha = maxalphai;
                    maximalAlphasIndex = 2;
                }
            }
            double halfMinIncrement = minSidonIncrement * 0.5;
            double3 alphasNext = alphasPrev + sidonIncrement;

            double alphaprev = minalpha;
            double alphanext, LEN, pos;
            int3 ind;
            int IND;
            while(((double*)&alphasNext)[maximalAlphasIndex] - halfMinIncrement < maxalpha)
            {
                if(alphasNext.x < alphasNext.y)
                {
                    if(alphasNext.x < alphasNext.z)
                    {
                        alphanext = alphasNext.x;
                        alphasNext.x += sidonIncrement.x;
                    } else
                    {
                        alphanext = alphasNext.z;
                        alphasNext.z += sidonIncrement.z;
                    }
                } else if(alphasNext.y < alphasNext.z)
                {
                    alphanext = alphasNext.y;
                    alphasNext.y += sidonIncrement.y;
                } else
                {
                    alphanext = alphasNext.z;
                    alphasNext.z += sidonIncrement.z;
                }
                LEN = alphanext - alphaprev;
                pos = alphaprev + 0.5 * (alphanext - alphaprev);
                ind = convert_int3_rtn(
                    (sourcePosition + pos * a - zerocorner_xyz)
                    / voxelSizes); // Not rounding but finds integer that is closest smaller
                IND = ind.x + ind.y * vdims.x + ind.z * vdims.x * vdims.y;
                AtomicAdd_g_f(&volume[IND], VAL * LEN / totalProbes);
                // assert(all(ind >= (int3)(0, 0, 0)) && all(ind < vdims));
                alphaprev = alphanext;
            }
        }
    }
}
//==============================END backprojector_cbct_siddon.cl=====================================
