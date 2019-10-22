void kernel FLOATsidon_project(global float* volume,
                               global float* projection,
                               private uint projectionOffset,
                               private double16 ICM,
                               private double3 sourcePosition,
                               private double3 normalToDetector,
                               private int3 vdims,
                               private double3 voxelSizes,
                               private int2 pdims,
                               private float scalingFactor,
                               private uint2 raysPerPixel)
{
    uint px = get_global_id(0);
    uint py = get_global_id(1);
    double totalProbes = (double)raysPerPixel.x * raysPerPixel.y;
    double VAL = 0.0;
    double2 pixelCorner = (double2)((double)px, (double)py) - (double2)0.5;
    double2 pixelSamplingGap = (double2)(1.0) / convert_double2(raysPerPixel);
    double4 P = { 0.0, 0.0, 1.0, 0.0 },
            V; // V is the point that will be projected to P by extended CM
    const double3 zerocorner_xyz
        = { -0.5 * (double)vdims.x * voxelSizes.x, -0.5 * (double)vdims.y * voxelSizes.y,
            -0.5 * (double)vdims.z * voxelSizes.z }; // -convert_double3(vdims) / 2.0;
    const double3 maxcorner_xyz = -zerocorner_xyz;
    for(uint pi = 0; pi < raysPerPixel.x; pi++)
    {
        for(uint pj = 0; pj < raysPerPixel.y; pj++)
        {
            P.x = pixelCorner.x + (pi + 0.5) * pixelSamplingGap.x;
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
            double cosine = -dot(normalToDetector, a);
            if(cosine < 0.0)
            {
                a = -a;
            } // Direction from the source to a given detector position sourcePosition + alpha * a
            double minalpha = 0.0;
            double maxalpha = DBL_MAX;
            double minalphai, maxalphai, tmp;
            double3 cornera_minus_s = zerocorner_xyz - sourcePosition;
            double3 cornerb_minus_s = maxcorner_xyz - sourcePosition;
            double3 sidonIncrement = 0.0;
            double minSidonIncrement = DBL_MAX;
            double3 alphasPrev; // Previous intersection with the plane in given direction
            if(a.x != 0.0)
            {
                sidonIncrement.x = fabs(voxelSizes.x / a.x);
                minSidonIncrement = sidonIncrement.x;
                minalphai = cornera_minus_s.x / a.x;
                maxalphai = cornerb_minus_s.x / a.x;
                if(minalphai > maxalphai)
                {
                    tmp = minalphai;
                    minalphai = maxalphai;
                    maxalphai = tmp;
                }
                minalpha = minalphai;
                maxalpha = maxalphai;
                alphasPrev.x = minalphai;
            }
            if(a.y != 0.0)
            {
                sidonIncrement.y = fabs(voxelSizes.y / a.y);
                minSidonIncrement = fmin(minSidonIncrement, sidonIncrement.x);
                minalphai = cornera_minus_s.y / a.y;
                maxalphai = cornerb_minus_s.y / a.y;
                if(minalphai > maxalphai)
                {
                    tmp = minalphai;
                    minalphai = maxalphai;
                    maxalphai = tmp;
                }
                if(minalphai > minalpha)
                {
                    minalpha = minalphai;
                    alphasPrev.y = minalphai;
                    if(a.x != 0.0)
                    {
                        while(alphasPrev.x + sidonIncrement.x < minalpha)
                        {
                            alphasPrev.x += sidonIncrement.x;
                        }
                    }
                } else
                {
                    alphasPrev.y = minalphai;
                    while(alphasPrev.y + sidonIncrement.y < minalpha)
                    {
                        alphasPrev.y += sidonIncrement.y;
                    }
                }
                maxalpha = fmin(maxalpha, maxalphai);
            }
            if(a.z != 0.0)
            {
                sidonIncrement.z = fabs(voxelSizes.z / a.z);
                minSidonIncrement = fmin(minSidonIncrement, sidonIncrement.z);
                minalphai = cornera_minus_s.z / a.z;
                maxalphai = cornerb_minus_s.z / a.z;
                if(minalphai > maxalphai)
                {
                    tmp = minalphai;
                    minalphai = maxalphai;
                    maxalphai = tmp;
                }
                if(minalphai > minalpha)
                {
                    minalpha = minalphai;
                    alphasPrev.z = minalphai;
                    if(a.x != 0.0)
                    {
                        while(alphasPrev.x + sidonIncrement.x < minalpha)
                        {
                            alphasPrev.x += sidonIncrement.x;
                        }
                    }
                    if(a.y != 0.0)
                    {
                        while(alphasPrev.y + sidonIncrement.y < minalpha)
                        {
                            alphasPrev.y += sidonIncrement.y;
                        }
                    }
                } else
                {
                    alphasPrev.z = minalphai;
                    while(alphasPrev.z + sidonIncrement.z < minalpha)
                    {
                        alphasPrev.z += sidonIncrement.z;
                    }
                }
                minalpha = fmax(minalpha, minalphai);
                maxalpha = fmin(maxalpha, maxalphai);
            }
            double halfMinIncrement = minSidonIncrement * 0.5;
            double3 alphasNext = alphasPrev + sidonIncrement;

            double alphaprev = minalpha;
            double alphanext, LEN, pos;
            int3 ind;
            int IND;
            while(alphaprev + halfMinIncrement < maxalpha)
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
                ind = convert_int3_rtn(sourcePosition + pos * a
                                       - (zerocorner_xyz + (double3)(0.5, 0.5, 0.5)));
                IND = ind.x + ind.y * vdims.x + ind.z * vdims.x * vdims.y;
                VAL += volume[IND] * LEN;
                assert(all(ind >= (int3)(0, 0, 0)) && all(ind < vdims));
                alphaprev = alphanext;
            }
        }
    }
    uint pin = px + pdims.x * py;
    projection[projectionOffset + pin] = VAL / totalProbes;
}
