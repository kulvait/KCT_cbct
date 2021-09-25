//==============================projector.cl=====================================
#ifndef zeroPrecisionTolerance
#define zeroPrecisionTolerance 1e-10
#endif
/**
 * We parametrize the line segment from A to B by parameter t such that t=0 for A and t=1 for B.
 * Then we will find the t corresponding to the point v = t*B+(1-t)A  that maps to the coordinate PX
 * on the detector. We assume that the mapping is linear and that A maps to PX_A and B maps to PX_B.
 * If A and B maps to the same PX, t=MAXFLOAT
 *
 * @param PX
 * @param PX_A PX index related to A
 * @param PX_B PX index related to B
 *
 * @return Parametrization of the line that maps to PX.
 */
inline double intersectionXTime(double PX, double PX_A, double PX_B)
{
    if(PX_A == PX_B)
    {
        return DBL_MAX;
    } else
    {
        return (PX - PX_A) / (PX_B - PX_A);
    }
}

/**
 * Let v0,v1,v2,v3 and v0,v3,v2,v1 be a piecewise lines that maps on detector on values PX_xyx0,
 * PX_xyx1, PX_xyx2, PX_xyx3. Find a two parametrization factors that maps to a PX on these
 * piecewise lines. We expect that the mappings are linear and nondecreasing up to the certain point
 * from both sides. Parametrization is returned in nextIntersections variable first from
 * v0,v1,v2,v3,v0 lines and next from v0,v3,v2,v1,v0. It expects that PX is between min(*PX_xyx0,
 * *PX_xyx1, *PX_xyx2, *PX_xyx3) and max(*PX_xyx0, *PX_xyx1, *PX_xyx2, *PX_xyx3).
 *
 * @param CM Projection camera matrix.
 * @param PX Mapping of projector
 * @param v0 Point v0
 * @param v1 Point v1
 * @param v2 Point v2
 * @param v3 Point v3
 * @param PX_xyx0 Mapping of v0
 * @param PX_xyx1 Mapping of v1
 * @param PX_xyx2 Mapping of v2
 * @param PX_xyx3 Mapping of v3
 * @param nextIntersections Output tuple of parametrizations that maps to PX.
 * @param v_xyx Output first coordinate that maps to PX on the parametrized piecewise line
 * v0,v1,v2,v3.
 * @param v_cw Output first coordinate that maps to PX on the parametrized piecewise line
 * v0,v3,v2,v1.
 */
inline double exactIntersectionPoints(const double PX,
                                      const double3* v0,
                                      const double3* v1,
                                      const double3* v2,
                                      const double3* v3,
                                      const double* PX_xyx0,
                                      const double* PX_xyx1,
                                      const double* PX_xyx2,
                                      const double* PX_xyx3,
                                      const double16 CM,
                                      double3* centroid)
{
    double p, q, tmp, totalweight;
    double3 v_cw, v_xyx;
    const double3 Fvector = CM.s012 - PX * CM.s89a;
    const double Fconstant = CM.s3 - PX * CM.sb;
    double FproductA, FproductB;
    if(PX < (*PX_xyx1))
    {
        FproductA = dot(*v0, Fvector);
        FproductB = dot(*v1, Fvector);
        p = (FproductA + Fconstant) / (FproductA - FproductB);
        v_xyx = (*v0) * (1.0 - p) + (*v1) * p;
        if(PX < (*PX_xyx3))
        {
            q = (FproductA + Fconstant) / (FproductA - dot(*v3, Fvector));
            v_cw = (*v0) * (1.0 - q) + (*v3) * q;
            (*centroid) = (v_xyx + v_cw + (*v0)) / 3.0;
            return p * q * 0.5;
        } else if(PX < (*PX_xyx2))
        {
            q = (dot(*v3, Fvector) + Fconstant) / (dot(*v3, Fvector) - dot(*v2, Fvector));
            v_cw = (*v3) * (1.0 - q) + (*v2) * q;
            if(p + q != 0.0) // Due to rounding errors equality might happen producing nan
            {
                (*centroid) = ((*v0) + v_cw + (q / (p + q)) * (*v3) + (p / (p + q)) * v_xyx) / 3.0;
                return (p + q) * 0.5;
            } else
            {
                (*centroid) = (*v0);
                return 0.0;
            }
        } else
        {
            q = (dot(*v2, Fvector) + Fconstant) / (dot(*v2, Fvector) - FproductB);
            v_cw = (*v2) * (1.0 - q) + (*v1) * q;
            tmp = (1.0 - p) * (1.0 - q) * 0.5;
            totalweight = 1 - tmp;
            (*centroid) = (((*v0) + (*v2)) / 2 - tmp * (v_xyx + v_cw + (*v1)) / 3.0) / totalweight;
            return totalweight;
        }
    } else if(PX < (*PX_xyx2))
    {
        FproductA = dot(*v1, Fvector);
        FproductB = dot(*v2, Fvector);
        p = (FproductA + Fconstant) / (FproductA - FproductB);
        v_xyx = (*v1) * (1.0 - p) + (*v2) * p;
        if(PX < (*PX_xyx3))
        {
            q = (dot(*v0, Fvector) + Fconstant) / (dot(*v0, Fvector) - dot(*v3, Fvector));
            v_cw = (*v0) * (1.0 - q) + (*v3) * q;
            if(p + q != 0.0) // Due to rounding errors equality might happen producing nan
            {
                (*centroid) = ((*v1) + v_cw + (q / (p + q)) * (*v0) + (p / (p + q)) * v_xyx) / 3.0;
                return (p + q) * 0.5;
            } else
            {
                (*centroid) = (*v0);
                return 0.0;
            }
        } else
        {
            q = (dot(*v3, Fvector) + Fconstant) / (dot(*v3, Fvector) - FproductB);
            v_cw = (*v3) * (1.0 - q) + (*v2) * q;
            tmp = (1.0 - p) * (1.0 - q) * 0.5;
            totalweight = 1.0 - tmp;
            (*centroid) = (((*v0) + (*v2)) / 2 - tmp * (v_xyx + v_cw + (*v2)) / 3.0) / totalweight;
            return totalweight;
        }
    } else if(PX >= *PX_xyx3)
    {
        (*centroid) = ((*v0) + (*v2)) / 2;
        return 1.0;

    } else
    {
        FproductA = dot(*v3, Fvector);
        p = (FproductA + Fconstant) / (FproductA - dot(*v2, Fvector));
        v_xyx = (*v3) * (1.0 - p) + (*v2) * p;
        q = (FproductA + Fconstant) / (FproductA - dot(*v0, Fvector));
        v_cw = (*v3) * (1.0 - q) + (*v0) * q;
        tmp = p * q * 0.5;
        totalweight = 1.0 - tmp;
        (*centroid) = (((*v1) + (*v3)) / 2 - tmp * (v_xyx + v_cw + (*v3)) / 3.0) / totalweight;
        return totalweight;
    }
}

inline double exactIntersectionPoints0(const double PX,
                                       const double3* v0,
                                       const double3* v1,
                                       const double3* v2,
                                       const double3* v3,
                                       const double* PX_xyx0,
                                       const double* PX_xyx1,
                                       const double* PX_xyx2,
                                       const double* PX_xyx3,
                                       const double16 CM,
                                       double3* centroid)
{
    const double3 Fvector = CM.s012 - PX * CM.s89a;
    const double3 vd1 = (*v1) - (*v0);
    const double3 vd3 = (*v3) - (*v0);
    double Fproduct, FproductVD;
    double p, q;
    double A, w, wcomplement;
    if(PX < (*PX_xyx1))
    {
        Fproduct = -dot(*v0, Fvector);
        FproductVD = dot(vd1, Fvector); // VD1
        p = Fproduct / FproductVD; // v0+p*(v1-v0)
        if(PX < (*PX_xyx3))
        {
            q = Fproduct / dot(vd3, Fvector);
            (*centroid) = (*v0) + (p / 3.0) * vd1 + (q / 3.0) * vd3;
            return 0.5 * p * q;
        } else if(PX < (*PX_xyx2))
        {
            q = -dot(*v3, Fvector) / FproductVD;
            A = 0.5 * (p + q);
            if(A != 0.0) // Due to rounding errors equality might happen producing nan
            {
                w = p / A;
                //    (*centroid) = (*v0)
                //        + mad(p, mad(-1.0 / 6.0, w, 2.0 / 3.0), mad(-q, w, q) / 3.0) * (vd1)
                //        + mad(-1.0 / 6.0, w, 2.0 / 3.0) * (vd3);
                (*centroid) = (*v0) + (p * (2.0 / 3.0 - w / 6.0) + q * (1 - w) / 3.0) * (vd1)
                    + (2.0 / 3.0 - w / 6.0) * (vd3);
            } else
            {
                (*centroid) = (*v0);
            }
            return A;
        } else
        {
            p = 1.0 - p;
            q = -dot(*v1, Fvector) / dot(vd3, Fvector);
            A = 1.0 - 0.5 * p * q;
            w = 1.0 / A;
            //(*centroid) = (*v0) - mad(0.5, w, mad(p, -w, p) / 3.0) * vd1
            //    + mad(0.5, w, mad(q, -w, q) / 3.0) * vd3;
            (*centroid) = (*v1) - (0.5 * w + (p * (1 - w)) / 3.0) * vd1
                + (0.5 * w + (q * (1 - w)) / 3.0) * vd3;
            return A;
        }
    } else if(PX < (*PX_xyx2))
    {
        Fproduct = dot(*v2, Fvector);
        FproductVD = dot(vd3, Fvector);
        p = Fproduct / FproductVD; // V2 + p * (V1-V2)
        if(PX < (*PX_xyx3))
        {
            p = 1.0 - p; // V1 + p * (V2-V1)
            q = -dot(*v0, Fvector) / FproductVD; // V0 + q (V3-V0)
            A = 0.5 * (p + q);
            if(A != 0.0) // Due to rounding errors equality might happen producing nan
            {
                w = q / A;
                //(*centroid) = (*v0)
                //    + mad(q, mad(-1.0 / 6.0, w, 2.0 / 3.0), mad(-p, w, p) / 3.0) * (vd3)
                //    + mad(-1.0 / 6.0, w, 2.0 / 3.0) * (vd1);
                (*centroid) = (*v0) + (q * (2.0 / 3.0 - w / 6.0) + p * (1 - w) / 3.0) * (vd3)
                    + (2.0 / 3.0 - w / 6.0) * (vd1);
            } else
            {
                (*centroid) = (*v0);
            }
            return A;
        } else
        {
            q = Fproduct / dot(vd1, Fvector); // v2+q(v3-v2)
            A = 1.0 - 0.5 * p * q;
            w = 1.0 / A;
            //(*centroid) = (*v2) - mad(0.5, w, mad(q, -w, q) / 3.0) * vd1
            //    - mad(0.5, w, mad(p, -w, p) / 3.0) * vd3;
            (*centroid) = (*v2) - (0.5 * w + (q * (1 - w)) / 3.0) * vd1
                - (0.5 * w + (p * (1 - w)) / 3.0) * vd3;
            return A;
        }
    } else if(PX >= *PX_xyx3)
    {
        (*centroid) = ((*v0) + (*v2)) / 2.0;
        return 1.0;

    } else
    {
        Fproduct = dot(*v3, Fvector);
        p = Fproduct / dot(vd3, Fvector);
        q = -Fproduct / dot(vd1, Fvector);
        A = 1.0 - 0.5 * p * q;
        w = 1.0 / A;
        //(*centroid) = (*v3) + mad(0.5, w, mad(q, -w, q) / 3.0) * vd1
        //    - mad(0.5, w, mad(p, -w, p) / 3.0) * vd3;
        (*centroid)
            = (*v3) + (0.5 * w + (p * (1 - w)) / 3.0) * vd1 - (0.5 * w + (q * (1 - w)) / 3.0) * vd3;
        return A;
    }
}

inline double exactIntersectionPoints0_stable743(const double PX,
                                                 const double3* v0,
                                                 const double3* v1,
                                                 const double3* v2,
                                                 const double3* v3,
                                                 const double* PX_xyx0,
                                                 const double* PX_xyx1,
                                                 const double* PX_xyx2,
                                                 const double* PX_xyx3,
                                                 const double16 CM,
                                                 double3* centroid)
{
    double p, q, tmp, totalweight;
    double3 v_cw, v_xyx, shift;
    const double3 Fvector = CM.s012 - PX * CM.s89a;
    double FproductA, FproductB;
    if(PX < (*PX_xyx1))
    {
        FproductA = dot(*v0, Fvector);
        FproductB = dot(*v1, Fvector);
        p = FproductA / (FproductA - FproductB);
        v_xyx = (*v0) * (1.0 - p) + (*v1) * p;
        if(PX < (*PX_xyx3))
        {
            q = FproductA / (FproductA - dot(*v3, Fvector));
            // v_cw = (*v0) * (1.0 - q) + (*v3) * q;
            (*centroid) = ((3.0 - p - q) * (*v0) + p * (*v1) + q * (*v3)) / 3.0;
            return p * q * 0.5;
        } else if(PX < (*PX_xyx2))
        {
            q = dot(*v3, Fvector) / (dot(*v3, Fvector) - dot(*v2, Fvector));
            v_cw = (*v3) * (1.0 - q) + (*v2) * q;
            if(p + q != 0.0) // Due to rounding errors equality might happen producing nan
            {
                (*centroid) = ((*v0) + v_cw + (q / (p + q)) * (*v3) + (p / (p + q)) * v_xyx) / 3.0;
                return (p + q) * 0.5;
            } else
            {
                (*centroid) = (*v0);
                return 0.0;
            }
        } else
        {
            q = dot(*v2, Fvector) / (dot(*v2, Fvector) - FproductB);
            v_cw = (*v2) * (1.0 - q) + (*v1) * q;
            tmp = (1.0 - p) * (1.0 - q) * 0.5;
            totalweight = 1 - tmp;
            (*centroid) = (((*v0) + (*v2)) / 2 - tmp * (v_xyx + v_cw + (*v1)) / 3.0) / totalweight;
            return totalweight;
        }
    } else if(PX < (*PX_xyx2))
    {
        FproductA = dot(*v1, Fvector);
        FproductB = dot(*v2, Fvector);
        p = FproductA / (FproductA - FproductB);
        v_xyx = (*v1) * (1.0 - p) + (*v2) * p;
        if(PX < (*PX_xyx3))
        {
            q = dot(*v0, Fvector) / (dot(*v0, Fvector) - dot(*v3, Fvector));
            v_cw = (*v0) * (1.0 - q) + (*v3) * q;
            if(p + q != 0.0) // Due to rounding errors equality might happen producing nan
            {
                (*centroid) = ((*v1) + v_cw + (q / (p + q)) * (*v0) + (p / (p + q)) * v_xyx) / 3.0;
                return (p + q) * 0.5;
            } else
            {
                (*centroid) = (*v0);
                return 0.0;
            }
        } else
        {
            q = dot(*v3, Fvector) / (dot(*v3, Fvector) - FproductB);
            v_cw = (*v3) * (1.0 - q) + (*v2) * q;
            tmp = (1.0 - p) * (1.0 - q) * 0.5;
            totalweight = 1.0 - tmp;
            (*centroid) = (((*v0) + (*v2)) / 2 - tmp * (v_xyx + v_cw + (*v2)) / 3.0) / totalweight;
            return totalweight;
        }
    } else if(PX >= *PX_xyx3)
    {
        (*centroid) = ((*v0) + (*v2)) / 2;
        return 1.0;

    } else
    {
        FproductA = dot(*v3, Fvector);
        p = FproductA / (FproductA - dot(*v2, Fvector));
        v_xyx = (*v3) * (1.0 - p) + (*v2) * p;
        q = FproductA / (FproductA - dot(*v0, Fvector));
        v_cw = (*v3) * (1.0 - q) + (*v0) * q;
        tmp = p * q * 0.5;
        totalweight = 1.0 - tmp;
        (*centroid) = (((*v1) + (*v3)) / 2 - tmp * (v_xyx + v_cw + (*v3)) / 3.0) / totalweight;
        return totalweight;
    }
}

/**
 * Let v0,v1,v2,v3 and v0,v3,v2,v1 be a piecewise lines that maps on detector on values PX_xyx0,
 * PX_xyx1, PX_xyx2, PX_xyx3. Find a two parametrization factors that maps to a PX on these
 * piecewise lines. We expect that the mappings are linear and nondecreasing up to the certain point
 * from both sides. Parametrization is returned in nextIntersections variable first from
 * v0,v1,v2,v3,v0 lines and next from v0,v3,v2,v1,v0. It expects that PX is between min(*PX_xyx0,
 * *PX_xyx1, *PX_xyx2, *PX_xyx3) and max(*PX_xyx0, *PX_xyx1, *PX_xyx2, *PX_xyx3).
 *
 * @param CM Projection camera matrix.
 * @param PX Mapping of projector
 * @param v0 Point v0
 * @param v1 Point v1
 * @param v2 Point v2
 * @param v3 Point v3
 * @param PX_xyx0 Mapping of v0
 * @param PX_xyx1 Mapping of v1
 * @param PX_xyx2 Mapping of v2
 * @param PX_xyx3 Mapping of v3
 * @param nextIntersections Output tuple of parametrizations that maps to PX.
 * @param v_xyx Output first coordinate that maps to PX on the parametrized piecewise line
 * v0,v1,v2,v3.
 * @param v_cw Output first coordinate that maps to PX on the parametrized piecewise line
 * v0,v3,v2,v1.
 */
inline double findIntersectionPoints(const double PX,
                                     const double3* v0,
                                     const double3* v1,
                                     const double3* v2,
                                     const double3* v3,
                                     const double* PX_xyx0,
                                     const double* PX_xyx1,
                                     const double* PX_xyx2,
                                     const double* PX_xyx3,
                                     double3* centroid)
{
    double p, q, tmp, totalweight;
    double3 v_cw, v_xyx;
    if(PX <= (*PX_xyx1))
    {
        p = intersectionXTime(PX, *PX_xyx0, *PX_xyx1);
        v_xyx = (*v0) * (1.0 - p) + (*v1) * p;
        if(PX <= (*PX_xyx3))
        {
            q = intersectionXTime(PX, *PX_xyx0, *PX_xyx3);
            v_cw = (*v0) * (1.0 - q) + (*v3) * q;
            (*centroid) = (v_xyx + v_cw + (*v0)) / 3.0;
            return p * q * 0.5;
        } else if(PX <= (*PX_xyx2))
        {
            q = intersectionXTime(PX, *PX_xyx3, *PX_xyx2);
            v_cw = (*v3) * (1.0 - q) + (*v2) * q;
            (*centroid) = ((*v0) + v_cw + (q / (p + q)) * (*v3) + (p / (p + q)) * v_xyx) / 3.0;
            return (p + q) * 0.5;
        } else
        {
            q = intersectionXTime(PX, *PX_xyx2, *PX_xyx1);
            v_cw = (*v2) * (1.0 - q) + (*v1) * q;
            tmp = (1.0 - p) * (1.0 - q) * 0.5;
            totalweight = 1 - tmp;
            (*centroid) = (((*v0) + (*v2)) / 2 - tmp * (v_xyx + v_cw + (*v1)) / 3.0) / totalweight;
            return totalweight;
        }
    } else if(PX <= (*PX_xyx2))
    {
        p = intersectionXTime(PX, *PX_xyx1, *PX_xyx2);
        v_xyx = (*v1) * (1.0 - p) + (*v2) * p;
        if(PX <= (*PX_xyx3))
        {
            q = intersectionXTime(PX, *PX_xyx0, *PX_xyx3);
            v_cw = (*v0) * (1.0 - q) + (*v3) * q;
            (*centroid) = ((*v1) + v_cw + (q / (p + q)) * (*v0) + (p / (p + q)) * v_xyx) / 3.0;
            return (p + q) * 0.5;
        } else
        {
            q = intersectionXTime(PX, *PX_xyx3, *PX_xyx2);
            v_cw = (*v3) * (1.0 - q) + (*v2) * q;
            tmp = (1.0 - p) * (1.0 - q) * 0.5;
            totalweight = 1.0 - tmp;
            (*centroid) = (((*v0) + (*v2)) / 2 - tmp * (v_xyx + v_cw + (*v2)) / 3.0) / totalweight;
            return totalweight;
        }
    } else
    {
        p = intersectionXTime(PX, *PX_xyx2, *PX_xyx3);
        v_xyx = (*v2) * (1.0 - p) + (*v3) * p;
        q = intersectionXTime(PX, *PX_xyx0, *PX_xyx3);
        v_cw = (*v0) * (1.0 - q) + (*v3) * q;
        tmp = (1.0 - p) * (1.0 - q) * 0.5;
        totalweight = 1.0 - tmp;
        (*centroid) = (((*v1) + (*v3)) / 2 - tmp * (v_xyx + v_cw + (*v3)) / 3.0) / totalweight;
        return totalweight;
    }
}

/**
 * Compute point parametrized by p on piecewise line segments V_xyx0 ... V_xyx1 ... V_xyx2 ...
 * V_xyx3
 *
 * @param p
 * @param V_xyx0
 * @param V_xyx1
 * @param V_xyx2
 * @param V_xyx3
 */
double3
intersectionPoint(double p, double3* V_xyx0, double3* V_xyx1, double3* V_xyx2, double3* V_xyx3)
{
    double3 v;
    if(p <= 1.0)
    {
        v = (*V_xyx0) * (1.0 - p) + p * (*V_xyx1);
    } else if(p <= 2.0)
    {
        p -= 1.0;
        v = (*V_xyx1) * (1.0 - p) + p * (*V_xyx2);
    } else if(p <= 3.0)
    {
        p -= 2.0;
        v = (*V_xyx2) * (1.0 - p) + p * (*V_xyx3);
    }
    return v;
}

/**
 * From parameters dox.x and dox.y compute the size of the square that is parametrized by these
 * parameters starting from single vertex.
 *
 * @param abc
 *
 * @return
 */
double computeSquareSize(double2 abc)
{
    if(abc.x > abc.y)
    {
        double tmp = abc.x;
        abc.x = abc.y;
        abc.y = tmp;
    }
    // abc.x<=abc.y
    if(abc.x <= 1.0)
    {
        if(abc.y <= 1.0)
        {
            return abc.x * abc.y / 2.0; // Triangle that is bounded by corners (0, a, b)
        } else if(abc.y <= 2.0)
        {
            abc.y = abc.y - 1.0; // Upper edge length
            return abc.x + (abc.y - abc.x) / 2.0;
        } else if(abc.y <= 3.0)
        {
            abc.x = 1.0 - abc.x;
            abc.y = 3.0 - abc.y;
            return 1.0 - (abc.x * abc.y) / 2.0; // Whole square but the area of
                                                // the triangle bounded by // corners (a, 1, b)
        }
    } else if(abc.x <= 2.0)
    {
        // abc.y<=2
        abc.x = 2.0 - abc.x;
        abc.y = 2.0 - abc.y;
        return 1.0 - (abc.x * abc.y) / 2.0; // Whole square but the area of the
                                            // triangle bounded by (a, 2, b)
    }
    return 0; // This is not gonna happen
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
 */
void kernel FLOATcutting_voxel_project(global const float* restrict volume,
                                       global float* restrict projection,
                                       private uint projectionOffset,
                                       private double16 _CM,
                                       private double3 _sourcePosition,
                                       private double3 _normalToDetector,
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
    const float16 CM = convert_float16(_CM);
    const float3 sourcePosition = convert_float3(_sourcePosition);
    const float3 voxelSizes = convert_float3(_voxelSizes);
    const float3 volumeCenter = convert_float3(_volumeCenter);
#else
#define CM _CM
#define sourcePosition _sourcePosition
#define voxelSizes _voxelSizes
#define volumeCenter _volumeCenter
#endif
    const REAL3 halfVoxelSizes = HALF * voxelSizes;
    const REAL3 volumeCenter_voxelcenter_offset
        = (REAL3)(2 * i + 1 - vdims.x, 2 * j + 1 - vdims.y, 2 * k + 1 - vdims.z) * halfVoxelSizes;
    const REAL3 voxelcenter_xyz = volumeCenter + volumeCenter_voxelcenter_offset - sourcePosition;
#ifdef ELEVATIONCORRECTION
    const REAL tgelevation = fabs(voxelcenter_xyz.z)
        / sqrt(voxelcenter_xyz.x * voxelcenter_xyz.x + voxelcenter_xyz.y * voxelcenter_xyz.y);
#endif

    const uint IND = voxelIndex(i, j, k, vdims);
    const float voxelValue = volume[IND];
    if(voxelValue == 0.0f)
    {
        return;
    }
#ifdef DROPCENTEROFFPROJECTORVOXELS
    int xindex = INDEX(PROJECTX0(CM, voxelcenter_xyz));
    int yindex = INDEX(PROJECTY0(CM, voxelcenter_xyz));
    if(xindex < 0 || yindex < 0 || xindex >= pdims.x || yindex >= pdims.y)
    {
        return;
    }
#endif

#ifdef DROPINCOMPLETEVOXELS // Here I need to do more
    int xindex = INDEX(PROJECTX0(CM, voxelcenter_xyz));
    int yindex = INDEX(PROJECTY0(CM, voxelcenter_xyz));
    if(xindex < 0 || yindex < 0 || xindex >= pdims.x || yindex >= pdims.y)
    {
        return;
    }
#endif
    const REAL voxelVolume = voxelSizes.x * voxelSizes.y * voxelSizes.z;
    REAL sourceToVoxel_xyz_norm2 = dot(voxelcenter_xyz, voxelcenter_xyz);
#ifdef RELAXED
    float value = (voxelValue * voxelVolume * scalingFactor / sourceToVoxel_xyz_norm2);
#else
    float value = (float)(voxelValue * voxelVolume * scalingFactor / sourceToVoxel_xyz_norm2);
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
    REAL px00, px10, px01, px11;
    REAL3 vx00, vx10, vx01, vx11; // Last is the voxel, where minimum PX is reached
    vx00 = voxelcenter_xyz + halfVoxelSizes * (REAL3)(-ONE, -ONE, ZERO);
    vx10 = voxelcenter_xyz + halfVoxelSizes * (REAL3)(ONE, -ONE, ZERO);
    vx01 = voxelcenter_xyz + halfVoxelSizes * (REAL3)(-ONE, ONE, ZERO);
    vx11 = voxelcenter_xyz + halfVoxelSizes * (REAL3)(ONE, ONE, ZERO);
    {
        REAL nx = dot(voxelcenter_xyz, CM.s012);
        REAL dv = dot(voxelcenter_xyz, CM.s89a);
        REAL nhx = halfVoxelSizes.x * CM.s0;
        REAL nhy = halfVoxelSizes.y * CM.s1;
        REAL dhx = halfVoxelSizes.x * CM.s8;
        REAL dhy = halfVoxelSizes.y * CM.s9;
        px00 = (nx - nhx - nhy) / (dv - dhx - dhy);
        px01 = (nx - nhx + nhy) / (dv - dhx + dhy);
        px10 = (nx + nhx - nhy) / (dv + dhx - dhy);
        px11 = (nx + nhx + nhy) / (dv + dhx + dhy);
    }
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
    if(px00 < px10)
    {
        if(px00 < px01)
        {
            pxx_min = px00;
            V0 = &vx00;
            vd1 = voxelSizes.x;
            vd3 = voxelSizes.y;
            PX_xyx[0] = &px00;
            PX_xyx[1] = &px10;
            PX_xyx[2] = &px11;
            PX_xyx[3] = &px01;
            if(px01 > px11)
            {
                pxx_max = px01;
            } else if(px10 > px11)
            {
                pxx_max = px10;
            } else
            {
                pxx_max = px11;
            }
        } else if(px01 < px11)
        {
            pxx_min = px01;
            V0 = &vx01;
            vd1 = voxelSizes.x;
            vd3 = -voxelSizes.y;
            PX_xyx[0] = &px01;
            PX_xyx[1] = &px11;
            PX_xyx[2] = &px10;
            PX_xyx[3] = &px00;
            if(px10 > px11)
            {
                pxx_max = px10;
            } else
            {
                pxx_max = px11;
            }
        } else
        {
            pxx_min = px11;
            pxx_max = px10;
            V0 = &vx11;
            vd1 = -voxelSizes.x;
            vd3 = -voxelSizes.y;
            PX_xyx[0] = &px11;
            PX_xyx[1] = &px01;
            PX_xyx[2] = &px00;
            PX_xyx[3] = &px10;
        }

    } else if(px10 < px11)
    {
        pxx_min = px10;
        V0 = &vx10;
        vd1 = -voxelSizes.x;
        vd3 = voxelSizes.y;
        PX_xyx[0] = &px10;
        PX_xyx[1] = &px00;
        PX_xyx[2] = &px01;
        PX_xyx[3] = &px11;
        if(px00 > px01)
        {
            pxx_max = px00;
        } else if(px11 > px01)
        {
            pxx_max = px11;
        } else
        {
            pxx_max = px01;
        }
    } else if(px11 < px01)
    {
        pxx_min = px11;
        V0 = &vx11;
        vd1 = -voxelSizes.x;
        vd3 = -voxelSizes.y;
        PX_xyx[0] = &px11;
        PX_xyx[1] = &px01;
        PX_xyx[2] = &px00;
        PX_xyx[3] = &px10;
        if(px00 > px01)
        {
            pxx_max = px00;
        } else
        {
            pxx_max = px01;
        }
    } else
    {
        pxx_min = px01;
        pxx_max = px00;
        V0 = &vx01;
        vd1 = voxelSizes.x;
        vd3 = -voxelSizes.y;
        PX_xyx[0] = &px01;
        PX_xyx[1] = &px11;
        PX_xyx[2] = &px10;
        PX_xyx[3] = &px00;
    }

    min_PX = convert_int_rtn(pxx_min + zeroPrecisionTolerance + 0.5);
    max_PX = convert_int_rtn(pxx_max - zeroPrecisionTolerance + 0.5);
    if(max_PX >= 0 && min_PX < pdims.x)
    {
        if(max_PX <= min_PX) // These indices are in the admissible range
        {
            min_PX = convert_int_rtn(HALF * (pxx_min + pxx_max) + HALF);
            exactEdgeValues0(projection, CM, (vx00 + vx11) * HALF, min_PX, value, voxelSizes,
                             pdims);
        } else
        {
            REAL lastSectionSize, nextSectionSize, polygonSize;
            REAL3 lastInt, nextInt, Int;
            REAL factor;
#ifdef DROPINCOMPLETEVOXELS
            if(min_PX < 0 || max_PX >= pdims.x)
            {
                return;
            }
#endif
            int I = max(-1, min_PX);
            int I_STOP = min(max_PX, pdims.x);
            // Section of the square that corresponds to the indices < i
            // CCW and CW coordinates of the last intersection on the lines specified by the
            // points in V_xyx
            REAL2 CENTROID, CENTROID_cur, CENTROID_prev;
            REAL llength_next, llength_prev, corlambda;
            lastSectionSize = exactIntersectionPolygons0(((REAL)I) + HALF, vd1, vd3, V0, PX_xyx[0],
                                                         PX_xyx[1], PX_xyx[2], PX_xyx[3], CM,
                                                         voxelSizes, &CENTROID_prev, &llength_prev);
            if(I >= 0)
            {
                factor = value * lastSectionSize;
                Int = (REAL3)(CENTROID_prev, vx00.z);
#ifdef ELEVATIONCORRECTION
                corlambda = QUARTER * llength_prev * tgelevation / voxelSizes.z;
                // HALF*tgelevation works reasonable
                exactEdgeValues0ElevationCorrection(projection, CM, Int, I, factor, voxelSizes,
                                                    pdims, corlambda);
#else
                exactEdgeValues0(projection, CM, Int, I, factor, voxelSizes, pdims);
#endif
            }
            for(I = I + 1; I < I_STOP; I++)
            {
                nextSectionSize = exactIntersectionPolygons0(
                    ((REAL)I) + HALF, vd1, vd3, V0, PX_xyx[0], PX_xyx[1], PX_xyx[2], PX_xyx[3], CM,
                    voxelSizes, &CENTROID_cur, &llength_next);
                polygonSize = nextSectionSize - lastSectionSize;
                CENTROID = (nextSectionSize * CENTROID_cur - lastSectionSize * CENTROID_prev)
                    / polygonSize;
                Int = (REAL3)(CENTROID, vx00.z);
                CENTROID_prev = CENTROID_cur;
                factor = value * polygonSize;
#ifdef ELEVATIONCORRECTION
                corlambda = QUARTER * (llength_next + llength_prev) * tgelevation / voxelSizes.z;
                llength_prev = llength_next;
                exactEdgeValues0ElevationCorrection(projection, CM, Int, I, factor, voxelSizes,
                                                    pdims, corlambda);
#else
                exactEdgeValues0(projection, CM, Int, I, factor, voxelSizes, pdims);
#endif
                lastSectionSize = nextSectionSize;
                lastInt = nextInt;
            }
            polygonSize = ONE - lastSectionSize;
            if(I_STOP < pdims.x && polygonSize > zeroPrecisionTolerance) // If polygonsize==0 it might trigger division by zero
            {
                CENTROID_cur = V0->s01 + (REAL2)(HALF * vd1, HALF * vd3);
                CENTROID = (CENTROID_cur - lastSectionSize * CENTROID_prev) / polygonSize;
                Int = (REAL3)(CENTROID, vx00.z);
                factor = value * polygonSize;
#ifdef ELEVATIONCORRECTION
                corlambda = QUARTER * llength_next * tgelevation / voxelSizes.z;
                exactEdgeValues0ElevationCorrection(projection, CM, Int, I, factor, voxelSizes,
                                                    pdims, corlambda);
#else
                exactEdgeValues0(projection, CM, Int, I, factor, voxelSizes, pdims);
#endif
            }
        }
    }
}
//==============================END projector.cl=====================================
