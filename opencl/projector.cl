/** Projection of a volume point v onto X coordinate on projector.
 * No checks for boundaries.
 *
 * @param CM Projection camera matrix
 * @param v Volume point
 * @param PX_out Output
 */
inline double projectX(private const double16 CM, private const double3 v)
{
    return (dot(v, CM.s012) + CM.s3) / (dot(v, CM.s89a) + CM.sb);
}

/** Projection of a volume point v onto Y coordinate on projector.
 * No checks for boundaries.
 *
 * @param CM Projection camera matrix
 * @param v Volume point
 * @param PY_out Output
 */
inline double projectY(private const double16 CM, private const double3 v)
{
    return (dot(v, CM.s456) + CM.s7) / (dot(v, CM.s89a) + CM.sb);
}

/** Projection of a volume point v onto P coordinate on projector.
 * No checks for boundaries.
 *
 * @param CM Projection camera matrix
 * @param v Volume point
 * @param P_out Output
 */
inline void project(private const double16* CM, private const double3* v, private double2* P_out)
{

    double3 coord;
    coord.x = dot(*v, CM->s012) + CM->s3;
    coord.y = dot(*v, CM->s456) + CM->s7;
    coord.z = dot(*v, CM->s89a) + CM->sb;
    P_out->x = coord.x / coord.z;
    P_out->y = coord.y / coord.z;
}

int2 projectionIndices(private double16 CM, private double3 v, int2 pdims)
{
    double3 coord;
    coord.x = dot(v, CM.s012);
    coord.y = dot(v, CM.s456);
    coord.z = dot(v, CM.s89a);
    coord += CM.s37b;
    coord.x /= coord.z;
    coord.y /= coord.z;
    int2 ind;
    ind.x = convert_int_rtn(coord.x + 0.5);
    ind.y = convert_int_rtn(coord.y + 0.5);
    if(ind.x >= 0 && ind.y >= 0 && ind.x < pdims.x && ind.y < pdims.y)
    {
        return ind;
    } else
    {
        return pdims;
    }
}

int projectionIndex(private double16 CM, private double3 v, int2 pdims)
{
    double3 coord;
    coord.x = dot(v, CM.s012);
    coord.y = dot(v, CM.s456);
    coord.z = dot(v, CM.s89a);
    coord += CM.s37b;
    coord.x /= coord.z;
    coord.y /= coord.z;
    int2 ind;
    ind.x = convert_int_rtn(coord.x + 0.5);
    ind.y = convert_int_rtn(coord.y + 0.5);
    if(ind.x >= 0 && ind.y >= 0 && ind.x < pdims.x && ind.y < pdims.y)
    {
        return ind.x + pdims.x * ind.y;
    } else
    {
        return -1;
    }
}

/// insertEdgeValues(factor, V, P, projection, pdims);
void inline exactEdgeValues(global float* projection,
                            private double16 CM,
                            private double3 v,
                            private int PX,
                            private double value,
                            private double3 voxelSizes,
                            private int2 pdims)
{
    const double3 distanceToEdge = (double3)(0.0, 0.0, 0.5 * voxelSizes[2]);
    const double3 v_up = v + distanceToEdge;
    const double3 v_down = v - distanceToEdge;
    // const double3 v_diff = v_down - v_up;
    const double negativeEdgeLength = -voxelSizes[2];
    const double PY_up = projectY(CM, v_up);
    const double PY_down = projectY(CM, v_down);
    const int PJ_up = convert_int_rtn(PY_up + 0.5);
    const int PJ_down = convert_int_rtn(PY_down + 0.5);
    double lambda;
    double lastLambda = 0.0;
    double leastLambda;
    double3 Fvector;
    int PJ_max;
    if(PJ_down < PJ_up)
    {
        if(PJ_up >= 0 && PJ_down < pdims.y)
        {
            int J;
            if(PJ_down < 0)
            {
                J = 0;
                Fvector = CM.s456 + 0.5 * CM.s89a;
                // lastLambda = (dot(v_down, Fvector) + CM.s7 + 0.5 * CM.sb) / (dot(v_diff,
                // Fvector));
                lastLambda = (dot(v_down, Fvector) + CM.s7 + 0.5 * CM.sb)
                    / (negativeEdgeLength * Fvector[2]);
            } else
            {
                J = PJ_down;
                Fvector = CM.s456 - (J - 0.5) * CM.s89a;
            }
            if(PJ_up >= pdims.y)
            {
                PJ_max = pdims.y - 1;
                double3 Qvector = CM.s456 - (PJ_max + 0.5) * CM.s89a;
                // leastLambda = (dot(v_down, Qvector) + CM.s7 - ((double)PJ_max + 0.5) * CM.sb)
                //    / (dot(v_diff, Qvector));
                leastLambda = (dot(v_down, Qvector) + CM.s7 - ((double)PJ_max + 0.5) * CM.sb)
                    / (negativeEdgeLength * Qvector[2]);
            } else
            {
                PJ_max = PJ_up;
                leastLambda = 1.0;
            }
            for(; J < PJ_max; J++)
            {
                Fvector -= CM.s89a; // Fvector = CM.s456 - (J + 0.5) * CM.s89a;
                lambda = (dot(v_down, Fvector) + CM.s7 - ((double)J + 0.5) * CM.sb)
                    / (negativeEdgeLength * Fvector[2]);
                AtomicAdd_g_f(&projection[PX + pdims.x * J],
                              (lambda - lastLambda)
                                  * value); // Atomic version of projection[ind] += value;
                lastLambda = lambda;
            }
            // PJ_max
            AtomicAdd_g_f(&projection[PX + pdims.x * PJ_max],
                          (leastLambda - lastLambda)
                              * value); // Atomic version of projection[ind] += value;
        }
    } else if(PJ_down > PJ_up)
    {
        if(PJ_down >= 0 && PJ_up < pdims.y)
        {
            // We will count with negative value of lambda by dividing by dot(v_diff, Fvector)
            // instead of dot(-v_diff, Fvector)  Because valuePerUnit is negative the value (lambda
            // - lastLambda)*valuePerUnit will be positive
            // lambda here measures negative distance from v_up to a given intersection point
            int J;
            if(PJ_up < 0)
            {
                J = 0;
                Fvector = CM.s456 + 0.5 * CM.s89a;
                lastLambda = (dot(v_up, Fvector) + CM.s7 + 0.5 * CM.sb)
                    / (negativeEdgeLength * Fvector[2]);
            } else
            {
                J = PJ_up;
                Fvector = CM.s456 - (J - 0.5) * CM.s89a;
            }
            if(PJ_down >= pdims.y)
            {
                PJ_max = pdims.y - 1;
                double3 Qvector = CM.s456 - (PJ_max + 0.5) * CM.s89a;
                leastLambda = (dot(v_up, Qvector) + CM.s7 - ((double)PJ_max + 0.5) * CM.sb)
                    / (negativeEdgeLength * Qvector[2]);
            } else
            {
                PJ_max = PJ_down;
                leastLambda = -1.0;
            }
            for(; J < PJ_max; J++)
            {
                Fvector -= CM.s89a; // Fvector = CM.s456 - (J + 0.5) * CM.s89a;
                lambda = (dot(v_up, Fvector) + CM.s7 - ((double)J + 0.5) * CM.sb)
                    / (negativeEdgeLength * Fvector[2]);
                AtomicAdd_g_f(&projection[PX + pdims.x * J],
                              (lastLambda - lambda)
                                  * value); // Atomic version of projection[ind] += value;
                lastLambda = lambda;
            }
            // PJ_max
            AtomicAdd_g_f(&projection[PX + pdims.x * PJ_max],
                          (lastLambda - leastLambda)
                              * value); // Atomic version of projection[ind] += value;
        }
    } else if(PJ_down == PJ_up && PJ_down >= 0 && PJ_down < pdims.y)
    {
        AtomicAdd_g_f(&projection[PX + pdims.x * PJ_down],
                      value); // Atomic version of projection[ind] += value;
    }
}

/// insertEdgeValues(factor, V, P, projection, pdims);
void inline insertEdgeValues(global float* projection,
                             private double16 CM,
                             private double3 v,
                             private int PX,
                             private double value,
                             private double3 voxelSizes,
                             private int2 pdims)
{
    double3 v_down, v_up;
    double PY_down, PY_up;
    int PJ_down, PJ_up;
    v_down = v + voxelSizes * (double3)(0.0, 0.0, -0.5);
    v_up = v + voxelSizes * (double3)(0.0, 0.0, 0.5);
    PY_down = projectY(CM, v_down);
    PY_up = projectY(CM, v_up);
    PJ_down = convert_int_rtn(PY_down + 0.5);
    PJ_up = convert_int_rtn(PY_up + 0.5);
    if(PJ_down > PJ_up)
    {
        int tmp_i;
        double tmp_d;
        tmp_i = PJ_down;
        PJ_down = PJ_up;
        PJ_up = tmp_i;
        tmp_d = PY_down;
        PY_down = PY_up;
        PY_up = tmp_d;
    }
    if(PJ_up < 0 || PJ_down >= pdims.y)
    {
        return;
    }
    if(PJ_down == PJ_up)
    {
        AtomicAdd_g_f(&projection[PX + pdims.x * PJ_down],
                      value); // Atomic version of projection[ind] += value;
        return;
    }
    double stepSize = value
        / (PY_up
           - PY_down); // Length of z in volume to increase y in projection by 1 multiplied by value
    // int j = max(-1, PJ_down);
    // int j_STOP = min(PJ_up, pdims.y);
    int j, j_STOP;
    // Add part that maps to PJ_down
    if(PJ_down >= 0)
    {
        // double nextGridY;
        // nextGridY = (double)PJ_down + 0.5;
        // factor = (nextGridY - PY_down) * stepSize * value;
        AtomicAdd_g_f(&projection[PX + pdims.x * PJ_down],
                      ((double)PJ_down + 0.5 - PY_down)
                          * stepSize); // Atomic version of projection[ind] += value;
        j = PJ_down + 1;
    } else
    {
        j = 0;
    }
    // Add part that maps to PJ_up
    if(PJ_up < pdims.y)
    {
        // double prevGridY;
        // prevGridY = (double)PJ_up - 0.5;
        // factor = (PY_up - prevGridY) * stepSize * value;
        AtomicAdd_g_f(&projection[PX + pdims.x * PJ_up],
                      (PY_up - ((double)PJ_up - 0.5))
                          * stepSize); // Atomic version of projection[ind] += value;
        j_STOP = PJ_up;
    } else
    {
        j_STOP = pdims.y;
    }
    for(; j < j_STOP; j++)
    {
        AtomicAdd_g_f(&projection[PX + pdims.x * j],
                      stepSize); // Atomic version of projection[ind] += value;
    }
}

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
 * Let v0,v1,v2,v3 and v0,v3,v2,v1 be a piecewise lines that maps on detector on values PX_ccw0,
 * PX_ccw1, PX_ccw2, PX_ccw3. Find a two parametrization factors that maps to a PX on these
 * piecewise lines. We expect that the mappings are linear and nondecreasing up to the certain point
 * from both sides. Parametrization is returned in nextIntersections variable first from
 * v0,v1,v2,v3,v0 lines and next from v0,v3,v2,v1,v0. It expects that PX is between min(*PX_ccw0,
 * *PX_ccw1, *PX_ccw2, *PX_ccw3) and max(*PX_ccw0, *PX_ccw1, *PX_ccw2, *PX_ccw3).
 *
 * @param CM Projection camera matrix.
 * @param PX Mapping of projector
 * @param v0 Point v0
 * @param v1 Point v1
 * @param v2 Point v2
 * @param v3 Point v3
 * @param PX_ccw0 Mapping of v0
 * @param PX_ccw1 Mapping of v1
 * @param PX_ccw2 Mapping of v2
 * @param PX_ccw3 Mapping of v3
 * @param nextIntersections Output tuple of parametrizations that maps to PX.
 * @param v_ccw Output first coordinate that maps to PX on the parametrized piecewise line
 * v0,v1,v2,v3.
 * @param v_cw Output first coordinate that maps to PX on the parametrized piecewise line
 * v0,v3,v2,v1.
 */
inline double exactIntersectionPoints(const double PX,
                                      const double3* v0,
                                      const double3* v1,
                                      const double3* v2,
                                      const double3* v3,
                                      const double* PX_ccw0,
                                      const double* PX_ccw1,
                                      const double* PX_ccw2,
                                      const double* PX_ccw3,
                                      const double16 CM,
                                      double3* centroid)
{
    double p, q, tmp, totalweight;
    double3 v_cw, v_ccw, shift;
    const double3 Fvector = CM.s012 - PX * CM.s89a;
    const double Fconstant = CM.s3 - PX * CM.sb;
    double FproductA, FproductB;
    if(PX < (*PX_ccw1))
    {
        FproductA = dot(*v0, Fvector);
        FproductB = dot(*v1, Fvector);
        p = (FproductA + Fconstant) / (FproductA - FproductB);
        v_ccw = (*v0) * (1.0 - p) + (*v1) * p;
        if(PX < (*PX_ccw3))
        {
            q = (FproductA + Fconstant) / (FproductA - dot(*v3, Fvector));
            v_cw = (*v0) * (1.0 - q) + (*v3) * q;
            (*centroid) = (v_ccw + v_cw + (*v0)) / 3.0;
            return p * q * 0.5;
        } else if(PX < (*PX_ccw2))
        {
            q = (dot(*v3, Fvector) + Fconstant) / (dot(*v3, Fvector) - dot(*v2, Fvector));
            v_cw = (*v3) * (1.0 - q) + (*v2) * q;
            if(p + q != 0.0) // Due to rounding errors equality might happen producing nan
            {
                (*centroid) = ((*v0) + v_cw + (q / (p + q)) * (*v3) + (p / (p + q)) * v_ccw) / 3.0;
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
            (*centroid) = (((*v0) + (*v2)) / 2 - tmp * (v_ccw + v_cw + (*v1)) / 3.0) / totalweight;
            return totalweight;
        }
    } else if(PX < (*PX_ccw2))
    {
        FproductA = dot(*v1, Fvector);
        FproductB = dot(*v2, Fvector);
        p = (FproductA + Fconstant) / (FproductA - FproductB);
        v_ccw = (*v1) * (1.0 - p) + (*v2) * p;
        if(PX < (*PX_ccw3))
        {
            q = (dot(*v0, Fvector) + Fconstant) / (dot(*v0, Fvector) - dot(*v3, Fvector));
            v_cw = (*v0) * (1.0 - q) + (*v3) * q;
            if(p + q != 0.0) // Due to rounding errors equality might happen producing nan
            {
                (*centroid) = ((*v1) + v_cw + (q / (p + q)) * (*v0) + (p / (p + q)) * v_ccw) / 3.0;
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
            (*centroid) = (((*v0) + (*v2)) / 2 - tmp * (v_ccw + v_cw + (*v2)) / 3.0) / totalweight;
            return totalweight;
        }
    } else if(PX >= *PX_ccw3)
    {
        (*centroid) = ((*v0) + (*v2)) / 2;
        return 1.0;

    } else
    {
        FproductA = dot(*v3, Fvector);
        p = (FproductA + Fconstant) / (FproductA - dot(*v2, Fvector));
        v_ccw = (*v3) * (1.0 - p) + (*v2) * p;
        q = (FproductA + Fconstant) / (FproductA - dot(*v0, Fvector));
        v_cw = (*v3) * (1.0 - q) + (*v0) * q;
        tmp = p * q * 0.5;
        totalweight = 1.0 - tmp;
        (*centroid) = (((*v1) + (*v3)) / 2 - tmp * (v_ccw + v_cw + (*v3)) / 3.0) / totalweight;
        return totalweight;
    }
}

/**
 * Let v0,v1,v2,v3 and v0,v3,v2,v1 be a piecewise lines that maps on detector on values PX_ccw0,
 * PX_ccw1, PX_ccw2, PX_ccw3. Find a two parametrization factors that maps to a PX on these
 * piecewise lines. We expect that the mappings are linear and nondecreasing up to the certain point
 * from both sides. Parametrization is returned in nextIntersections variable first from
 * v0,v1,v2,v3,v0 lines and next from v0,v3,v2,v1,v0. It expects that PX is between min(*PX_ccw0,
 * *PX_ccw1, *PX_ccw2, *PX_ccw3) and max(*PX_ccw0, *PX_ccw1, *PX_ccw2, *PX_ccw3).
 *
 * @param CM Projection camera matrix.
 * @param PX Mapping of projector
 * @param v0 Point v0
 * @param v1 Point v1
 * @param v2 Point v2
 * @param v3 Point v3
 * @param PX_ccw0 Mapping of v0
 * @param PX_ccw1 Mapping of v1
 * @param PX_ccw2 Mapping of v2
 * @param PX_ccw3 Mapping of v3
 * @param nextIntersections Output tuple of parametrizations that maps to PX.
 * @param v_ccw Output first coordinate that maps to PX on the parametrized piecewise line
 * v0,v1,v2,v3.
 * @param v_cw Output first coordinate that maps to PX on the parametrized piecewise line
 * v0,v3,v2,v1.
 */
inline double findIntersectionPoints(const double PX,
                                     const double3* v0,
                                     const double3* v1,
                                     const double3* v2,
                                     const double3* v3,
                                     const double* PX_ccw0,
                                     const double* PX_ccw1,
                                     const double* PX_ccw2,
                                     const double* PX_ccw3,
                                     double3* centroid)
{
    double p, q, tmp, totalweight;
    double3 v_cw, v_ccw, shift;
    if(PX <= (*PX_ccw1))
    {
        p = intersectionXTime(PX, *PX_ccw0, *PX_ccw1);
        v_ccw = (*v0) * (1.0 - p) + (*v1) * p;
        if(PX <= (*PX_ccw3))
        {
            q = intersectionXTime(PX, *PX_ccw0, *PX_ccw3);
            v_cw = (*v0) * (1.0 - q) + (*v3) * q;
            (*centroid) = (v_ccw + v_cw + (*v0)) / 3.0;
            return p * q * 0.5;
        } else if(PX <= (*PX_ccw2))
        {
            q = intersectionXTime(PX, *PX_ccw3, *PX_ccw2);
            v_cw = (*v3) * (1.0 - q) + (*v2) * q;
            (*centroid) = ((*v0) + v_cw + (q / (p + q)) * (*v3) + (p / (p + q)) * v_ccw) / 3.0;
            return (p + q) * 0.5;
        } else
        {
            q = intersectionXTime(PX, *PX_ccw2, *PX_ccw1);
            v_cw = (*v2) * (1.0 - q) + (*v1) * q;
            tmp = (1.0 - p) * (1.0 - q) * 0.5;
            totalweight = 1 - tmp;
            (*centroid) = (((*v0) + (*v2)) / 2 - tmp * (v_ccw + v_cw + (*v1)) / 3.0) / totalweight;
            return totalweight;
        }
    } else if(PX <= (*PX_ccw2))
    {
        p = intersectionXTime(PX, *PX_ccw1, *PX_ccw2);
        v_ccw = (*v1) * (1.0 - p) + (*v2) * p;
        if(PX <= (*PX_ccw3))
        {
            q = intersectionXTime(PX, *PX_ccw0, *PX_ccw3);
            v_cw = (*v0) * (1.0 - q) + (*v3) * q;
            (*centroid) = ((*v1) + v_cw + (q / (p + q)) * (*v0) + (p / (p + q)) * v_ccw) / 3.0;
            return (p + q) * 0.5;
        } else
        {
            q = intersectionXTime(PX, *PX_ccw3, *PX_ccw2);
            v_cw = (*v3) * (1.0 - q) + (*v2) * q;
            tmp = (1.0 - p) * (1.0 - q) * 0.5;
            totalweight = 1.0 - tmp;
            (*centroid) = (((*v0) + (*v2)) / 2 - tmp * (v_ccw + v_cw + (*v2)) / 3.0) / totalweight;
            return totalweight;
        }
    } else
    {
        p = intersectionXTime(PX, *PX_ccw2, *PX_ccw3);
        v_ccw = (*v2) * (1.0 - p) + (*v3) * p;
        q = intersectionXTime(PX, *PX_ccw0, *PX_ccw3);
        v_cw = (*v0) * (1.0 - q) + (*v3) * q;
        tmp = (1.0 - p) * (1.0 - q) * 0.5;
        totalweight = 1.0 - tmp;
        (*centroid) = (((*v1) + (*v3)) / 2 - tmp * (v_ccw + v_cw + (*v3)) / 3.0) / totalweight;
        return totalweight;
    }
}

/**
 * Compute point parametrized by p on piecewise line segments V_ccw0 ... V_ccw1 ... V_ccw2 ...
 * V_ccw3
 *
 * @param p
 * @param V_ccw0
 * @param V_ccw1
 * @param V_ccw2
 * @param V_ccw3
 */
double3
intersectionPoint(double p, double3* V_ccw0, double3* V_ccw1, double3* V_ccw2, double3* V_ccw3)
{
    double3 v;
    if(p <= 1.0)
    {
        v = (*V_ccw0) * (1.0 - p) + p * (*V_ccw1);
    } else if(p <= 2.0)
    {
        p -= 1.0;
        v = (*V_ccw1) * (1.0 - p) + p * (*V_ccw2);
    } else if(p <= 3.0)
    {
        p -= 2.0;
        v = (*V_ccw2) * (1.0 - p) + p * (*V_ccw3);
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

inline uint voxelIndex(uint i, uint j, uint k, int3 vdims)
{
    return i + j * vdims.x + k * vdims.x * vdims.y;
}

/** Kernel to precompute projection indices to spare some redundancy.
 *
 * @param vertexProjectionIndices
 * @param CM
 * @param voxelSizes
 * @param vdims
 *
 * @return
 */
void kernel computeProjectionIndices(global int* vertexProjectionIndices,
                                     private double16 CM,
                                     double3 voxelSizes,
                                     int3 vdims,
                                     int2 pdims)
{
    uint i = get_global_id(2);
    uint j = get_global_id(1);
    uint k = get_global_id(0); // This is more effective from the perspective of atomic colisions
    const double3 IND_ijk = { (double)(i), (double)(j), (double)(k) };
    const double3 zerocorner_xyz = { -0.5 * (double)vdims.x, -0.5 * (double)vdims.y,
                                     -0.5 * (double)vdims.z }; // -convert_double3(vdims) / 2.0;
    vertexProjectionIndices[i + j * (vdims.x + 1) + k * (vdims.x + 1) * (vdims.y + 1)]
        = projectionIndex(CM, zerocorner_xyz + voxelSizes * IND_ijk, pdims);
}

/**
 * Scale projections by dividing by the area of the pixel times cos^3(\theta) and multiplying by f^2
 *
 * @param projection Projection buffer.
 * @param projectionOffset Offset of projection buffer.
 * @param ICM Inverse camera matrix.
 * @param sourcePosition
 * @param normalToDetector
 * @param pdims Dimensions of projection
 * @param scalingFactor f^2/pixelArea
 *
 */
void kernel FLOATrescale_projections_cos(global float* projection,
                                         private uint projectionOffset,
                                         private double16 ICM,
                                         private double3 sourcePosition,
                                         private double3 normalToDetector,
                                         private uint2 pdims,
                                         private float scalingFactor)
{
    uint px = get_global_id(0);
    uint py = get_global_id(1);
    const uint IND = projectionOffset + px + pdims.x * py;
    const double4 P = { (double)px, (double)py, 1.0, 0.0 };
    double4 V; // Point that will be projected to P by CM

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
    double3 n = normalize(V.s012 - sourcePosition);
    double cosine = fabs(-dot(normalToDetector, n));
    double cosPowThree = cosine * cosine * cosine;
    float pixelValue = projection[IND];
    float value = pixelValue * scalingFactor / cosPowThree;
    projection[IND] = value;
}

/**
 * Scaling by dividing by the area S that is the area of the unit splhere that gets projected onto
 * given pixel.
 *
 * @param projection Projection buffer.
 * @param projectionOffset Offset of projection buffer.
 * @param pdims Dimensions of projection
 * @param normalProjection Point on the detector to that normal to the detector is projected and
 * where z axis of the local coordinates crosses projector.
 * @param pixelSizes Sizes of pixels in voxel related coordinate system.
 * @param sourceToDetector Length from the source to detector.
 *
 * @return
 */
void kernel FLOATrescale_projections_exact(global float* projection,
                                           private const uint projectionOffset,
                                           private const uint2 pdims,
                                           private const double2 normalProjection,
                                           private const double2 pixelSizes,
                                           private const double sourceToDetector)
{
    const uint px = get_global_id(0);
    const uint py = get_global_id(1);
    const uint IND = projectionOffset + px + pdims.x * py;
    const double2 pxD = { px, py };
    const double2 v2 = (pxD - normalProjection) * pixelSizes;
    const double3 v = (double3)(v2, sourceToDetector);
    const double3 halfpiX = { 0.5 * pixelSizes.x, 0.0, 0.0 };
    const double3 halfpiY = { 0.0, 0.5 * pixelSizes.y, 0.0 };
    const double3 v_A = v - halfpiX + halfpiY;
    const double3 v_B = v - halfpiX - halfpiY;
    const double3 v_C = v + halfpiX - halfpiY;
    const double3 v_D = v + halfpiX + halfpiY;
    const double3 n_AB = normalize(cross(v_A, v_B));
    const double3 n_BC = normalize(cross(v_B, v_C));
    const double3 n_CD = normalize(cross(v_C, v_D));
    const double3 n_DA = normalize(cross(v_D, v_A));
    const double S = acos(-dot(n_AB, n_BC)) + acos(-dot(n_BC, n_CD)) - acos(dot(n_CD, n_DA))
        - acos(dot(n_DA, n_AB));
    projection[IND] = projection[IND] / S;
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
void kernel FLOATcutting_voxel_project(global float* volume,
                                       global float* projection,
                                       private uint projectionOffset,
                                       private double16 CM,
                                       private double3 sourcePosition,
                                       private double3 normalToDetector,
                                       private int3 vdims,
                                       private double3 voxelSizes,
                                       private int2 pdims,
                                       private float scalingFactor)
{
    /*
        const uint groupSize = 32;
        const size_t mxi = get_global_size(2);
        const size_t mxj = get_global_size(1);
        const size_t mxk = get_global_size(0);
        const uint groupCounti = (mxi + groupSize - 1) / groupSize;
        const uint groupCountj = (mxj + groupSize - 1) / groupSize;
        const uint groupCountk = (mxk + groupSize - 1) / groupSize;
        const size_t ii = get_global_id(2);
        const size_t ij = get_global_id(1);
        const size_t ik = get_global_id(0);
        uint ai, aj, ak, bi, bj, bk;
        ai = ii % groupCounti;
        aj = ij % groupCountj;
        ak = ik % groupCountk;
        bi = ii / groupCounti;
        bj = ij / groupCountj;
        bk = ik / groupCountk;
        uint i = groupSize * ai + bi;
        uint j = groupSize * aj + bj;
        uint k = groupSize * ak + bk;
        if(i >= mxi)
        {
            int off = (mxi - 1) % groupSize;
            int excess = bi - off;
            i = groupSize * (groupCounti - excess) - 1;
        }
        if(j >= mxj)
        {
            int off = (mxj - 1) % groupSize;
            int excess = bj - off;
            j = groupSize * (groupCountj - excess) - 1;
        }
        if(k >= mxk)
        {
            int off = (mxk - 1) % groupSize;
            int excess = bk - off;
            k = groupSize * (groupCountk - excess) - 1;
        }
    */
    uint i = get_global_id(2);
    uint j = get_global_id(1);
    uint k = get_global_id(0); // This is more effective from the perspective of atomic colisions

    const double3 IND_ijk = { (double)(i), (double)(j), (double)(k) };
    const double3 zerocorner_xyz
        = { -0.5 * (double)vdims.x * voxelSizes.x, -0.5 * (double)vdims.y * voxelSizes.y,
            -0.5 * (double)vdims.z * voxelSizes.z }; // -convert_double3(vdims) / 2.0;
    const double3 voxelcorner_xyz = zerocorner_xyz
        + (IND_ijk * voxelSizes); // Using widening and vector multiplication operations
    const uint IND = voxelIndex(i, j, k, vdims);
    const float voxelVolume = voxelSizes.x * voxelSizes.y * voxelSizes.z;
    const float voxelValue = volume[IND];
    const double3 voxelcenter_xyz
        = voxelcorner_xyz + voxelSizes * 0.5; // Using widening and vector multiplication operations
    if(voxelValue != 0.0)
    {
        // EXPERIMENTAL ... reconstruct inner circle
        /*   const double3 pixcoords = zerocorner_xyz + voxelSizes * (IND_ijk + (double3)(0.5, 0.5,
           0.5)); if(sqrt(pixcoords.x * pixcoords.x + pixcoords.y * pixcoords.y) > 110.0)
           {
               return;
           }*/
        // EXPERIMENTAL ... reconstruct inner circle
        // If all the corners of given voxel points to a common coordinate, then compute the value
        // based on the center
        int cornerProjectionIndex = projectionIndex(CM, voxelcorner_xyz, pdims);
        if(cornerProjectionIndex
               == projectionIndex(CM, voxelcorner_xyz + voxelSizes * (double3)(1.0, 1.0, 1.0),
                                  pdims)
           && cornerProjectionIndex
               == projectionIndex(CM, voxelcorner_xyz + voxelSizes * (double3)(1.0, 1.0, 0.0),
                                  pdims)
           && cornerProjectionIndex
               == projectionIndex(CM, voxelcorner_xyz + voxelSizes * (double3)(1.0, 0.0, 1.0),
                                  pdims)
           && cornerProjectionIndex
               == projectionIndex(CM, voxelcorner_xyz + voxelSizes * (double3)(0.0, 1.0, 1.0),
                                  pdims)
           && cornerProjectionIndex
               == projectionIndex(CM, voxelcorner_xyz + voxelSizes * (double3)(1.0, 0.0, 0.0),
                                  pdims)
           && cornerProjectionIndex
               == projectionIndex(CM, voxelcorner_xyz + voxelSizes * (double3)(0.0, 1.0, 0.0),
                                  pdims)
           && cornerProjectionIndex
               == projectionIndex(CM, voxelcorner_xyz + voxelSizes * (double3)(0.0, 0.0, 1.0),
                                  pdims)) // When all projections are the same
        {
            if(cornerProjectionIndex != -1)
            {
                double3 sourceToVoxel_xyz = voxelcenter_xyz - sourcePosition;
                double sourceToVoxel_xyz_norm2 = dot(sourceToVoxel_xyz, sourceToVoxel_xyz);
                float value = voxelValue * voxelVolume * scalingFactor / sourceToVoxel_xyz_norm2;
                AtomicAdd_g_f(&projection[projectionOffset + cornerProjectionIndex],
                              value); // Atomic version of projection[ind] += value;
            }
        } else
        {
            double3 sourceToVoxel_xyz = voxelcenter_xyz - sourcePosition;
            double sourceToVoxel_xyz_norm2 = dot(sourceToVoxel_xyz, sourceToVoxel_xyz);
            float value = voxelValue * scalingFactor * voxelVolume / sourceToVoxel_xyz_norm2;
            // I assume that the volume point (x,y,z_1) projects to the same px as (x,y,z_2) for any
            // z_1, z_2  This assumption is restricted to the voxel edges, where it holds very
            // accurately  We project the rectangle that lies on the z midline of the voxel on the
            // projector
            double px00, px01, px10, px11;
            double3 vx00, vx01, vx10, vx11;
            vx00 = voxelcorner_xyz + voxelSizes * (double3)(0.0, 0.0, 0.5);
            vx01 = voxelcorner_xyz + voxelSizes * (double3)(1.0, 0.0, 0.5);
            vx10 = voxelcorner_xyz + voxelSizes * (double3)(0.0, 1.0, 0.5);
            vx11 = voxelcorner_xyz + voxelSizes * (double3)(1.0, 1.0, 0.5);
            px00 = projectX(CM, vx00);
            px01 = projectX(CM, vx01);
            px10 = projectX(CM, vx10);
            px11 = projectX(CM, vx11);
            // We now figure out the vertex that projects to minimum and maximum px
            double pxx_min, pxx_max; // Minimum and maximum values of projector x coordinate
            int max_PX,
                min_PX; // Pixel to which are the voxels with minimum and maximum values are
                        // projected
            // pxx_min = fmin(fmin(px00, px01), fmin(px10, px11));
            // pxx_max = fmax(fmax(px00, px01), fmax(px10, px11));
            double3* V_ccw[4]; // Point in which maximum is achieved and counter clock wise points
            // from the minimum voxel
            double* PX_ccw[4]; // Point in which maximum is achieved and counter clock wise  points
            // from the minimum voxel
            if(px00 < px01)
            {
                if(px00 < px10)
                {
                    pxx_min = px00;
                    V_ccw[0] = &vx00;
                    V_ccw[1] = &vx01;
                    V_ccw[2] = &vx11;
                    V_ccw[3] = &vx10;
                    PX_ccw[0] = &px00;
                    PX_ccw[1] = &px01;
                    PX_ccw[2] = &px11;
                    PX_ccw[3] = &px10;
                    if(px10 > px11)
                    {
                        pxx_max = px10;
                    } else if(px01 > px11)
                    {
                        pxx_max = px01;
                    } else
                    {
                        pxx_max = px11;
                    }
                } else if(px10 < px11)
                {
                    pxx_min = px10;
                    V_ccw[0] = &vx10;
                    V_ccw[1] = &vx00;
                    V_ccw[2] = &vx01;
                    V_ccw[3] = &vx11;
                    PX_ccw[0] = &px10;
                    PX_ccw[1] = &px00;
                    PX_ccw[2] = &px01;
                    PX_ccw[3] = &px11;
                    if(px01 > px11)
                    {
                        pxx_max = px01;
                    } else
                    {
                        pxx_max = px11;
                    }
                } else
                {
                    pxx_min = px11;
                    pxx_max = px01;
                    V_ccw[0] = &vx11;
                    V_ccw[1] = &vx10;
                    V_ccw[2] = &vx00;
                    V_ccw[3] = &vx01;
                    PX_ccw[0] = &px11;
                    PX_ccw[1] = &px10;
                    PX_ccw[2] = &px00;
                    PX_ccw[3] = &px01;
                }

            } else if(px01 < px11)
            {
                pxx_min = px01;
                V_ccw[0] = &vx01;
                V_ccw[1] = &vx11;
                V_ccw[2] = &vx10;
                V_ccw[3] = &vx00;
                PX_ccw[0] = &px01;
                PX_ccw[1] = &px11;
                PX_ccw[2] = &px10;
                PX_ccw[3] = &px00;
                if(px00 > px10)
                {
                    pxx_max = px00;
                } else if(px11 > px10)
                {
                    pxx_max = px11;
                } else
                {
                    pxx_max = px10;
                }
            } else if(px11 < px10)
            {
                pxx_min = px11;
                V_ccw[0] = &vx11;
                V_ccw[1] = &vx10;
                V_ccw[2] = &vx00;
                V_ccw[3] = &vx01;
                PX_ccw[0] = &px11;
                PX_ccw[1] = &px10;
                PX_ccw[2] = &px00;
                PX_ccw[3] = &px01;
                if(px00 > px10)
                {
                    pxx_max = px00;
                } else
                {
                    pxx_max = px10;
                }
            } else
            {
                pxx_min = px10;
                pxx_max = px00;
                V_ccw[0] = &vx10;
                V_ccw[1] = &vx00;
                V_ccw[2] = &vx01;
                V_ccw[3] = &vx11;
                PX_ccw[0] = &px10;
                PX_ccw[1] = &px00;
                PX_ccw[2] = &px01;
                PX_ccw[3] = &px11;
            }

            max_PX = convert_int_rtn(pxx_max + 0.5);
            min_PX = convert_int_rtn(pxx_min + 0.5);
            /*
                        // DEBUG START
                        if(i == 1 && j == 0 && k == 0)
                        {
                            AtomicAdd_g_f(&projection[0], min_PX);
                            AtomicAdd_g_f(&projection[1], max_PX);
                            AtomicAdd_g_f(&projection[2], pxx_min);
                            AtomicAdd_g_f(&projection[3], pxx_max);
                            AtomicAdd_g_f(&projection[4], *PX_ccw[0]);
                            AtomicAdd_g_f(&projection[5], *PX_ccw[1]);
                            AtomicAdd_g_f(&projection[6], *PX_ccw[2]);
                            AtomicAdd_g_f(&projection[7], *PX_ccw[3]);
                            AtomicAdd_g_f(&projection[8], (*(V_ccw[0]))[0]);
                            AtomicAdd_g_f(&projection[9], (*(V_ccw[0]))[1]);
                            AtomicAdd_g_f(&projection[10], (*(V_ccw[0]))[2]);
                            AtomicAdd_g_f(&projection[11], (*(V_ccw[1]))[0]);
                            AtomicAdd_g_f(&projection[12], (*(V_ccw[1]))[1]);
                            AtomicAdd_g_f(&projection[13], (*(V_ccw[1]))[2]);
                            AtomicAdd_g_f(&projection[14], (*(V_ccw[2]))[0]);
                            AtomicAdd_g_f(&projection[15], (*(V_ccw[2]))[1]);
                            AtomicAdd_g_f(&projection[16], (*(V_ccw[2]))[2]);
                            AtomicAdd_g_f(&projection[17], (*(V_ccw[3]))[0]);
                            AtomicAdd_g_f(&projection[18], (*(V_ccw[3]))[1]);
                            AtomicAdd_g_f(&projection[19], (*(V_ccw[3]))[2]);
                        }
                        // DEBUG
            */
            if(max_PX >= 0 && min_PX < pdims.x)
            {
                if(max_PX == min_PX) // These indices are in the admissible range
                {
                    // insertEdgeValues(&projection[projectionOffset], CM, (vx00 + vx11) / 2,
                    // min_PX, value,
                    //                 voxelSizes, pdims);
                    exactEdgeValues(&projection[projectionOffset], CM, (vx00 + vx11) / 2, min_PX,
                                    value, voxelSizes, pdims);
                } else
                {

                    double lastSectionSize, nextSectionSize, polygonSize;
                    double3 lastInt, nextInt, Int;
                    int I = max(-1, min_PX);
                    int I_STOP = min(max_PX, pdims.x);
                    int numberOfEdges;
                    double factor;
                    // Section of the square that corresponds to the indices < i
                    // CCW and CW coordinates of the last intersection on the lines specified by the
                    // points in V_ccw
                    lastSectionSize = exactIntersectionPoints(
                        ((double)I) + 0.5, V_ccw[0], V_ccw[1], V_ccw[2], V_ccw[3], PX_ccw[0],
                        PX_ccw[1], PX_ccw[2], PX_ccw[3], CM, &lastInt);
                    /*                  // DEBUG START
                                      if(i == 1 && j == 0 && k == 0)
                                      {
                                          double XXX = ((double)I) + 0.5;
                                          AtomicAdd_g_f(&projection[19], XXX);
                                          if(XXX < (*PX_ccw[1]))
                                          {
                                              AtomicAdd_g_f(&projection[20], 1.0);
                                          } else
                                          {
                                              AtomicAdd_g_f(&projection[20], -1.0);
                                          }
                                          if(XXX < (*PX_ccw[2]))
                                          {
                                              AtomicAdd_g_f(&projection[21], 1.0);
                                          } else
                                          {
                                              AtomicAdd_g_f(&projection[21], -1.0);
                                          }
                                          double p, q, tmp, totalweight;
                                          double3 v_cw, v_ccw, shift;
                                          double3 Fvector = CM.s012 - XXX * CM.s89a;
                                          double Fconstant = CM.s3 - XXX * CM.sb;
                                          double3 v0, v1, v2, v3;
                                          v0 = *V_ccw[0];
                                          v1 = *V_ccw[1];
                                          v2 = *V_ccw[2];
                                          v3 = *V_ccw[3];
                                          double FproductA = dot(v1, Fvector);
                                          double FproductB = dot(v2, Fvector);
                                          double3 centroid;
                                          p = (FproductA + Fconstant) / (FproductA - FproductB);
                                          v_ccw = (v1) * (1.0 - p) + (v2)*p;
                                          AtomicAdd_g_f(&projection[22], p);
                                          AtomicAdd_g_f(&projection[22], p);
                                          AtomicAdd_g_f(&projection[22], p);
                                          AtomicAdd_g_f(&projection[22], p);
                                          if(XXX < (*PX_ccw[3]))
                                          {
                                              AtomicAdd_g_f(&projection[23], 1.0);
                                              q = (dot(v0, Fvector) + Fconstant)
                                                  / (dot(v0, Fvector) - dot(v3, Fvector));
                                              v_cw = (v0) * (1.0 - q) + (v3)*q;
                                              centroid = ((v1) + v_cw + (q / (p + q)) * (v0) + (p /
                       (p + q)) * v_ccw) / 3.0; AtomicAdd_g_f(&projection[24], p / (p + q));
                                              AtomicAdd_g_f(&projection[27], v_ccw[0] * v_ccw[1] *
                       v_ccw[2]); AtomicAdd_g_f(&projection[28], v_cw[0] * v_cw[1] * v_cw[2]);
                                              AtomicAdd_g_f(&projection[29], v_ccw[2]);
                                          } else
                                          {
                                              AtomicAdd_g_f(&projection[23], -1.0);
                                              q = (dot(v3, Fvector) + Fconstant) / (dot(v3, Fvector)
                       - FproductB); v_cw = (v3) * (1.0 - q) + (v2)*q; tmp = (1.0 - p) * (1.0 - q) *
                       0.5; totalweight = 1.0 - tmp; centroid = (((v0) + (v2)) / 2 - tmp * (v_ccw +
                       v_cw + (v2)) / 3.0) / totalweight; AtomicAdd_g_f(&projection[24],
                       totalweight);
                                          }
                                          AtomicAdd_g_f(&projection[25], centroid[0]);
                                          AtomicAdd_g_f(&projection[26], centroid[1]);
                                          AtomicAdd_g_f(&projection[27], centroid[2]);
                                      }
                                      // DEBUG
                      */
                    if(I >= 0)
                    {
                        factor = value * lastSectionSize;
                        // insertEdgeValues(&projection[projectionOffset], CM, lastInt, I, factor,
                        // voxelSizes, pdims);
                        exactEdgeValues(&projection[projectionOffset], CM, lastInt, I, factor,
                                        voxelSizes, pdims);
                    }
                    for(I = I + 1; I < I_STOP; I++)
                    {
                        nextSectionSize = exactIntersectionPoints(
                            ((double)I) + 0.5, V_ccw[0], V_ccw[1], V_ccw[2], V_ccw[3], PX_ccw[0],
                            PX_ccw[1], PX_ccw[2], PX_ccw[3], CM, &nextInt);
                        polygonSize = nextSectionSize - lastSectionSize;
                        Int = (nextSectionSize * nextInt - lastSectionSize * lastInt) / polygonSize;
                        factor = value * polygonSize;
                        // insertEdgeValues(&projection[projectionOffset], CM, Int, I, factor,
                        // voxelSizes, pdims);
                        exactEdgeValues(&projection[projectionOffset], CM, Int, I, factor,
                                        voxelSizes, pdims);
                        lastSectionSize = nextSectionSize;
                        lastInt = nextInt;
                    }
                    if(I_STOP < pdims.x)
                    {
                        polygonSize = 1.0 - lastSectionSize;
                        Int = ((*V_ccw[0] + *V_ccw[2]) * 0.5 - lastSectionSize * lastInt)
                            / polygonSize;
                        factor = value * polygonSize;
                        // insertEdgeValues(&projection[projectionOffset], CM, Int, I, factor,
                        // voxelSizes, pdims);
                        exactEdgeValues(&projection[projectionOffset], CM, Int, I, factor,
                                        voxelSizes, pdims);
                    }
                }
            }
        }
    }
}
