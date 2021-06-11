//==============================rescaleProjections.cl=====================================
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
    const uint IND = projectionOffset + px * pdims.y + py;
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
    const uint IND = projectionOffset + px * pdims.y + py;
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
//==============================END rescaleProjections.cl=====================================
