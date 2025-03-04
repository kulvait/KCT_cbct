//==============================proximal.cl=====================================
// This file contains the OpenCL code for the proximal operators mainly used in
// PDHG algorithm associated with Total Variation (TV) minimization.

// This kernel implements the proximal operator of the indicator function for the dual problem
// in the context of Total Variation (TV) minimization. In Chambolle-Pock or PDHG algorithms,
// this operator corresponds to the "shrinkage" or "dual projection" step. It projects the
// vector (G1, G2), which represents the dual variable (associated with the gradient), onto
// the l2-ball of radius 'mu' in L_inf norm.
// This is equivalent to applying the proximal operator of
// the conjugate of the TV norm (the l1 norm of the gradient).

// The goal is to ensure that the gradient norm (magnitude of the vector) does not exceed mu.
// This is a critical step in the PDHG update for the dual variable.
void kernel FLOATvector_infProjectionToLambda2DBall(global float* restrict G1,
                                                    global float* restrict G2,
                                                    float const mu)
{
    size_t gid = get_global_id(0);
    float gx = G1[gid];
    float gy = G2[gid];
    // Compute the magnitude of the vector (gx, gy)
    float f = sqrt(gx * gx + gy * gy);
    // If the magnitude is greater than mu, project the vector onto the mu ball surface
    if(f > mu)
    {
        f = mu / f;
        G1[gid] = gx * f;
        G2[gid] = gy * f;
    }
}

// L1 proximal operator, soft thresholding solving min omega | u - u0 | + 1/2 | u - x_prox |_2^2
void kernel FLOATvector_distL1ProxSoftThreasholding(global float* restrict U0,
                                                    global float* restrict XPROX,
                                                    float const omega)
{
    size_t gid = get_global_id(0);
    float u_0 = U0[gid];
    float x_prox = XPROX[gid];
    float x_min;
    if(u_0 < x_prox - omega)
    {
        x_min = x_prox - omega;
    } else if(u_0 > x_prox + omega)
    {
        x_min = x_prox + omega;
    } else
    {

        x_min = u_0;
        // Relax proper proximal
        // x_min = x_prox;
    }
    XPROX[gid] = x_min;
}
//==============================END proximal.cl=====================================
