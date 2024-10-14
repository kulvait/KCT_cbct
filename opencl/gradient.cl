//==============================gradient.cl=====================================
// This file contains the functions for gradient and its adjoint operators.
// The aim is to have fully adjoint nonsignular matrices of these operations.
// In turn boundary conditions are very relaxed.
// Operators are fully Toeplitz.
// Minus divergence is the adjoint of gradient.
// Implementention with central differences as well as forward differences.
// Table can be found https://en.wikipedia.org/wiki/Finite_difference_coefficient

void kernel FLOATvector_Gradient2D_centralDifference_3point(global const float* restrict F,
                                                            global float* restrict GX,
                                                            global float* restrict GY,
                                                            private int3 vdims,
                                                            private float3 voxelSizes)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int k = get_global_id(2);
    const int IND = VOXELINDEX(i, j, k, vdims);

    // Precomputed coefficients for central differences
    const float coef_x = 0.5f / voxelSizes.x;
    const float coef_y = 0.5f / voxelSizes.y;

    // Compute central difference in X direction
    if(i > 0 && i < vdims.x - 1)
    {
        // Central difference: (F[i+1] - F[i-1]) / (2*voxelSizes.x)
        GX[IND] = coef_x * (F[IND + 1] - F[IND - 1]);
    } else if(i == 0)
    {
        GX[IND] = coef_x * F[IND + 1];
    } else if(i == vdims.x - 1)
    {
        GX[IND] = -coef_x * F[IND - 1];
    }

    // Compute central difference in Y direction
    if(j > 0 && j < vdims.y - 1)
    {
        // Central difference: (F[j+1] - F[j-1]) / (2*voxelSizes.y)
        GY[IND] = coef_y * (F[IND + vdims.x] - F[IND - vdims.x]);
    } else if(j == 0)
    {
        GY[IND] = coef_y * F[IND + vdims.x];
    } else if(j == vdims.y - 1)
    {
        GY[IND] = -coef_y * F[IND - vdims.x];
    }
}

void kernel FLOATvector_Gradient2D_centralDifference_3point_adjoint(global const float* restrict GX,
                                                                    global const float* restrict GY,
                                                                    global float* restrict D,
                                                                    private int3 vdims,
                                                                    private float3 voxelSizes)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int k = get_global_id(2);
    const int IND = VOXELINDEX(i, j, k, vdims);

    float divX = 0.0f;
    float divY = 0.0f;

    // Precomputed coefficients for central differences
    const float coef_x = 0.5f / voxelSizes.x;
    const float coef_y = 0.5f / voxelSizes.y;

    // Compute minus divergence in X direction (adjoint to central difference)
    if(i > 0 && i < vdims.x - 1)
    {
        // Central difference: GX[i+1] - GX[i-1] / (2 * voxelSizes.x)
        divX = coef_x * (GX[IND - 1] - GX[IND + 1]);
    } else if(i == 0)
    {
        divX = -coef_x * GX[IND + 1];
    } else if(i == vdims.x - 1)
    {
        divX = coef_x * GX[IND - 1];
    }

    // Compute minus divergence in Y direction (adjoint to central difference)
    if(j > 0 && j < vdims.y - 1)
    {
        // Central difference: GY[j+1] - GY[j-1] / (2 * voxelSizes.y)
        divY = coef_y * (GY[IND - vdim.x] - GY[IND + vdims.x]);
    } else if(j == 0)
    {
        divY = -coef_y * GY[IND + vdim.x];
    } else if(j == vdims.y - 1)
    {
        divY = coef_y * GY[IND - vdims.x];
    }

    // Store the computed minus divergence in D
    D[IND] = divX + divY;
}

void kernel FLOATvector_Gradient2D_forwardDifference_2point(global const float* restrict F,
                                                            global float* restrict GX,
                                                            global float* restrict GY,
                                                            private int3 vdims,
                                                            private float3 voxelSizes)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int k = get_global_id(2);
    const int IND = VOXELINDEX(i, j, k, vdims);

    // Precompute the value of F at the current voxel index
    const float F_IND = F[IND];

    // Precomputed coefficients for forward differences
    const float coef_x = 1.0f / voxelSizes.x;
    const float coef_y = 1.0f / voxelSizes.y;

    // Compute forward difference in X direction
    if(i < vdims.x - 1)
    {
        // Forward difference: F[i+1] - F[i]
        GX[IND] = coef_x * (F[IND + 1] - F_IND);
    } else
    {
        GX[IND] = coef_x * F_IND; // Boundary case, just use F[IND]
    }

    // Compute forward difference in Y direction
    if(j < vdims.y - 1)
    {
        // Forward difference: F[j+1] - F[j]
        GY[IND] = coef_y * (F[IND + vdims.x] - F_IND);
    } else
    {
        GY[IND] = coef_y * F_IND; // Boundary case, just use F[IND]
    }
}

void kernel
FLOATvector_Gradient2D_backwardDifference_2point_adjoint(global const float* restrict GX,
                                                         global const float* restrict GY,
                                                         global float* restrict D,
                                                         private int3 vdims,
                                                         private float3 voxelSizes)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int k = get_global_id(2);
    const int IND = VOXELINDEX(i, j, k, vdims);

    // Precomputed coefficients for backward differences
    const float coef_x = 1.0f / voxelSizes.x;
    const float coef_y = 1.0f / voxelSizes.y;

    float divX = 0.0f;
    float divY = 0.0f;

    // Compute backward difference in X direction (adjoint of forward difference)
    if(i > 0)
    {
        // Backward difference: GX[i] - GX[i-1]
        divX = coef_x * (GX[IND - 1] - GX[IND]);
    } else
    {
        divX = -coef_x * GX[IND]; // At the boundary, just use GX[IND]
    }

    // Compute backward difference in Y direction (adjoint of forward difference)
    if(j > 0)
    {
        // Backward difference: GY[j] - GY[j-1]
        divY = coef_y * (GY[IND - vdims.x] - GY[IND]);
    } else
    {
        divY = -coef_y * GY[IND]; // At the boundary, just use GY[IND]
    }

    // Store the computed minus divergence in D (i.e., the adjoint operation)
    D[IND] = divX + divY;
}

void kernel FLOATvector_Gradient2D_forwardDifference_3point(global const float* restrict F,
                                                            global float* restrict GX,
                                                            global float* restrict GY,
                                                            private int3 vdims,
                                                            private float3 voxelSizes)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int k = get_global_id(2);
    const int IND = VOXELINDEX(i, j, k, vdims);

    // Precompute the value of F at the current voxel index
    const float F_IND = F[IND];

    // Precomputed coefficients for the 3-point forward difference
    const float coef_x = 1.0f / voxelSizes.x;
    const float coef_y = 1.0f / voxelSizes.y;

    // Compute forward difference in X direction using the 3-point stencil
    if(i < vdims.x - 2)
    {
        // 3-point forward difference: -3/2*F[i] + 2*F[i+1] - 1/2*F[i+2]
        GX[IND] = coef_x * (-1.5f * F_IND + 2.0f * F[IND + 1] - 0.5f * F[IND + 2]);
    } else if(i == vdims.x - 2)
    {
        // Use two-point forward difference for near-boundary case
        GX[IND] = coef_x * (-1.5f * F_IND + 2.0f * F[IND + 1]);
    } else
    {
        // Boundary case, just use forward difference approximation
        GX[IND] = coef_x * (-1.5f * F_IND);
    }

    // Compute forward difference in Y direction using the 3-point stencil
    if(j < vdims.y - 2)
    {
        // 3-point forward difference: -3/2*F[j] + 2*F[j+1] - 1/2*F[j+2]
        GY[IND] = coef_y * (-1.5f * F_IND + 2.0f * F[IND + vdims.x] - 0.5f * F[IND + 2 * vdims.x]);
    } else if(j == vdims.y - 2)
    {
        // Use two-point forward difference for near-boundary case
        GY[IND] = coef_y * (-1.5f * F_IND + 2.0f * F[IND + vdims.x]);
    } else
    {
        // Boundary case, just use forward difference approximation
        GY[IND] = coef_y * (-1.5f * F_IND);
    }
}

void kernel
FLOATvector_Gradient2D_forwardDifference_3point_adjoint(global const float* restrict GX,
                                                         global const float* restrict GY,
                                                         global float* restrict D,
                                                         private int3 vdims,
                                                         private float3 voxelSizes)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int k = get_global_id(2);
    const int IND = VOXELINDEX(i, j, k, vdims);

    // Precomputed coefficients for the 3-point backward difference
    const float coef_x = 1.0f / voxelSizes.x;
    const float coef_y = 1.0f / voxelSizes.y;

    float divX = 0.0f;
    float divY = 0.0f;

    // Compute backward difference in X direction (adjoint of the forward difference)
    if(i > 1)
    {
        // 3-point backward difference: 1/2*GX[i-2] - 2*GX[i-1] + 3/2*GX[i]
        divX = coef_x * (-0.5f * GX[IND - 2] + 2.0f * GX[IND - 1] - 1.5f * GX[IND]);
    } else if(i == 1)
    {
        // Near-boundary case, use two-point backward difference
        divX = coef_x * (2.0f * GX[IND - 1] - 1.5f * GX[IND]);
    } else
    {
        // Boundary case, just use the backward difference approximation
        divX = coef_x * -1.5f * GX[IND];
    }

    // Compute backward difference in Y direction (adjoint of the forward difference)
    if(j > 1)
    {
        // 3-point backward difference: 1/2*GY[j-2] - 2*GY[j-1] + 3/2*GY[j]
        divY = coef_y * (-0.5f * GY[IND - 2 * vdims.x] + 2.0f * GY[IND - vdims.x] - 1.5f * GY[IND]);
    } else if(j == 1)
    {
        // Near-boundary case, use two-point backward difference
        divY = coef_y * (2.0f * GY[IND - vdims.x] - 1.5f * GY[IND]);
    } else
    {
        // Boundary case, just use the backward difference approximation
        divY = coef_y * -1.5f * GY[IND];
    }

    // Store the computed minus divergence
    D[IND] = divX + divY;
}

void kernel FLOATvector_Gradient2D_centralDifference_5point(global const float* restrict F,
                                                            global float* restrict GX,
                                                            global float* restrict GY,
                                                            private int3 vdims,
                                                            private float3 voxelSizes)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int k = get_global_id(2);
    const int IND = VOXELINDEX(i, j, k, vdims);
    float gradX = 0.0f;
    float gradY = 0.0f;

    // Precomputed coefficients incorporating voxel sizes
    const float coef1_x = 2.0f / (3.0f * voxelSizes.x);
    const float coef2_x = 1.0f / (12.0f * voxelSizes.x);

    const float coef1_y = 2.0f / (3.0f * voxelSizes.y);
    const float coef2_y = 1.0f / (12.0f * voxelSizes.y);

    // Gradient in X direction (handle boundary cases)
    if(i > 1 && i < vdims.x - 2)
    {
        // Apply full 5-point stencil
        gradX = coef2_x * F[IND - 2] - coef1_x * F[IND - 1] + coef1_x * F[IND + 1]
            - coef2_x * F[IND + 2];
    } else if(i == 1)
    {
        // Near-boundary stencil (truncate at boundary)
        gradX = -coef1_x * F[IND - 1] + coef1_x * F[IND + 1] - coef2_x * F[IND + 2];
    } else if(i == 0)
    {
        // Boundary case: only use forward difference with truncation
        gradX = coef1_x * F[IND + 1] - coef2_x * F[IND + 2];
    } else if(i == vdims.x - 2)
    {
        // Near-boundary on the other side
        gradX = coef2_x * F[IND - 2] - coef1_x * F[IND - 1] + coef1_x * F[IND + 1];
    } else if(i == vdims.x - 1)
    {
        // Boundary case: backward difference with truncation
        gradX = coef2_x * F[IND - 2] - coef1_x * F[IND - 1];
    }

    // Gradient in Y direction (handle boundary cases)
    if(j > 1 && j < vdims.y - 2)
    {
        // Apply full 5-point stencil
        gradY = coef2_y * F[IND - 2 * vdims.x] - coef1_y * F[IND - vdims.x]
            + coef1_y * F[IND + vdims.x] - coef2_y * F[IND + 2 * vdims.x];
    } else if(j == 1)
    {
        // Near-boundary stencil (truncate at boundary)
        gradY = -coef1_y * F[IND - vdims.x] + coef1_y * F[IND + vdims.x]
            - coef2_y * F[IND + 2 * vdims.x];
    } else if(j == 0)
    {
        // Boundary case: only use forward difference with truncation
        gradY = coef1_y * F[IND + vdims.x] - coef2_y * F[IND + 2 * vdims.x];
    } else if(j == vdims.y - 2)
    {
        // Near-boundary on the other side
        gradY = coef2_y * F[IND - 2 * vdims.x] - coef1_y * F[IND - vdims.x]
            + coef1_y * F[IND + vdims.x];
    } else if(j == vdims.y - 1)
    {
        // Boundary case: backward difference with truncation
        gradY = coef2_y * F[IND - 2 * vdims.x] - coef1_y * F[IND - vdims.x];
    }

    // Store the computed gradients in GX and GY
    GX[IND] = gradX;
    GY[IND] = gradY;
}

void kernel FLOATvector_Gradient2D_centralDifference_5point_adjoint(global const float* restrict GX,
                                                                    global const float* restrict GY,
                                                                    global float* restrict D,
                                                                    private int3 vdims,
                                                                    private float3 voxelSizes)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int k = get_global_id(2);
    const int IND = VOXELINDEX(i, j, k, vdims);

    float divX = 0.0f;
    float divY = 0.0f;

    // Precomputed coefficients incorporating voxel sizes
    const float coef1_x = 2.0f / (3.0f * voxelSizes.x);
    const float coef2_x = 1.0f / (12.0f * voxelSizes.x);

    const float coef1_y = 2.0f / (3.0f * voxelSizes.y);
    const float coef2_y = 1.0f / (12.0f * voxelSizes.y);

    // Compute minus divergence in X direction (handle boundary cases)
    if(i > 1 && i < vdims.x - 2)
    {
        // Full 5-point stencil for divergence
        divX = -coef2_x * GX[IND - 2] + coef1_x * GX[IND - 1] - coef1_x * GX[IND + 1]
            + coef2_x * GX[IND + 2];
    } else if(i == 1)
    {
        // Near-boundary stencil (truncate at boundary)
        divX = coef1_x * GX[IND - 1] - coef1_x * GX[IND + 1] + coef2_x * GX[IND + 2];
    } else if(i == 0)
    {
        // Boundary case: forward difference with truncation
        divX = -coef1_x * GX[IND + 1] + coef2_x * GX[IND + 2];
    } else if(i == vdims.x - 2)
    {
        // Near-boundary on the other side
        divX = -coef2_x * GX[IND - 2] + coef1_x * GX[IND - 1] - coef1_x * GX[IND + 1];
    } else if(i == vdims.x - 1)
    {
        // Boundary case: backward difference with truncation
        divX = -coef2_x * GX[IND - 2] + coef1_x * GX[IND - 1];
    }

    // Compute minus divergence in Y direction (handle boundary cases)
    if(j > 1 && j < vdims.y - 2)
    {
        // Full 5-point stencil for divergence
        divY = -coef2_y * GY[IND - 2 * vdims.x] + coef1_y * GY[IND - vdims.x]
            - coef1_y * GY[IND + vdims.x] + coef2_y * GY[IND + 2 * vdims.x];
    } else if(j == 1)
    {
        // Near-boundary stencil (truncate at boundary)
        divY = coef1_y * GY[IND - vdims.x] - coef1_y * GY[IND + vdims.x]
            + coef2_y * GY[IND + 2 * vdims.x];
    } else if(j == 0)
    {
        // Boundary case: forward difference with truncation
        divY = coef1_y * GY[IND + vdims.x] - coef2_y * GY[IND + 2 * vdims.x];
    } else if(j == vdims.y - 2)
    {
        // Near-boundary on the other side
        divY = -coef2_y * GY[IND - 2 * vdims.x] + coef1_y * GY[IND - vdims.x]
            - coef1_y * GY[IND + vdims.x];
    } else if(j == vdims.y - 1)
    {
        // Boundary case: backward difference with truncation
        divY = -coef2_y * GY[IND - 2 * vdims.x] + coef1_y * GY[IND - vdims.x];
    }

    // Store the computed minus divergence in D
    D[IND] = divX + divY;
}

//==============================END gradient.cl=================================
