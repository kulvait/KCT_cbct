//==============================convolution.cl=====================================
// see https://www.evl.uic.edu/kreda/gpu/image-convolution/
#define VOXELINDEX(i, j, k, vdims)                                                                 \
    {                                                                                              \
        (i) + ((j) + (k)*vdims.y) * vdims.x                                                        \
    }

#define REFLECTION33BOUNDARY()                                                                     \
    {                                                                                              \
        if(i == 0)                                                                                 \
        {                                                                                          \
            for(int lj = 0; lj < 3; lj++)                                                          \
            {                                                                                      \
                cube[0][lj] = cube[2][lj];                                                         \
            }                                                                                      \
        }                                                                                          \
        if(i + 1 == vdims.x)                                                                       \
        {                                                                                          \
            for(int lj = 0; lj < 3; lj++)                                                          \
            {                                                                                      \
                cube[2][lj] = cube[0][lj];                                                         \
            }                                                                                      \
        }                                                                                          \
        if(j == 0)                                                                                 \
        {                                                                                          \
            for(int li = 0; li < 3; li++)                                                          \
            {                                                                                      \
                cube[li][0] = cube[li][2];                                                         \
            }                                                                                      \
        }                                                                                          \
        if(j + 1 == vdims.y)                                                                       \
        {                                                                                          \
            for(int li = 0; li < 3; li++)                                                          \
            {                                                                                      \
                cube[li][2] = cube[li][0];                                                         \
            }                                                                                      \
        }                                                                                          \
    }

void kernel FLOATvector_2Dconvolution3x3ZeroBoundary(global const float* restrict A,
                                                     global float* restrict B,
                                                     private int3 vdims,
                                                     float16 _convolutionKernel)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int k = get_global_id(2);
    const int IND = VOXELINDEX(i, j, k, vdims);
    const int convolutionKernelRadius = 1;
    const int convolutionKernelSize = 3;
    float sum = 0.0f;
    float* convolutionKernel = (float*)&_convolutionKernel;
    // Manipulation to fix problems on boundaries
    int LIMIN = -min(i - convolutionKernelRadius, 0);
    int LJMIN = -min(j - convolutionKernelRadius, 0);
    // Find one behind
    int ILIMIT = convolutionKernelSize + vdims.x - max(i + convolutionKernelRadius + 1, vdims.x);
    int JLIMIT = convolutionKernelSize + vdims.y - max(j + convolutionKernelRadius + 1, vdims.y);
    int IRANGE = ILIMIT - LIMIN;
    int yskip = vdims.x - IRANGE;
    int ylocalskip = convolutionKernelSize - IRANGE;
    int localIndex = LJMIN * convolutionKernelSize + LIMIN;
    int index = VOXELINDEX(i + LIMIN - 1, j + LJMIN - 1, k, vdims);
    for(int j_loc = LJMIN; j_loc < JLIMIT; j_loc++)
    {
        for(int i_loc = LIMIN; i_loc < ILIMIT; i_loc++)
        {
            sum += A[index] * convolutionKernel[localIndex];
            index++;
            localIndex++;
        }
        index += yskip;
        localIndex += ylocalskip;
    }
    B[IND] = sum;
}

void kernel FLOATvector_2Dconvolution3x3ReflectionBoundary(global const float* restrict A,
                                                           global float* restrict B,
                                                           private int3 vdims,
                                                           float16 _convolutionKernel)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int k = get_global_id(2);
    const int IND = VOXELINDEX(i, j, k, vdims);
    const int convolutionKernelRadius = 1;
    const int convolutionKernelSize = 3;
    const int LIMIN = -min(i - convolutionKernelRadius, 0);
    const int LJMIN = -min(j - convolutionKernelRadius, 0);
    const int ILIMIT
        = convolutionKernelSize + vdims.x - max(i + convolutionKernelRadius + 1, vdims.x);
    const int JLIMIT
        = convolutionKernelSize + vdims.y - max(j + convolutionKernelRadius + 1, vdims.y);
    const int JSKIP = vdims.x - (ILIMIT - LIMIN);
    float cube[3][3]; // First fill this object where possible
    int index = VOXELINDEX(i + LIMIN - convolutionKernelRadius, j + LJMIN - convolutionKernelRadius,
                           k, vdims);
    for(int lj = LJMIN; lj < JLIMIT; lj++)
    {
        for(int li = LIMIN; li < ILIMIT; li++)
        {
            cube[li][lj] = A[index];
            index++;
        }
        index += JSKIP;
    }
    REFLECTION33BOUNDARY();
    float sum = 0.0f;
    float* convolutionKernel = (float*)&_convolutionKernel;
    int localIndex = 0;
    for(int lj = 0; lj < convolutionKernelSize; lj++)
    {
        for(int li = 0; li < convolutionKernelSize; li++)
        {
            sum += cube[li][lj] * convolutionKernel[localIndex];
            localIndex++;
        }
    }
    B[IND] = sum;
}

/**
 * @brief Computes a gradient in 3D based on the Sobel operator
 *
 * @param F
 * @param GX
 * @param GY
 * @param GZ
 * @param vdims
 * @param voxelSizes
 *
 * @return
 */
void kernel
FLOATvector_3DconvolutionGradientSobelFeldmanReflectionBoundary(global const float* restrict F,
                                                                global float* restrict GX,
                                                                global float* restrict GY,
                                                                global float* restrict GZ,
                                                                private int3 vdims,
                                                                private float3 voxelSizes)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int k = get_global_id(2); /*
     if(k == 3 && i == 5 && j == 55)
     {
         printf("Calling with i=%d, j=%d, k=%d", i, j, k);
     }*/
    const int IND = VOXELINDEX(i, j, k, vdims);
    float cube[3][3][3]; // First fill this object where possible
    int DJ = vdims.x;
    int DK = vdims.x * vdims.y;
    int LIMIN = -min(i - 1, 0);
    int LJMIN = -min(j - 1, 0);
    int LKMIN = -min(k - 1, 0);
    // One behind the limit
    int ILIMIT = 3 + vdims.x - max(i + 2, vdims.x);
    int JLIMIT = 3 + vdims.y - max(j + 2, vdims.y);
    int KLIMIT = 3 + vdims.z - max(k + 2, vdims.z);
    int IRANGE = ILIMIT - LIMIN;
    int JRANGE = JLIMIT - LJMIN;
    int JSKIP = DJ - IRANGE;
    int KSKIP = DK - JRANGE * DJ;
    int index = VOXELINDEX(i + LIMIN - 1, j + LJMIN - 1, k + LKMIN - 1, vdims);
    /*
        if(k == 30 && j == 511)
        {
            printf("i=%d j=%d k=%d index=%d IND=%d LIMIN=%d, LJMIN=%d, LKMIN=%d start=(%d, %d, %d) "
                   "LIMITS=(%d, "
                   "%d, %d)",
                   i, j, k, index, IND, LIMIN, LJMIN, LKMIN, i - 1 + LIMIN, j - 1 + LJMIN,
                   k - 1 + LKMIN, ILIMIT, JLIMIT, KLIMIT);
        }
    */
    for(int lk = LKMIN; lk < KLIMIT; lk++)
    {
        for(int lj = LJMIN; lj < JLIMIT; lj++)
        {
            for(int li = LIMIN; li < ILIMIT; li++)
            {

                int index_computed = VOXELINDEX(i - 1 + li, j - 1 + lj, k - 1 + lk, vdims);
                if(index_computed != index)
                {
                    printf("Problem with index alignment index=%d, index_computed=%d", index,
                           index_computed);
                }

                cube[li][lj][lk] = F[index];
                /*
                        if(k == 3 && i == 5 && j == 55)
                        {
                            printf("I in [%d, %d] J in [%d, %d] K in [%d, %d] index=%d vdims.x=%d
                   vdims.y=%d vdims.z=%d" "F[index]=%f voxelSizes.x=%f", LIMIN, ILIMIT, LJMIN,
                   JLIMIT, LKMIN, KLIMIT, index, vdims.x, vdims.y, vdims.z, cube[li][lj][lk],
                   voxelSizes.x);
                        }*/
                index++;
            }
            index += JSKIP;
        }
        index += KSKIP;
    }
    // reflections
    if(i == 0)
    {
        // i reflection li=0 lj=. lk=. equals li=2 lj=. lk=.
        // if 0, lj, lk invalid so is 2,lj,lk
        for(int lj = 0; lj < 3; lj++)
        {
            for(int lk = 0; lk < 3; lk++)
            {
                cube[0][lj][lk] = cube[2][lj][lk];
            }
        }
    } else if(i + 1 == vdims.x)
    {
        // i reflection li=2 lj=. lk=. equals li=0 lj=. lk=.
        // if 2, lj, lk invalid so is 0,lj,lk
        for(int lj = 0; lj < 3; lj++)
        {
            for(int lk = 0; lk < 3; lk++)
            {
                cube[2][lj][lk] = cube[0][lj][lk];
            }
        }
    }
    if(j == 0)
    {
        // j reflection li=. lj=0 lk=. equals li=. lj=2 lk=.
        // if li, 0, lk invalid so is li,0,lk
        for(int li = 0; li < 3; li++)
        {
            for(int lk = 0; lk < 3; lk++)
            {
                cube[li][0][lk] = cube[li][2][lk];
            }
        }
    } else if(j + 1 == vdims.y)
    {
        // j reflection li=. lj=2 lk=. equals li=. lj=0 lk=.
        // if li, 0, lk invalid so is li,0,lk
        for(int li = 0; li < 3; li++)
        {
            for(int lk = 0; lk < 3; lk++)
            {
                cube[li][2][lk] = cube[li][0][lk];
            }
        }
    }
    if(k == 0)
    {
        // k reflection li=. lj=. lk=0 equals li=. lj=. lk=2
        // if li, lj, 2 invalid so is li,lj,0
        for(int li = 0; li < 3; li++)
        {
            for(int lj = 0; lj < 3; lj++)
            {
                cube[li][lj][0] = cube[li][lj][2];
            }
        }
    } else if(k + 1 == vdims.z)
    {
        // k reflection li=. lj=. lk=2 equals li=. lj=. lk=0
        // if li, lj, 2 invalid so is li,lj,0
        for(int li = 0; li < 3; li++)
        {
            for(int lj = 0; lj < 3; lj++)
            {
                cube[li][lj][2] = cube[li][lj][0];
            }
        }
    }
    // Generated code to avoid errors
    // sobeld0=[1,2,1]
    // sobeld1=[-1,0,1]
    // sobelnd0=sobeld0/np.linalg.norm(sobeld0, ord=1)
    // sobelnd1=sobeld1/np.linalg.norm(sobeld1, ord=1)
    // sx=generateConvolutionCompact3D(sobelnd1, sobelnd0, sobelnd0)#Sobel3Dx
    // sy=generateConvolutionCompact3D(sobelnd0, sobelnd1, sobelnd0)#Sobel3Dy
    // sz=generateConvolutionCompact3D(sobelnd0, sobelnd0, sobelnd1)#Sobel3Dz
    // print("grad.x=%s;"%sx)
    // print("grad.y=%s;"%sy)
    // print("grad.z=%s;"%sz)
    float3 grad;
    grad.x = 0.03125f
            * (cube[2][0][0] + cube[2][0][2] + cube[2][2][0] + cube[2][2][2] - cube[0][0][0]
               - cube[0][0][2] - cube[0][2][0] - cube[0][2][2])
        + 0.0625f
            * (cube[2][0][1] + cube[2][1][0] + cube[2][1][2] + cube[2][2][1] - cube[0][0][1]
               - cube[0][1][0] - cube[0][1][2] - cube[0][2][1])
        + 0.125f * (cube[2][1][1] - cube[0][1][1]);
    grad.y = 0.03125f
            * (cube[0][2][0] + cube[0][2][2] + cube[2][2][0] + cube[2][2][2] - cube[0][0][0]
               - cube[0][0][2] - cube[2][0][0] - cube[2][0][2])
        + 0.0625f
            * (cube[0][2][1] + cube[1][2][0] + cube[1][2][2] + cube[2][2][1] - cube[0][0][1]
               - cube[1][0][0] - cube[1][0][2] - cube[2][0][1])
        + 0.125f * (cube[1][2][1] - cube[1][0][1]);
    grad.z = 0.03125f
            * (cube[0][0][2] + cube[0][2][2] + cube[2][0][2] + cube[2][2][2] - cube[0][0][0]
               - cube[0][2][0] - cube[2][0][0] - cube[2][2][0])
        + 0.0625f
            * (cube[0][1][2] + cube[1][0][2] + cube[1][2][2] + cube[2][1][2] - cube[0][1][0]
               - cube[1][0][0] - cube[1][2][0] - cube[2][1][0])
        + 0.125f * (cube[1][1][2] - cube[1][1][0]);
    grad /= voxelSizes;
    GX[IND] = grad.x;
    GY[IND] = grad.y;
    GZ[IND] = grad.z;
}

/**
 * @brief Computes a gradient in 3D based on the Sobel operator
 *
 * @param F
 * @param GX
 * @param GY
 * @param GZ
 * @param vdims
 * @param voxelSizes
 *
 * @return
 */
void kernel
FLOATvector_3DconvolutionGradientSobelFeldmanZeroBoundary(global const float* restrict F,
                                                          global float* restrict GX,
                                                          global float* restrict GY,
                                                          global float* restrict GZ,
                                                          private int3 vdims,
                                                          private float3 voxelSizes)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int k = get_global_id(2); /*
     if(k == 3 && i == 5 && j == 55)
     {
         printf("Calling with i=%d, j=%d, k=%d", i, j, k);
     }*/
    const int IND = VOXELINDEX(i, j, k, vdims);
    float cube[3][3][3]; // First fill this object where possible
    int DJ = vdims.x;
    int DK = vdims.x * vdims.y;
    int LIMIN = -min(i - 1, 0);
    int LJMIN = -min(j - 1, 0);
    int LKMIN = -min(k - 1, 0);
    // One behind the limit
    int ILIMIT = 3 + vdims.x - max(i + 2, vdims.x);
    int JLIMIT = 3 + vdims.y - max(j + 2, vdims.y);
    int KLIMIT = 3 + vdims.z - max(k + 2, vdims.z);
    int IRANGE = ILIMIT - LIMIN;
    int JRANGE = JLIMIT - LJMIN;
    int JSKIP = DJ - IRANGE;
    int KSKIP = DK - JRANGE * DJ;
    int index = VOXELINDEX(i + LIMIN - 1, j + LJMIN - 1, k + LKMIN - 1, vdims);
    for(int lk = LKMIN; lk < KLIMIT; lk++)
    {
        for(int lj = LJMIN; lj < JLIMIT; lj++)
        {
            for(int li = LIMIN; li < ILIMIT; li++)
            {

                int index_computed = VOXELINDEX(i - 1 + li, j - 1 + lj, k - 1 + lk, vdims);
                if(index_computed != index)
                {
                    printf("Problem with index alignment index=%d, index_computed=%d", index,
                           index_computed);
                }

                cube[li][lj][lk] = F[index];
                index++;
            }
            index += JSKIP;
        }
        index += KSKIP;
    }
    // fill with zeros
    if(i == 0)
    {
        // i reflection li=0 lj=. lk=. equals li=2 lj=. lk=.
        // if 0, lj, lk invalid so is 2,lj,lk
        for(int lj = 0; lj < 3; lj++)
        {
            for(int lk = 0; lk < 3; lk++)
            {
                cube[0][lj][lk] = 0.0f;
            }
        }
    } else if(i + 1 == vdims.x)
    {
        // i reflection li=2 lj=. lk=. equals li=0 lj=. lk=.
        // if 2, lj, lk invalid so is 0,lj,lk
        for(int lj = 0; lj < 3; lj++)
        {
            for(int lk = 0; lk < 3; lk++)
            {
                cube[2][lj][lk] = 0.0f;
            }
        }
    }
    if(j == 0)
    {
        // j reflection li=. lj=0 lk=. equals li=. lj=2 lk=.
        // if li, 0, lk invalid so is li,0,lk
        for(int li = 0; li < 3; li++)
        {
            for(int lk = 0; lk < 3; lk++)
            {
                cube[li][0][lk] = 0.0f;
            }
        }
    } else if(j + 1 == vdims.y)
    {
        // j reflection li=. lj=2 lk=. equals li=. lj=0 lk=.
        // if li, 0, lk invalid so is li,0,lk
        for(int li = 0; li < 3; li++)
        {
            for(int lk = 0; lk < 3; lk++)
            {
                cube[li][2][lk] = 0.0f;
            }
        }
    }
    if(k == 0)
    {
        // k reflection li=. lj=. lk=0 equals li=. lj=. lk=2
        // if li, lj, 2 invalid so is li,lj,0
        for(int li = 0; li < 3; li++)
        {
            for(int lj = 0; lj < 3; lj++)
            {
                cube[li][lj][0] = 0.0f;
            }
        }
    } else if(k + 1 == vdims.z)
    {
        // k reflection li=. lj=. lk=2 equals li=. lj=. lk=0
        // if li, lj, 2 invalid so is li,lj,0
        for(int li = 0; li < 3; li++)
        {
            for(int lj = 0; lj < 3; lj++)
            {
                cube[li][lj][2] = 0.0f;
            }
        }
    }
    float3 grad;
    grad.x = 0.03125f
            * (cube[2][0][0] + cube[2][0][2] + cube[2][2][0] + cube[2][2][2] - cube[0][0][0]
               - cube[0][0][2] - cube[0][2][0] - cube[0][2][2])
        + 0.0625f
            * (cube[2][0][1] + cube[2][1][0] + cube[2][1][2] + cube[2][2][1] - cube[0][0][1]
               - cube[0][1][0] - cube[0][1][2] - cube[0][2][1])
        + 0.125f * (cube[2][1][1] - cube[0][1][1]);
    grad.y = 0.03125f
            * (cube[0][2][0] + cube[0][2][2] + cube[2][2][0] + cube[2][2][2] - cube[0][0][0]
               - cube[0][0][2] - cube[2][0][0] - cube[2][0][2])
        + 0.0625f
            * (cube[0][2][1] + cube[1][2][0] + cube[1][2][2] + cube[2][2][1] - cube[0][0][1]
               - cube[1][0][0] - cube[1][0][2] - cube[2][0][1])
        + 0.125f * (cube[1][2][1] - cube[1][0][1]);
    grad.z = 0.03125f
            * (cube[0][0][2] + cube[0][2][2] + cube[2][0][2] + cube[2][2][2] - cube[0][0][0]
               - cube[0][2][0] - cube[2][0][0] - cube[2][2][0])
        + 0.0625f
            * (cube[0][1][2] + cube[1][0][2] + cube[1][2][2] + cube[2][1][2] - cube[0][1][0]
               - cube[1][0][0] - cube[1][2][0] - cube[2][1][0])
        + 0.125f * (cube[1][1][2] - cube[1][1][0]);
    grad /= voxelSizes;
    GX[IND] = grad.x;
    GY[IND] = grad.y;
    GZ[IND] = grad.z;
}

void kernel FLOATvector_3DisotropicGradient_(global const float* restrict F,
                                             global float* restrict GX,
                                             global float* restrict GY,
                                             global float* restrict GZ,
                                             private int3 vdims,
                                             private float3 voxelSizes)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int k = get_global_id(2);
    const int IND = VOXELINDEX(i, j, k, vdims);
    float v = F[IND];
    float3 grad;
    if(i + 1 == vdims.x)
    {
        grad.x = 0.0f;
    } else
    {
        grad.x = F[IND + 1] - v;
    }
    if(j + 1 == vdims.y)
    {
        grad.y = 0.0f;
    } else
    {
        grad.y = F[IND + vdims.x] - v;
    }
    if(k + 1 == vdims.z)
    {
        grad.z = 0.0f;
    } else
    {
        grad.z = F[IND + vdims.x * vdims.y] - v;
    }
    grad /= voxelSizes;
    GX[IND] = grad.x;
    GY[IND] = grad.y;
    GZ[IND] = grad.z;
}

// Reflection boundary for gradinet and its adjoint divergence operator
void kernel FLOATvector_2DisotropicGradient(global const float* restrict F,
                                            global float* restrict GX,
                                            global float* restrict GY,
                                            private int3 vdims,
                                            private float3 voxelSizes)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int k = get_global_id(2);
    const int IND = VOXELINDEX(i, j, k, vdims);
    float v = F[IND];
    float3 grad;
    if(i + 1 == vdims.x)
    {
        grad.x = -v;
    } else
    {
        grad.x = F[IND + 1] - v;
    }
    if(j + 1 == vdims.y)
    {
        grad.y = -v;
    } else
    {
        grad.y = F[IND + vdims.x] - v;
    }
    grad /= voxelSizes;
    GX[IND] = grad.x;
    GY[IND] = grad.y;
}

// Reflection boundary for gradinet and its adjoint divergence operator
void kernel FLOATvector_isotropicBackDivergence2D(global const float* restrict FX,
                                                  global const float* restrict FY,
                                                  global float* restrict DIV,
                                                  private int3 vdims,
                                                  private float3 voxelSizes)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int k = get_global_id(2);
    const int IND = VOXELINDEX(i, j, k, vdims);
    float VX = FX[IND];
    float VY = FY[IND];
    float DX, DY;
    if(i == 0)
    {
        // out = 0.0f;
        DX = -VX;
    } else
    {
        DX = FX[IND - 1] - VX;
    }
    if(j == 0)
    {
        //        out = 0.0f;
        DY = -VY;
    } else
    {
        DY = FY[IND - vdims.x] - VY;
    }
    DX = DX / voxelSizes.x;
    DY = DY / voxelSizes.y;
    DIV[IND] = DX + DY;
}

void kernel FLOATvector_isotropicBackDx_(global const float* restrict F,
                                         global float* restrict DX,
                                         private int3 vdims,
                                         private float3 voxelSizes)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int k = get_global_id(2);
    const int IND = VOXELINDEX(i, j, k, vdims);
    float v = F[IND];
    float out;
    if(i == 0)
    {
        // out = 0.0f;
        out = -v;
    } else if(i + 1 == vdims.x)
    {
        out = F[IND - 1];
    } else
    {
        out = F[IND - 1] - v;
    }
    DX[IND] = out / voxelSizes.x;
}

void kernel FLOATvector_isotropicBackDy_(global const float* restrict F,
                                         global float* restrict DY,
                                         private int3 vdims,
                                         private float3 voxelSizes)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int k = get_global_id(2);
    const int IND = VOXELINDEX(i, j, k, vdims);
    float v = F[IND];
    float out;
    if(j == 0)
    {
        //        out = 0.0f;
        out = -v;
    } else if(j + 1 == vdims.y)
    {
        out = F[IND - vdims.x];

    } else
    {
        out = F[IND - vdims.x] - v;
    }
    DY[IND] = out / voxelSizes.y;
}

void kernel FLOATvector_isotropicBackDz_(global const float* restrict F,
                                         global float* restrict DZ,
                                         private int3 vdims,
                                         private float3 voxelSizes)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int k = get_global_id(2);
    const int IND = VOXELINDEX(i, j, k, vdims);
    float v = F[IND];
    float out;
    if(k == 0)
    {
        // out = 0.0f;
        out = -v;
    } else if(k + 1 == vdims.z)
    {
        out = F[IND - vdims.x * vdims.y];
    } else
    {
        out = F[IND - vdims.x * vdims.y] - v;
    }
    DZ[IND] = out / voxelSizes.z;
}

float inline power(float x)
{
    return x;
    if(x < 0)
    {
        return -min(1.0f, -x);
    } else
    {
        return min(1.0f, x);
    }
}

void kernel FLOATvector_3DisotropicGradient(global const float* restrict F,
                                            global float* restrict GX,
                                            global float* restrict GY,
                                            global float* restrict GZ,
                                            private int3 vdims,
                                            private float3 voxelSizes)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int k = get_global_id(2);
    const int IND = VOXELINDEX(i, j, k, vdims);
    float3 grad;
    if(i + 1 == vdims.x || i == 0)
    {
        grad.x = 0.0f;
    } else
    {
        grad.x = F[IND + 1] - F[IND - 1];
    }
    if(j + 1 == vdims.y || j == 0)
    {
        grad.y = 0.0f;
    } else
    {
        grad.y = F[IND + vdims.x] - F[IND - vdims.x];
    }
    if(k + 1 == vdims.z || k == 0)
    {
        grad.z = 0.0f;
    } else
    {
        grad.z = F[IND + vdims.x * vdims.y] - F[IND - vdims.x * vdims.y];
    }
    grad /= voxelSizes;
    GX[IND] = power(grad.x);
    GY[IND] = power(grad.y);
    GZ[IND] = power(grad.z);
}

void kernel FLOATvector_2DisotropicGradient_(global const float* restrict F,
                                             global float* restrict GX,
                                             global float* restrict GY,
                                             private int3 vdims,
                                             private float3 voxelSizes)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int k = get_global_id(2);
    const int IND = VOXELINDEX(i, j, k, vdims);
    float3 grad;
    grad.z = 1.0f;
    if(i + 1 == vdims.x || i == 0)
    {
        grad.x = 0.0f;
    } else
    {
        grad.x = F[IND + 1] - F[IND - 1];
    }
    if(j + 1 == vdims.y || j == 0)
    {
        grad.y = 0.0f;
    } else
    {
        grad.y = F[IND + vdims.x] - F[IND - vdims.x];
    }
    grad /= voxelSizes;
    GX[IND] = power(grad.x);
    GY[IND] = power(grad.y);
}

void kernel FLOATvector_isotropicBackDx(global const float* restrict F,
                                        global float* restrict DX,
                                        private int3 vdims,
                                        private float3 voxelSizes)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int k = get_global_id(2);
    const int IND = VOXELINDEX(i, j, k, vdims);
    float out;
    if(i + 1 == vdims.x || i == 0)
    {
        out = 0.0f;
    } else
    {
        out = F[IND - 1] - F[IND + 1];
    }
    DX[IND] = power(out / voxelSizes.x);
}

void kernel FLOATvector_isotropicBackDy(global const float* restrict F,
                                        global float* restrict DY,
                                        private int3 vdims,
                                        private float3 voxelSizes)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int k = get_global_id(2);
    const int IND = VOXELINDEX(i, j, k, vdims);
    float out;
    if(j + 1 == vdims.y || j == 0)
    {
        out = 0.0f;
    } else
    {
        out = F[IND - vdims.x] - F[IND + vdims.x];
    }
    DY[IND] = power(out / voxelSizes.y);
}

void kernel FLOATvector_isotropicBackDz(global const float* restrict F,
                                        global float* restrict DZ,
                                        private int3 vdims,
                                        private float3 voxelSizes)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int k = get_global_id(2);
    const int IND = VOXELINDEX(i, j, k, vdims);
    float out;
    if(k + 1 == vdims.z || k == 0)
    {
        out = 0.0f;
    } else
    {
        out = F[IND - vdims.x * vdims.y] - F[IND + vdims.x * vdims.y];
    }
    DZ[IND] = power(out / voxelSizes.z);
}

#define ZERO555BOUNDARY()                                                                          \
    {                                                                                              \
        if(i < 2)                                                                                  \
        {                                                                                          \
            for(int lj = 0; lj < 5; lj++)                                                          \
            {                                                                                      \
                for(int lk = 0; lk < 5; lk++)                                                      \
                {                                                                                  \
                    cube[0][lj][lk] = 0.0f;                                                        \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
        if(i == 0)                                                                                 \
        {                                                                                          \
            for(int lj = 0; lj < 5; lj++)                                                          \
            {                                                                                      \
                for(int lk = 0; lk < 5; lk++)                                                      \
                {                                                                                  \
                    cube[1][lj][lk] = 0.0f;                                                        \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
        if(i + 3 > vdims.x)                                                                        \
        {                                                                                          \
            for(int lj = 0; lj < 5; lj++)                                                          \
            {                                                                                      \
                for(int lk = 0; lk < 5; lk++)                                                      \
                {                                                                                  \
                    cube[4][lj][lk] = 0.0f;                                                        \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
        if(i + 1 == vdims.x)                                                                       \
        {                                                                                          \
            for(int lj = 0; lj < 5; lj++)                                                          \
            {                                                                                      \
                for(int lk = 0; lk < 5; lk++)                                                      \
                {                                                                                  \
                    cube[3][lj][lk] = 0.0f;                                                        \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
        if(j < 2)                                                                                  \
        {                                                                                          \
            for(int li = 0; li < 5; li++)                                                          \
            {                                                                                      \
                for(int lk = 0; lk < 5; lk++)                                                      \
                {                                                                                  \
                    cube[li][0][lk] = 0.0f;                                                        \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
        if(j == 0)                                                                                 \
        {                                                                                          \
            for(int li = 0; li < 5; li++)                                                          \
            {                                                                                      \
                for(int lk = 0; lk < 5; lk++)                                                      \
                {                                                                                  \
                    cube[li][1][lk] = 0.0f;                                                        \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
        if(j + 3 > vdims.y)                                                                        \
        {                                                                                          \
            for(int li = 0; li < 5; li++)                                                          \
            {                                                                                      \
                for(int lk = 0; lk < 5; lk++)                                                      \
                {                                                                                  \
                    cube[li][4][lk] = 0.0f;                                                        \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
        if(j + 1 == vdims.y)                                                                       \
        {                                                                                          \
            for(int li = 0; li < 5; li++)                                                          \
            {                                                                                      \
                for(int lk = 0; lk < 5; lk++)                                                      \
                {                                                                                  \
                    cube[li][3][lk] = 0.0f;                                                        \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
        if(k < 2)                                                                                  \
        {                                                                                          \
            for(int li = 0; li < 5; li++)                                                          \
            {                                                                                      \
                for(int lj = 0; lj < 5; lj++)                                                      \
                {                                                                                  \
                    cube[li][lj][0] = 0.0f;                                                        \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
        if(k == 0)                                                                                 \
        {                                                                                          \
            for(int li = 0; li < 5; li++)                                                          \
            {                                                                                      \
                for(int lj = 0; lj < 5; lj++)                                                      \
                {                                                                                  \
                    cube[li][lj][1] = 0.0f;                                                        \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
        if(k + 3 > vdims.z)                                                                        \
        {                                                                                          \
            for(int li = 0; li < 5; li++)                                                          \
            {                                                                                      \
                for(int lj = 0; lj < 5; lj++)                                                      \
                {                                                                                  \
                    cube[li][lj][4] = 0.0f;                                                        \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
        if(k + 1 == vdims.z)                                                                       \
        {                                                                                          \
            for(int li = 0; li < 5; li++)                                                          \
            {                                                                                      \
                for(int lj = 0; lj < 5; lj++)                                                      \
                {                                                                                  \
                    cube[li][lj][3] = 0.0f;                                                        \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

#define REFLECTION555BOUNDARY()                                                                    \
    {                                                                                              \
        if(i < 2)                                                                                  \
        {                                                                                          \
            for(int lj = 0; lj < 5; lj++)                                                          \
            {                                                                                      \
                for(int lk = 0; lk < 5; lk++)                                                      \
                {                                                                                  \
                    cube[0][lj][lk] = cube[4][lj][lk];                                             \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
        if(i == 0)                                                                                 \
        {                                                                                          \
            for(int lj = 0; lj < 5; lj++)                                                          \
            {                                                                                      \
                for(int lk = 0; lk < 5; lk++)                                                      \
                {                                                                                  \
                    cube[1][lj][lk] = cube[3][lj][lk];                                             \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
        if(i + 3 > vdims.x)                                                                        \
        {                                                                                          \
            for(int lj = 0; lj < 5; lj++)                                                          \
            {                                                                                      \
                for(int lk = 0; lk < 5; lk++)                                                      \
                {                                                                                  \
                    cube[4][lj][lk] = cube[0][lj][lk];                                             \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
        if(i + 1 == vdims.x)                                                                       \
        {                                                                                          \
            for(int lj = 0; lj < 5; lj++)                                                          \
            {                                                                                      \
                for(int lk = 0; lk < 5; lk++)                                                      \
                {                                                                                  \
                    cube[3][lj][lk] = cube[1][lj][lk];                                             \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
        if(j < 2)                                                                                  \
        {                                                                                          \
            for(int li = 0; li < 5; li++)                                                          \
            {                                                                                      \
                for(int lk = 0; lk < 5; lk++)                                                      \
                {                                                                                  \
                    cube[li][0][lk] = cube[li][4][lk];                                             \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
        if(j == 0)                                                                                 \
        {                                                                                          \
            for(int li = 0; li < 5; li++)                                                          \
            {                                                                                      \
                for(int lk = 0; lk < 5; lk++)                                                      \
                {                                                                                  \
                    cube[li][1][lk] = cube[li][3][lk];                                             \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
        if(j + 3 > vdims.y)                                                                        \
        {                                                                                          \
            for(int li = 0; li < 5; li++)                                                          \
            {                                                                                      \
                for(int lk = 0; lk < 5; lk++)                                                      \
                {                                                                                  \
                    cube[li][4][lk] = cube[li][0][lk];                                             \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
        if(j + 1 == vdims.y)                                                                       \
        {                                                                                          \
            for(int li = 0; li < 5; li++)                                                          \
            {                                                                                      \
                for(int lk = 0; lk < 5; lk++)                                                      \
                {                                                                                  \
                    cube[li][3][lk] = cube[li][1][lk];                                             \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
        if(k < 2)                                                                                  \
        {                                                                                          \
            for(int li = 0; li < 5; li++)                                                          \
            {                                                                                      \
                for(int lj = 0; lj < 5; lj++)                                                      \
                {                                                                                  \
                    cube[li][lj][0] = cube[li][lj][4];                                             \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
        if(k == 0)                                                                                 \
        {                                                                                          \
            for(int li = 0; li < 5; li++)                                                          \
            {                                                                                      \
                for(int lj = 0; lj < 5; lj++)                                                      \
                {                                                                                  \
                    cube[li][lj][1] = cube[li][lj][3];                                             \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
        if(k + 3 > vdims.z)                                                                        \
        {                                                                                          \
            for(int li = 0; li < 5; li++)                                                          \
            {                                                                                      \
                for(int lj = 0; lj < 5; lj++)                                                      \
                {                                                                                  \
                    cube[li][lj][4] = cube[li][lj][0];                                             \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
        if(k + 1 == vdims.z)                                                                       \
        {                                                                                          \
            for(int li = 0; li < 5; li++)                                                          \
            {                                                                                      \
                for(int lj = 0; lj < 5; lj++)                                                      \
                {                                                                                  \
                    cube[li][lj][3] = cube[li][lj][1];                                             \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

#define ZERO55BOUNDARY()                                                                           \
    {                                                                                              \
        if(i < 2)                                                                                  \
        {                                                                                          \
            for(int lj = 0; lj < 5; lj++)                                                          \
            {                                                                                      \
                cube[0][lj] = 0.0f;                                                                \
            }                                                                                      \
        }                                                                                          \
        if(i == 0)                                                                                 \
        {                                                                                          \
            for(int lj = 0; lj < 5; lj++)                                                          \
            {                                                                                      \
                cube[1][lj] = 0.0f;                                                                \
            }                                                                                      \
        }                                                                                          \
        if(i + 3 > vdims.x)                                                                        \
        {                                                                                          \
            for(int lj = 0; lj < 5; lj++)                                                          \
            {                                                                                      \
                cube[4][lj] = 0.0f;                                                                \
            }                                                                                      \
        }                                                                                          \
        if(i + 1 == vdims.x)                                                                       \
        {                                                                                          \
            for(int lj = 0; lj < 5; lj++)                                                          \
            {                                                                                      \
                cube[3][lj] = 0.0f;                                                                \
            }                                                                                      \
        }                                                                                          \
        if(j < 2)                                                                                  \
        {                                                                                          \
            for(int li = 0; li < 5; li++)                                                          \
            {                                                                                      \
                cube[li][0] = 0.0f;                                                                \
            }                                                                                      \
        }                                                                                          \
        if(j == 0)                                                                                 \
        {                                                                                          \
            for(int li = 0; li < 5; li++)                                                          \
            {                                                                                      \
                cube[li][1] = 0.0f;                                                                \
            }                                                                                      \
        }                                                                                          \
        if(j + 3 > vdims.y)                                                                        \
        {                                                                                          \
            for(int li = 0; li < 5; li++)                                                          \
            {                                                                                      \
                cube[li][4] = 0.0f;                                                                \
            }                                                                                      \
        }                                                                                          \
        if(j + 1 == vdims.y)                                                                       \
        {                                                                                          \
            for(int li = 0; li < 5; li++)                                                          \
            {                                                                                      \
                cube[li][3] = 0.0f;                                                                \
            }                                                                                      \
        }                                                                                          \
    }

#define REFLECTION55BOUNDARY()                                                                     \
    {                                                                                              \
        if(i < 2)                                                                                  \
        {                                                                                          \
            for(int lj = 0; lj < 5; lj++)                                                          \
            {                                                                                      \
                cube[0][lj] = cube[4][lj];                                                         \
            }                                                                                      \
        }                                                                                          \
        if(i == 0)                                                                                 \
        {                                                                                          \
            for(int lj = 0; lj < 5; lj++)                                                          \
            {                                                                                      \
                cube[1][lj] = cube[3][lj];                                                         \
            }                                                                                      \
        }                                                                                          \
        if(i + 3 > vdims.x)                                                                        \
        {                                                                                          \
            for(int lj = 0; lj < 5; lj++)                                                          \
            {                                                                                      \
                cube[4][lj] = cube[0][lj];                                                         \
            }                                                                                      \
        }                                                                                          \
        if(i + 1 == vdims.x)                                                                       \
        {                                                                                          \
            for(int lj = 0; lj < 5; lj++)                                                          \
            {                                                                                      \
                cube[3][lj] = cube[1][lj];                                                         \
            }                                                                                      \
        }                                                                                          \
        if(j < 2)                                                                                  \
        {                                                                                          \
            for(int li = 0; li < 5; li++)                                                          \
            {                                                                                      \
                cube[li][0] = cube[li][4];                                                         \
            }                                                                                      \
        }                                                                                          \
        if(j == 0)                                                                                 \
        {                                                                                          \
            for(int li = 0; li < 5; li++)                                                          \
            {                                                                                      \
                cube[li][1] = cube[li][3];                                                         \
            }                                                                                      \
        }                                                                                          \
        if(j + 3 > vdims.y)                                                                        \
        {                                                                                          \
            for(int li = 0; li < 5; li++)                                                          \
            {                                                                                      \
                cube[li][4] = cube[li][0];                                                         \
            }                                                                                      \
        }                                                                                          \
        if(j + 1 == vdims.y)                                                                       \
        {                                                                                          \
            for(int li = 0; li < 5; li++)                                                          \
            {                                                                                      \
                cube[li][3] = cube[li][1];                                                         \
            }                                                                                      \
        }                                                                                          \
    }
void kernel FLOATvector_3DconvolutionGradientFarid5x5x5(global const float* restrict F,
                                                        global float* restrict GX,
                                                        global float* restrict GY,
                                                        global float* restrict GZ,
                                                        private int3 vdims,
                                                        private float3 voxelSizes,
                                                        private int reflectionBoundary)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int k = get_global_id(2);
    const int IND = VOXELINDEX(i, j, k, vdims);
    float cube[5][5][5]; // First fill this object where possible
    int DJ = vdims.x;
    int DK = vdims.x * vdims.y;
    int LIMIN = -min(i - 2, 0);
    int LJMIN = -min(j - 2, 0);
    int LKMIN = -min(k - 2, 0);
    // One behind the limit
    int ILIMIT = 5 + vdims.x - max(i + 3, vdims.x);
    int JLIMIT = 5 + vdims.y - max(j + 3, vdims.y);
    int KLIMIT = 5 + vdims.z - max(k + 3, vdims.z);
    int IRANGE = ILIMIT - LIMIN;
    int JRANGE = JLIMIT - LJMIN;
    int JSKIP = DJ - IRANGE;
    int KSKIP = DK - JRANGE * DJ;
    int index = VOXELINDEX(i + LIMIN - 2, j + LJMIN - 2, k + LKMIN - 2, vdims);
    for(int lk = LKMIN; lk < KLIMIT; lk++)
    {
        for(int lj = LJMIN; lj < JLIMIT; lj++)
        {
            for(int li = LIMIN; li < ILIMIT; li++)
            {
                cube[li][lj][lk] = F[index];
                index++;
            }
            index += JSKIP;
        }
        index += KSKIP;
    }
    if(reflectionBoundary != 0)
    {
        REFLECTION555BOUNDARY();
    } else
    {
        ZERO555BOUNDARY();
    }
    // fill with zeros
    float3 grad;
    grad.x = 0.00050790724f
            * (cube[3][0][0] + cube[3][0][4] + cube[3][4][0] + cube[3][4][4] - cube[1][0][0]
               - cube[1][0][4] - cube[1][4][0] - cube[1][4][4])
        + 0.0033603285f
            * (cube[3][0][1] + cube[3][0][3] + cube[3][1][0] + cube[3][1][4] + cube[3][3][0]
               + cube[3][3][4] + cube[3][4][1] + cube[3][4][3] - cube[1][0][1] - cube[1][0][3]
               - cube[1][1][0] - cube[1][1][4] - cube[1][3][0] - cube[1][3][4] - cube[1][4][1]
               - cube[1][4][3])
        + 0.038045622f
            * (cube[3][1][2] + cube[3][2][1] + cube[3][2][3] + cube[3][3][2] - cube[1][1][2]
               - cube[1][2][1] - cube[1][2][3] - cube[1][3][2])
        + 0.005750523f
            * (cube[3][0][2] + cube[3][2][0] + cube[3][2][4] + cube[3][4][2] - cube[1][0][2]
               - cube[1][2][0] - cube[1][2][4] - cube[1][4][2])
        + 0.022232028f
            * (cube[3][1][1] + cube[3][1][3] + cube[3][3][1] + cube[3][3][3] - cube[1][1][1]
               - cube[1][1][3] - cube[1][3][1] - cube[1][3][3])
        + 0.06510739f * (cube[3][2][2] - cube[1][2][2])
        + 0.00020119434f
            * (cube[4][0][0] + cube[4][0][4] + cube[4][4][0] + cube[4][4][4] - cube[0][0][0]
               - cube[0][0][4] - cube[0][4][0] - cube[0][4][4])
        + 0.0013311073f
            * (cube[4][0][1] + cube[4][0][3] + cube[4][1][0] + cube[4][1][4] + cube[4][3][0]
               + cube[4][3][4] + cube[4][4][1] + cube[4][4][3] - cube[0][0][1] - cube[0][0][3]
               - cube[0][1][0] - cube[0][1][4] - cube[0][3][0] - cube[0][3][4] - cube[0][4][1]
               - cube[0][4][3])
        + 0.0022779212f
            * (cube[4][0][2] + cube[4][2][0] + cube[4][2][4] + cube[4][4][2] - cube[0][0][2]
               - cube[0][2][0] - cube[0][2][4] - cube[0][4][2])
        + 0.008806644f
            * (cube[4][1][1] + cube[4][1][3] + cube[4][3][1] + cube[4][3][3] - cube[0][1][1]
               - cube[0][1][3] - cube[0][3][1] - cube[0][3][3])
        + 0.01507079f
            * (cube[4][1][2] + cube[4][2][1] + cube[4][2][3] + cube[4][3][2] - cube[0][1][2]
               - cube[0][2][1] - cube[0][2][3] - cube[0][3][2])
        + 0.025790613f * (cube[4][2][2] - cube[0][2][2]);

    grad.y = 0.00050790724f
            * (cube[0][3][0] + cube[0][3][4] + cube[4][3][0] + cube[4][3][4] - cube[0][1][0]
               - cube[0][1][4] - cube[4][1][0] - cube[4][1][4])
        + 0.0033603285f
            * (cube[0][3][1] + cube[0][3][3] + cube[1][3][0] + cube[1][3][4] + cube[3][3][0]
               + cube[3][3][4] + cube[4][3][1] + cube[4][3][3] - cube[0][1][1] - cube[0][1][3]
               - cube[1][1][0] - cube[1][1][4] - cube[3][1][0] - cube[3][1][4] - cube[4][1][1]
               - cube[4][1][3])
        + 0.0013311073f
            * (cube[0][4][1] + cube[0][4][3] + cube[1][4][0] + cube[1][4][4] + cube[3][4][0]
               + cube[3][4][4] + cube[4][4][1] + cube[4][4][3] - cube[0][0][1] - cube[0][0][3]
               - cube[1][0][0] - cube[1][0][4] - cube[3][0][0] - cube[3][0][4] - cube[4][0][1]
               - cube[4][0][3])
        + 0.005750523f
            * (cube[0][3][2] + cube[2][3][0] + cube[2][3][4] + cube[4][3][2] - cube[0][1][2]
               - cube[2][1][0] - cube[2][1][4] - cube[4][1][2])
        + 0.00020119434f
            * (cube[0][4][0] + cube[0][4][4] + cube[4][4][0] + cube[4][4][4] - cube[0][0][0]
               - cube[0][0][4] - cube[4][0][0] - cube[4][0][4])
        + 0.0022779212f
            * (cube[0][4][2] + cube[2][4][0] + cube[2][4][4] + cube[4][4][2] - cube[0][0][2]
               - cube[2][0][0] - cube[2][0][4] - cube[4][0][2])
        + 0.022232028f
            * (cube[1][3][1] + cube[1][3][3] + cube[3][3][1] + cube[3][3][3] - cube[1][1][1]
               - cube[1][1][3] - cube[3][1][1] - cube[3][1][3])
        + 0.038045622f
            * (cube[1][3][2] + cube[2][3][1] + cube[2][3][3] + cube[3][3][2] - cube[1][1][2]
               - cube[2][1][1] - cube[2][1][3] - cube[3][1][2])
        + 0.008806644f
            * (cube[1][4][1] + cube[1][4][3] + cube[3][4][1] + cube[3][4][3] - cube[1][0][1]
               - cube[1][0][3] - cube[3][0][1] - cube[3][0][3])
        + 0.01507079f
            * (cube[1][4][2] + cube[2][4][1] + cube[2][4][3] + cube[3][4][2] - cube[1][0][2]
               - cube[2][0][1] - cube[2][0][3] - cube[3][0][2])
        + 0.06510739f * (cube[2][3][2] - cube[2][1][2])
        + 0.025790613f * (cube[2][4][2] - cube[2][0][2]);

    grad.z = 0.00050790724f
            * (cube[0][0][3] + cube[0][4][3] + cube[4][0][3] + cube[4][4][3] - cube[0][0][1]
               - cube[0][4][1] - cube[4][0][1] - cube[4][4][1])
        + 0.00020119434f
            * (cube[0][0][4] + cube[0][4][4] + cube[4][0][4] + cube[4][4][4] - cube[0][0][0]
               - cube[0][4][0] - cube[4][0][0] - cube[4][4][0])
        + 0.005750523f
            * (cube[0][2][3] + cube[2][0][3] + cube[2][4][3] + cube[4][2][3] - cube[0][2][1]
               - cube[2][0][1] - cube[2][4][1] - cube[4][2][1])
        + 0.0033603285f
            * (cube[0][1][3] + cube[0][3][3] + cube[1][0][3] + cube[1][4][3] + cube[3][0][3]
               + cube[3][4][3] + cube[4][1][3] + cube[4][3][3] - cube[0][1][1] - cube[0][3][1]
               - cube[1][0][1] - cube[1][4][1] - cube[3][0][1] - cube[3][4][1] - cube[4][1][1]
               - cube[4][3][1])
        + 0.0013311073f
            * (cube[0][1][4] + cube[0][3][4] + cube[1][0][4] + cube[1][4][4] + cube[3][0][4]
               + cube[3][4][4] + cube[4][1][4] + cube[4][3][4] - cube[0][1][0] - cube[0][3][0]
               - cube[1][0][0] - cube[1][4][0] - cube[3][0][0] - cube[3][4][0] - cube[4][1][0]
               - cube[4][3][0])
        + 0.0022779212f
            * (cube[0][2][4] + cube[2][0][4] + cube[2][4][4] + cube[4][2][4] - cube[0][2][0]
               - cube[2][0][0] - cube[2][4][0] - cube[4][2][0])
        + 0.022232028f
            * (cube[1][1][3] + cube[1][3][3] + cube[3][1][3] + cube[3][3][3] - cube[1][1][1]
               - cube[1][3][1] - cube[3][1][1] - cube[3][3][1])
        + 0.008806644f
            * (cube[1][1][4] + cube[1][3][4] + cube[3][1][4] + cube[3][3][4] - cube[1][1][0]
               - cube[1][3][0] - cube[3][1][0] - cube[3][3][0])
        + 0.038045622f
            * (cube[1][2][3] + cube[2][1][3] + cube[2][3][3] + cube[3][2][3] - cube[1][2][1]
               - cube[2][1][1] - cube[2][3][1] - cube[3][2][1])
        + 0.01507079f
            * (cube[1][2][4] + cube[2][1][4] + cube[2][3][4] + cube[3][2][4] - cube[1][2][0]
               - cube[2][1][0] - cube[2][3][0] - cube[3][2][0])
        + 0.06510739f * (cube[2][2][3] - cube[2][2][1])
        + 0.025790613f * (cube[2][2][4] - cube[2][2][0]);
    grad /= voxelSizes;
    GX[IND] = grad.x;
    GY[IND] = grad.y;
    GZ[IND] = grad.z;
}

void kernel FLOATvector_2DconvolutionGradientFarid5x5(global const float* restrict F,
                                                      global float* restrict GX,
                                                      global float* restrict GY,
                                                      private int3 vdims,
                                                      private float3 voxelSizes,
                                                      private int reflectionBoundary)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int k = get_global_id(2);
    const int IND = VOXELINDEX(i, j, k, vdims);
    float cube[5][5]; // First fill this object where possible
    int DJ = vdims.x;
    int DK = vdims.x * vdims.y;
    int LIMIN = -min(i - 2, 0);
    int LJMIN = -min(j - 2, 0);
    int LKMIN = -min(k - 2, 0);
    // One behind the limit
    int ILIMIT = 5 + vdims.x - max(i + 3, vdims.x);
    int JLIMIT = 5 + vdims.y - max(j + 3, vdims.y);
    int KLIMIT = 5 + vdims.z - max(k + 3, vdims.z);
    int IRANGE = ILIMIT - LIMIN;
    int JRANGE = JLIMIT - LJMIN;
    int JSKIP = DJ - IRANGE;
    int index = VOXELINDEX(i + LIMIN - 2, j + LJMIN - 2, k, vdims);
    for(int lj = LJMIN; lj < JLIMIT; lj++)
    {
        for(int li = LIMIN; li < ILIMIT; li++)
        {
            cube[li][lj] = F[index];
            index++;
        }
        index += JSKIP;
    }
    if(reflectionBoundary != 0)
    {
        REFLECTION55BOUNDARY();
    } else
    {
        ZERO55BOUNDARY();
    }
    // fill with zeros
    float2 grad;
    grad.x = 0.013486994f * (cube[3][0] + cube[3][4] - cube[1][0] - cube[1][4])
        + 0.08923033f * (cube[3][1] + cube[3][3] - cube[1][1] - cube[1][3])
        + 0.035346292f * (cube[4][1] + cube[4][3] - cube[0][1] - cube[0][3])
        + 0.15269968f * (cube[3][2] - cube[1][2])
        + 0.0053425245f * (cube[4][0] + cube[4][4] - cube[0][0] - cube[0][4])
        + 0.060488038f * (cube[4][2] - cube[0][2]);

    grad.y = 0.013486994f * (cube[0][3] + cube[4][3] - cube[0][1] - cube[4][1])
        + 0.0053425245f * (cube[0][4] + cube[4][4] - cube[0][0] - cube[4][0])
        + 0.15269968f * (cube[2][3] - cube[2][1])
        + 0.08923033f * (cube[1][3] + cube[3][3] - cube[1][1] - cube[3][1])
        + 0.035346292f * (cube[1][4] + cube[3][4] - cube[1][0] - cube[3][0])
        + 0.060488038f * (cube[2][4] - cube[2][0]);
    grad /= voxelSizes.s01;
    GX[IND] = grad.x;
    GY[IND] = grad.y;
}

/**
 * @brief 3D Laplace as described by 27 point strecil see O’Reilly, H.; Beck, Jeffrey M. (2006). "A
 * Family of Large-Stencil Discrete Laplacian Approximations in Three Dimensions". International
 * Journal For Numerical Methods in Engineering: 1–16. and
 * https://en.wikipedia.org/wiki/Discrete_Laplace_operator
 *
 * @param F
 * @param GX
 * @param GY
 * @param GZ
 * @param vdims
 * @param voxelSizes
 *
 * @return
 */
void kernel FLOATvector_3DconvolutionLaplaceZeroBoundary(global const float* restrict F,
                                                         global float* restrict L,
                                                         private int3 vdims,
                                                         private float3 voxelSizes)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int k = get_global_id(2); /*
     if(k == 3 && i == 5 && j == 55)
     {
         printf("Calling with i=%d, j=%d, k=%d", i, j, k);
     }*/
    const int IND = VOXELINDEX(i, j, k, vdims);
    float cube[3][3][3]; // First fill this object where possible
    int DJ = vdims.x;
    int DK = vdims.x * vdims.y;
    int LIMIN = -min(i - 1, 0);
    int LJMIN = -min(j - 1, 0);
    int LKMIN = -min(k - 1, 0);
    // One behind the limit
    int ILIMIT = 3 + vdims.x - max(i + 2, vdims.x);
    int JLIMIT = 3 + vdims.y - max(j + 2, vdims.y);
    int KLIMIT = 3 + vdims.z - max(k + 2, vdims.z);
    int IRANGE = ILIMIT - LIMIN;
    int JRANGE = JLIMIT - LJMIN;
    int JSKIP = DJ - IRANGE;
    int KSKIP = DK - JRANGE * DJ;
    int index = VOXELINDEX(i + LIMIN - 1, j + LJMIN - 1, k + LKMIN - 1, vdims);
    /*
        if(k == 30 && j == 511)
        {
            printf("i=%d j=%d k=%d index=%d IND=%d LIMIN=%d, LJMIN=%d, LKMIN=%d start=(%d, %d, %d) "
                   "LIMITS=(%d, "
                   "%d, %d)",
                   i, j, k, index, IND, LIMIN, LJMIN, LKMIN, i - 1 + LIMIN, j - 1 + LJMIN,
                   k - 1 + LKMIN, ILIMIT, JLIMIT, KLIMIT);
        }
    */
    for(int lk = LKMIN; lk < KLIMIT; lk++)
    {
        for(int lj = LJMIN; lj < JLIMIT; lj++)
        {
            for(int li = LIMIN; li < ILIMIT; li++)
            {

                int index_computed = VOXELINDEX(i - 1 + li, j - 1 + lj, k - 1 + lk, vdims);
                if(index_computed != index)
                {
                    printf("Problem with index alignment index=%d, index_computed=%d", index,
                           index_computed);
                }

                cube[li][lj][lk] = F[index];
                /*
                        if(k == 3 && i == 5 && j == 55)
                        {
                            printf("I in [%d, %d] J in [%d, %d] K in [%d, %d] index=%d vdims.x=%d
                   vdims.y=%d vdims.z=%d" "F[index]=%f voxelSizes.x=%f", LIMIN, ILIMIT, LJMIN,
                   JLIMIT, LKMIN, KLIMIT, index, vdims.x, vdims.y, vdims.z, cube[li][lj][lk],
                   voxelSizes.x);
                        }*/
                index++;
            }
            index += JSKIP;
        }
        index += KSKIP;
    }
    // fill with zeros
    if(i == 0)
    {
        // i reflection li=0 lj=. lk=. equals li=2 lj=. lk=.
        // if 0, lj, lk invalid so is 2,lj,lk
        for(int lj = 0; lj < 3; lj++)
        {
            for(int lk = 0; lk < 3; lk++)
            {
                cube[0][lj][lk] = 0.0f;
            }
        }
    } else if(i + 1 == vdims.x)
    {
        // i reflection li=2 lj=. lk=. equals li=0 lj=. lk=.
        // if 2, lj, lk invalid so is 0,lj,lk
        for(int lj = 0; lj < 3; lj++)
        {
            for(int lk = 0; lk < 3; lk++)
            {
                cube[2][lj][lk] = 0.0f;
            }
        }
    }
    if(j == 0)
    {
        // j reflection li=. lj=0 lk=. equals li=. lj=2 lk=.
        // if li, 0, lk invalid so is li,0,lk
        for(int li = 0; li < 3; li++)
        {
            for(int lk = 0; lk < 3; lk++)
            {
                cube[li][0][lk] = 0.0f;
            }
        }
    } else if(j + 1 == vdims.y)
    {
        // j reflection li=. lj=2 lk=. equals li=. lj=0 lk=.
        // if li, 0, lk invalid so is li,0,lk
        for(int li = 0; li < 3; li++)
        {
            for(int lk = 0; lk < 3; lk++)
            {
                cube[li][2][lk] = 0.0f;
            }
        }
    }
    if(k == 0)
    {
        // k reflection li=. lj=. lk=0 equals li=. lj=. lk=2
        // if li, lj, 2 invalid so is li,lj,0
        for(int li = 0; li < 3; li++)
        {
            for(int lj = 0; lj < 3; lj++)
            {
                cube[li][lj][0] = 0.0f;
            }
        }
    } else if(k + 1 == vdims.z)
    {
        // k reflection li=. lj=. lk=2 equals li=. lj=. lk=0
        // if li, lj, 2 invalid so is li,lj,0
        for(int li = 0; li < 3; li++)
        {
            for(int lj = 0; lj < 3; lj++)
            {
                cube[li][lj][2] = 0.0f;
            }
        }
    }
    float laplace =

        cube[0][0][0] * 2.0f + cube[1][0][0] * 3.0f + cube[2][0][0] * 2.0f

        + cube[0][1][0] * 3.0f + cube[1][1][0] * 6.0f + cube[2][1][0] * 3.0f

        + cube[0][2][0] * 2.0f + cube[1][2][0] * 3.0f + cube[2][2][0] * 2.0f

        + cube[0][0][1] * 3.0f + cube[1][0][1] * 6.0f + cube[2][0][1] * 3.0f

        + cube[0][1][1] * 6.0f - cube[1][1][1] * 88.0f + cube[2][1][1] * 6.0f

        + cube[0][2][1] * 3.0f + cube[1][2][1] * 6.0f + cube[2][2][1] * 3.0f

        + cube[0][0][2] * 2.0f + cube[1][0][2] * 3.0f + cube[2][0][2] * 2.0f

        + cube[0][1][2] * 3.0f + cube[1][1][2] * 6.0f + cube[2][1][2] * 3.0f

        + cube[0][2][2] * 2.0f + cube[1][2][2] * 3.0f + cube[2][2][2] * 2.0f;

    laplace /= 26.0f;
    L[IND] = laplace;
}
//==============================END convolution.cl=====================================
