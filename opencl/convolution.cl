//==============================convolution.cl=====================================
// see https://www.evl.uic.edu/kreda/gpu/image-convolution/
#define VOXELINDEX(i, j, k, vdims) i + (j + (k)*vdims.y) * vdims.x

void kernel FLOATvector_2Dconvolution3x3(global const float* restrict A,
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
    grad /= voxelSizes;
    GX[IND] = grad.x;
    GY[IND] = grad.y;
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
    float v = F[IND];
    float out;
    if(i == 0)
    {
        out = 0.0f;
    } else
    {
        out = F[IND - 1] - v;
    }
    DX[IND] = out / voxelSizes.x;
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
    float v = F[IND];
    float out;
    if(i == 0)
    {
        out = 0.0f;
    } else
    {
        out = F[IND - vdims.x] - v;
    }
    DY[IND] = out / voxelSizes.y;
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
    float v = F[IND];
    float out;
    if(i == 0)
    {
        out = 0.0f;
    } else
    {
        out = F[IND - vdims.x * vdims.y] - v;
    }
    DZ[IND] = out / voxelSizes.z;
}

void kernel FLOATvector_3DconvolutionGradientFarid5x5x5ZeroBoundary(global const float* restrict F,
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
    // fill with zeros
    if(i < 2)
    {
        // i reflection li=0 lj=. lk=. equals li=2 lj=. lk=.
        // if 0, lj, lk invalid so is 2,lj,lk
        for(int lj = 0; lj < 5; lj++)
        {
            for(int lk = 0; lk < 5; lk++)
            {
                cube[0][lj][lk] = 0.0f;
            }
        }
    }
    if(i == 0)
    {
        // i reflection li=0 lj=. lk=. equals li=2 lj=. lk=.
        // if 0, lj, lk invalid so is 2,lj,lk
        for(int lj = 0; lj < 5; lj++)
        {
            for(int lk = 0; lk < 5; lk++)
            {
                cube[1][lj][lk] = 0.0f;
            }
        }
    }
    if(i + 3 > vdims.x)
    {
        // i reflection li=2 lj=. lk=. equals li=0 lj=. lk=.
        // if 2, lj, lk invalid so is 0,lj,lk
        for(int lj = 0; lj < 5; lj++)
        {
            for(int lk = 0; lk < 5; lk++)
            {
                cube[4][lj][lk] = 0.0f;
            }
        }
    }
    if(i + 1 == vdims.x)
    {
        // i reflection li=2 lj=. lk=. equals li=0 lj=. lk=.
        // if 2, lj, lk invalid so is 0,lj,lk
        for(int lj = 0; lj < 5; lj++)
        {
            for(int lk = 0; lk < 5; lk++)
            {
                cube[3][lj][lk] = 0.0f;
            }
        }
    }
    if(j < 2)
    {
        // j reflection li=. lj=0 lk=. equals li=. lj=2 lk=.
        // if li, 0, lk invalid so is li,0,lk
        for(int li = 0; li < 5; li++)
        {
            for(int lk = 0; lk < 5; lk++)
            {
                cube[li][0][lk] = 0.0f;
            }
        }
    }
    if(j == 0)
    {
        // j reflection li=. lj=0 lk=. equals li=. lj=2 lk=.
        // if li, 0, lk invalid so is li,0,lk
        for(int li = 0; li < 5; li++)
        {
            for(int lk = 0; lk < 5; lk++)
            {
                cube[li][1][lk] = 0.0f;
            }
        }
    }
    if(j + 3 > vdims.y)
    {
        // j reflection li=. lj=2 lk=. equals li=. lj=0 lk=.
        // if li, 0, lk invalid so is li,0,lk
        for(int li = 0; li < 5; li++)
        {
            for(int lk = 0; lk < 5; lk++)
            {
                cube[li][4][lk] = 0.0f;
            }
        }
    }
    if(j + 1 == vdims.y)
    {
        // j reflection li=. lj=2 lk=. equals li=. lj=0 lk=.
        // if li, 0, lk invalid so is li,0,lk
        for(int li = 0; li < 5; li++)
        {
            for(int lk = 0; lk < 5; lk++)
            {
                cube[li][3][lk] = 0.0f;
            }
        }
    }
    if(k < 2)
    {
        // k reflection li=. lj=. lk=0 equals li=. lj=. lk=2
        // if li, lj, 2 invalid so is li,lj,0
        for(int li = 0; li < 5; li++)
        {
            for(int lj = 0; lj < 5; lj++)
            {
                cube[li][lj][0] = 0.0f;
            }
        }
    }
    if(k == 0)
    {
        // k reflection li=. lj=. lk=0 equals li=. lj=. lk=2
        // if li, lj, 2 invalid so is li,lj,0
        for(int li = 0; li < 5; li++)
        {
            for(int lj = 0; lj < 5; lj++)
            {
                cube[li][lj][1] = 0.0f;
            }
        }
    }

    if(k + 3 > vdims.z)
    {
        // k reflection li=. lj=. lk=2 equals li=. lj=. lk=0
        // if li, lj, 2 invalid so is li,lj,0
        for(int li = 0; li < 5; li++)
        {
            for(int lj = 0; lj < 5; lj++)
            {
                cube[li][lj][4] = 0.0f;
            }
        }
    }
    if(k + 1 == vdims.z)
    {
        // k reflection li=. lj=. lk=2 equals li=. lj=. lk=0
        // if li, lj, 2 invalid so is li,lj,0
        for(int li = 0; li < 5; li++)
        {
            for(int lj = 0; lj < 5; lj++)
            {
                cube[li][lj][3] = 0.0f;
            }
        }
    }
    float3 grad;
    grad.x = 0.00050790724f
            * (cube[3][0][0] + cube[3][0][4] + cube[3][4][0] + cube[3][4][4] - cube[1][0][0]
               - cube[1][0][4] - cube[1][4][0] - cube[1][4][4])
        + 0.0033603285f
            * (cube[3][0][1] + cube[3][0][3] + cube[3][1][0] + cube[3][1][4] + cube[3][3][0]
               + cube[3][3][4] + cube[3][4][1] + cube[3][4][3] - cube[1][0][1] - cube[1][0][3]
               - cube[1][1][0] - cube[1][1][4] - cube[1][3][0] - cube[1][3][4] - cube[1][4][1]
               - cube[1][4][3])
        + 0.0022779212f
            * (cube[4][0][2] + cube[4][2][0] + cube[4][2][4] + cube[4][4][2] - cube[0][0][2]
               - cube[0][2][0] - cube[0][2][4] - cube[0][4][2])
        + 0.0013311073f
            * (cube[4][0][1] + cube[4][0][3] + cube[4][1][0] + cube[4][1][4] + cube[4][3][0]
               + cube[4][3][4] + cube[4][4][1] + cube[4][4][3] - cube[0][0][1] - cube[0][0][3]
               - cube[0][1][0] - cube[0][1][4] - cube[0][3][0] - cube[0][3][4] - cube[0][4][1]
               - cube[0][4][3])
        + 0.008806644f
            * (cube[4][1][1] + cube[4][1][3] + cube[4][3][1] + cube[4][3][3] - cube[0][1][1]
               - cube[0][1][3] - cube[0][3][1] - cube[0][3][3])
        + 0.025790613f * (cube[4][2][2] - cube[0][2][2])
        + 0.06510739f * (cube[3][2][2] - cube[1][2][2])
        + 0.038045622f
            * (cube[3][1][2] + cube[3][2][1] + cube[3][2][3] + cube[3][3][2] - cube[1][1][2]
               - cube[1][2][1] - cube[1][2][3] - cube[1][3][2])
        + 0.022232028f
            * (cube[3][1][1] + cube[3][1][3] + cube[3][3][1] + cube[3][3][3] - cube[1][1][1]
               - cube[1][1][3] - cube[1][3][1] - cube[1][3][3])
        + 0.005750523f
            * (cube[3][0][2] + cube[3][2][0] + cube[3][2][4] + cube[3][4][2] - cube[1][0][2]
               - cube[1][2][0] - cube[1][2][4] - cube[1][4][2])
        + 0.00020119434f
            * (cube[4][0][0] + cube[4][0][4] + cube[4][4][0] + cube[4][4][4] - cube[0][0][0]
               - cube[0][0][4] - cube[0][4][0] - cube[0][4][4])
        + 0.01507079f
            * (cube[4][1][2] + cube[4][2][1] + cube[4][2][3] + cube[4][3][2] - cube[0][1][2]
               - cube[0][2][1] - cube[0][2][3] - cube[0][3][2]);
    grad.y = 0.00050790724f
            * (cube[0][3][0] + cube[0][3][4] + cube[4][3][0] + cube[4][3][4] - cube[0][1][0]
               - cube[0][1][4] - cube[4][1][0] - cube[4][1][4])
        + 0.0033603285f
            * (cube[0][3][1] + cube[0][3][3] + cube[1][3][0] + cube[1][3][4] + cube[3][3][0]
               + cube[3][3][4] + cube[4][3][1] + cube[4][3][3] - cube[0][1][1] - cube[0][1][3]
               - cube[1][1][0] - cube[1][1][4] - cube[3][1][0] - cube[3][1][4] - cube[4][1][1]
               - cube[4][1][3])
        + 0.022232028f
            * (cube[1][3][1] + cube[1][3][3] + cube[3][3][1] + cube[3][3][3] - cube[1][1][1]
               - cube[1][1][3] - cube[3][1][1] - cube[3][1][3])
        + 0.0013311073f
            * (cube[0][4][1] + cube[0][4][3] + cube[1][4][0] + cube[1][4][4] + cube[3][4][0]
               + cube[3][4][4] + cube[4][4][1] + cube[4][4][3] - cube[0][0][1] - cube[0][0][3]
               - cube[1][0][0] - cube[1][0][4] - cube[3][0][0] - cube[3][0][4] - cube[4][0][1]
               - cube[4][0][3])
        + 0.008806644f
            * (cube[1][4][1] + cube[1][4][3] + cube[3][4][1] + cube[3][4][3] - cube[1][0][1]
               - cube[1][0][3] - cube[3][0][1] - cube[3][0][3])
        + 0.025790613f * (cube[2][4][2] - cube[2][0][2])
        + 0.0022779212f
            * (cube[0][4][2] + cube[2][4][0] + cube[2][4][4] + cube[4][4][2] - cube[0][0][2]
               - cube[2][0][0] - cube[2][0][4] - cube[4][0][2])
        + 0.00020119434f
            * (cube[0][4][0] + cube[0][4][4] + cube[4][4][0] + cube[4][4][4] - cube[0][0][0]
               - cube[0][0][4] - cube[4][0][0] - cube[4][0][4])
        + 0.005750523f
            * (cube[0][3][2] + cube[2][3][0] + cube[2][3][4] + cube[4][3][2] - cube[0][1][2]
               - cube[2][1][0] - cube[2][1][4] - cube[4][1][2])
        + 0.038045622f
            * (cube[1][3][2] + cube[2][3][1] + cube[2][3][3] + cube[3][3][2] - cube[1][1][2]
               - cube[2][1][1] - cube[2][1][3] - cube[3][1][2])
        + 0.06510739f * (cube[2][3][2] - cube[2][1][2])
        + 0.01507079f
            * (cube[1][4][2] + cube[2][4][1] + cube[2][4][3] + cube[3][4][2] - cube[1][0][2]
               - cube[2][0][1] - cube[2][0][3] - cube[3][0][2]);
    grad.z = 0.00050790724f
            * (cube[0][0][3] + cube[0][4][3] + cube[4][0][3] + cube[4][4][3] - cube[0][0][1]
               - cube[0][4][1] - cube[4][0][1] - cube[4][4][1])
        + 0.00020119434f
            * (cube[0][0][4] + cube[0][4][4] + cube[4][0][4] + cube[4][4][4] - cube[0][0][0]
               - cube[0][4][0] - cube[4][0][0] - cube[4][4][0])
        + 0.022232028f
            * (cube[1][1][3] + cube[1][3][3] + cube[3][1][3] + cube[3][3][3] - cube[1][1][1]
               - cube[1][3][1] - cube[3][1][1] - cube[3][3][1])
        + 0.0013311073f
            * (cube[0][1][4] + cube[0][3][4] + cube[1][0][4] + cube[1][4][4] + cube[3][0][4]
               + cube[3][4][4] + cube[4][1][4] + cube[4][3][4] - cube[0][1][0] - cube[0][3][0]
               - cube[1][0][0] - cube[1][4][0] - cube[3][0][0] - cube[3][4][0] - cube[4][1][0]
               - cube[4][3][0])
        + 0.008806644f
            * (cube[1][1][4] + cube[1][3][4] + cube[3][1][4] + cube[3][3][4] - cube[1][1][0]
               - cube[1][3][0] - cube[3][1][0] - cube[3][3][0])
        + 0.025790613f * (cube[2][2][4] - cube[2][2][0])
        + 0.005750523f
            * (cube[0][2][3] + cube[2][0][3] + cube[2][4][3] + cube[4][2][3] - cube[0][2][1]
               - cube[2][0][1] - cube[2][4][1] - cube[4][2][1])
        + 0.0022779212f
            * (cube[0][2][4] + cube[2][0][4] + cube[2][4][4] + cube[4][2][4] - cube[0][2][0]
               - cube[2][0][0] - cube[2][4][0] - cube[4][2][0])
        + 0.0033603285f
            * (cube[0][1][3] + cube[0][3][3] + cube[1][0][3] + cube[1][4][3] + cube[3][0][3]
               + cube[3][4][3] + cube[4][1][3] + cube[4][3][3] - cube[0][1][1] - cube[0][3][1]
               - cube[1][0][1] - cube[1][4][1] - cube[3][0][1] - cube[3][4][1] - cube[4][1][1]
               - cube[4][3][1])
        + 0.038045622f
            * (cube[1][2][3] + cube[2][1][3] + cube[2][3][3] + cube[3][2][3] - cube[1][2][1]
               - cube[2][1][1] - cube[2][3][1] - cube[3][2][1])
        + 0.06510739f * (cube[2][2][3] - cube[2][2][1])
        + 0.01507079f
            * (cube[1][2][4] + cube[2][1][4] + cube[2][3][4] + cube[3][2][4] - cube[1][2][0]
               - cube[2][1][0] - cube[2][3][0] - cube[3][2][0]);
    grad /= voxelSizes;
    GX[IND] = grad.x;
    GY[IND] = grad.y;
    GZ[IND] = grad.z;
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
