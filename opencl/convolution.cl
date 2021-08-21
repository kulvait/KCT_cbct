//==============================convolution.cl=====================================
// see https://www.evl.uic.edu/kreda/gpu/image-convolution/
#define VOXELINDEX(i, j, k, vdims) i + (j + k * vdims.y) * vdims.x

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
    float* convolutionKernel = &_convolutionKernel;
    // Manipulation to fix problems on boundaries
    int IMINDIFF = -min(i - convolutionKernelRadius, 0);
    int IMAXDIFF = max(i + convolutionKernelRadius, vdims.x) - vdims.x;
    int JMINDIFF = -min(j - convolutionKernelRadius, 0);
    int JMAXDIFF = max(j + convolutionKernelRadius, vdims.y) - vdims.y;
    int JLIMIT = convolutionKernelSize - JMAXDIFF;
    int ILIMIT = convolutionKernelSize - IMAXDIFF;
    int IRANGE = ILIMIT - IMINDIFF;
    int yskip = vdims.x - IRANGE;
    int ylocalskip = convolutionKernelSize - IRANGE;
    int localIndex = JMINDIFF * convolutionKernelSize + IMINDIFF;
    int index = VOXELINDEX(i - 1 + IMINDIFF, j - 1 + JMINDIFF, k, vdims);
    for(int j_loc = JMINDIFF; j_loc < JLIMIT; j_loc++)
    {
        for(int i_loc = IMINDIFF; i_loc < ILIMIT; i_loc++)
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
//==============================END convolution.cl=====================================
