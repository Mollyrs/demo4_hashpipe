/** 
 */

#include "demo4_gpu_thread.h"

extern cufftHandle g_stPlan;
extern float* g_pf4FFTIn_d;
extern float* g_pf4FFTOut_d;



__global__ void CopyDataForFFT(char4 *pc4Data,
                               float *pf4FFTIn)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = i*4;
    pf4FFTIn[j] = (float) pc4Data[i].x;
    pf4FFTIn[j+1] = (float) pc4Data[i].y;
    pf4FFTIn[j+2] = (float) pc4Data[i].z;
    pf4FFTIn[j+3] = (float) pc4Data[i].w;

    return;
}

/* function that performs the FFT - not a kernel, just a wrapper to an
   API call */
int DoFFT()
{
    cufftResult iCUFFTRet = CUFFT_SUCCESS;

    /* execute plan */
    iCUFFTRet = cufftExecR2C(g_stPlan,
                             (cufftReal*) g_pf4FFTIn_d,
                             (cufftComplex*) g_pf4FFTOut_d);
    if (iCUFFTRet != CUFFT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR! FFT for polarisation X failed!\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

__global__ void Accumulate(float2 *pf4FFTOut,
                           float4 *pf4SumStokes)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    float2 f4FFTOut = pf4FFTOut[i];
    float4 f4SumStokes = pf4SumStokes[i];

    /* Re(X)^2 + Im(X)^2 */
    f4SumStokes.x += (f4FFTOut.x * f4FFTOut.x) + (f4FFTOut.y * f4FFTOut.y);
    /* Re(Y)^2 + Im(Y)^2 */
    f4SumStokes.y += sqrtf((f4FFTOut.x * f4FFTOut.x) + (f4FFTOut.y * f4FFTOut.y));
    /* Re(XY*) */
    f4SumStokes.z += (f4FFTOut.x * f4FFTOut.x);
    /* Im(XY*) */
    f4SumStokes.w += (f4FFTOut.y * f4FFTOut.y);

    pf4SumStokes[i] = f4SumStokes;

    return;
}

