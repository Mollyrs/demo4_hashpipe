/** 
 */

#include "demo4_gpu_thread.h"

extern cufftHandle g_stPlan;
extern float* g_pf4FFTIn_d;
extern float* g_pf4FFTOut_d;

extern cufftHandle g_stPlan2;
extern float* g_pf4FFTOut2_d;



__global__ void CopyDataForFFT(char *pc4Data,
                               float *pf4FFTIn)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    pf4FFTIn[i] = (float) pc4Data[i];
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

    /* execute plan */
    
    iCUFFTRet = cufftExecR2C(g_stPlan2,
        (cufftReal*) g_pf4FFTIn_d,
        (cufftComplex*) g_pf4FFTOut2_d);
    if (iCUFFTRet != CUFFT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR! FFT for polarisation X failed!\n");
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}

__global__ void Accumulate(float2 *pf4FFTOut, //2 batch
                           float2 *pf4FFTOut2, //1 batch
                           float *pf4SumStokes)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    float2 f4FFTOut = pf4FFTOut2[i];
    float f4SumStokes = pf4SumStokes[i];

    f4SumStokes += sqrtf((f4FFTOut.x * f4FFTOut.x) + (f4FFTOut.y * f4FFTOut.y));

    pf4SumStokes[i] = f4SumStokes;

    return;
}

