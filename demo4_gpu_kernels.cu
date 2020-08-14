/** 
 */

#include "demo4_gpu_thread.h"

extern cufftHandle g_stPlan1;
extern float* g_pf4FFTIn_d;
extern float* g_pf4FFTOut1_d;

extern cufftHandle g_stPlan2;
extern float* g_pf4FFTOut2_d;

extern cufftHandle g_stPlan3;
extern float* g_pf4FFTOut3_d;

extern cufftHandle g_stPlan4;
extern float* g_pf4FFTOut4_d;

extern cufftHandle g_stPlan5;
extern float* g_pf4FFTOut5_d;



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
    iCUFFTRet = cufftExecR2C(g_stPlan1,
                             (cufftReal*) g_pf4FFTIn_d,
                             (cufftComplex*) g_pf4FFTOut1_d);
    if (iCUFFTRet != CUFFT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR! FFT1 failed!\n");
        return EXIT_FAILURE;
    }
    
    iCUFFTRet = cufftExecR2C(g_stPlan2, (cufftReal*) g_pf4FFTIn_d, (cufftComplex*) g_pf4FFTOut2_d);
    if (iCUFFTRet != CUFFT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR! FFT2 failed!\n");
        return EXIT_FAILURE;
    }
    
    iCUFFTRet = cufftExecR2C(g_stPlan3, (cufftReal*) g_pf4FFTIn_d, (cufftComplex*) g_pf4FFTOut3_d);
    if (iCUFFTRet != CUFFT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR! FFT3 failed!\n");
        return EXIT_FAILURE;
    }

    iCUFFTRet = cufftExecR2C(g_stPlan4, (cufftReal*) g_pf4FFTIn_d, (cufftComplex*) g_pf4FFTOut4_d);
    if (iCUFFTRet != CUFFT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR! FFT4 failed!\n");
        return EXIT_FAILURE;
    }

    iCUFFTRet = cufftExecR2C(g_stPlan5, (cufftReal*) g_pf4FFTIn_d, (cufftComplex*) g_pf4FFTOut5_d);
    if (iCUFFTRet != CUFFT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR! FFT5 failed!\n");
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}

__global__ void Accumulate(float2 *pf4FFTOut, 
                           float *pf4SumStokes)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    float2 f4FFTOut = pf4FFTOut[i];
    float f4SumStokes = pf4SumStokes[i];

    f4SumStokes += sqrtf((f4FFTOut.x * f4FFTOut.x) + (f4FFTOut.y * f4FFTOut.y));

    pf4SumStokes[i] = f4SumStokes;

    return;
}
/*
__global__ void BatchAccumulate(float2 *pf4FFTOut, 
                                int numBatch,
                                int sizeBatch,
                                float *sumBatches)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    float2 f4FFTOut;
    float isumBatches = sumBatches[i];
    
    for (int n=0; n < numBatch; n++){
        f4FFTOut = pf4FFTOut[i+n*sizeBatch];
        isumBatches += sqrtf((f4FFTOut.x * f4FFTOut.x) + (f4FFTOut.y * f4FFTOut.y));
    }

    sumBatches[i] = isumBatches;

    return;
}*/


__global__ void BatchAccumulate(float2 *pf4FFTOut1, float2 *pf4FFTOut2, float2 *pf4FFTOut3, float2 *pf4FFTOut4, float2 *pf4FFTOut5,
    float *sumBatches1, float *sumBatches2, float *sumBatches3, float *sumBatches4, float *sumBatches5,
    int len_odata)
{
int i = (blockIdx.x * blockDim.x) + threadIdx.x;

float2 f4FFTOut;

float isumBatches = sumBatches1[i];
f4FFTOut = pf4FFTOut1[i];
isumBatches += sqrtf((f4FFTOut.x * f4FFTOut.x) + (f4FFTOut.y * f4FFTOut.y));
sumBatches1[i] = isumBatches;

isumBatches = sumBatches2[i];
for (int n=0; n < 2; n++){
    f4FFTOut = pf4FFTOut2[i+n*(len_odata/2+1)];
    isumBatches += sqrtf((f4FFTOut.x * f4FFTOut.x) + (f4FFTOut.y * f4FFTOut.y));
}
sumBatches2[i] = isumBatches;

isumBatches = sumBatches3[i];
for (int n=0; n < 4; n++){
    f4FFTOut = pf4FFTOut3[i+n*(len_odata/4+1)];
    isumBatches += sqrtf((f4FFTOut.x * f4FFTOut.x) + (f4FFTOut.y * f4FFTOut.y));
}
sumBatches3[i] = isumBatches;

isumBatches = sumBatches4[i];
for (int n=0; n < 8; n++){
    f4FFTOut = pf4FFTOut4[i+n*(len_odata/8+1)];
    isumBatches += sqrtf((f4FFTOut.x * f4FFTOut.x) + (f4FFTOut.y * f4FFTOut.y));
}
sumBatches4[i] = isumBatches;

isumBatches = sumBatches5[i];
for (int n=0; n < 16; n++){
    f4FFTOut = pf4FFTOut5[i+n*(len_odata/16+1)];
    isumBatches += sqrtf((f4FFTOut.x * f4FFTOut.x) + (f4FFTOut.y * f4FFTOut.y));
}
sumBatches5[i] = isumBatches;

return;
}


__global__ void FIR(float *FFTIn, 
                    float *FIRFFTIn,
                    int len,
                    int FFTnum)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    float coeffs1[5] = {0.1474, 0.1092, 0.2135, 0.1092, 0.1474};
    float coeffs2[5] = {0.2312, 0.2270, 0.2549, 0.2270, 0.2312};
    float coeffs3[5] = {0.1625, 0.3244, 0.3085, 0.3244, 0.1625};
    float coeffs4[5] = {-0.0699, 0.3342, 0.6658, 0.3342, -0.0699};
    float sum=0.0;
    switch(FFTnum){
        case 1:
            for (int n=0; n<5; n++){
                if (i+n >= len) continue;
                sum += coeffs1[n]*FFTIn[i+n];
            }
        case 2:
            for (int n=0; n<5; n++){
                if (i+n >= len) continue;
                sum += coeffs2[n]*FFTIn[i+n];
            }
            break;
        case 3:
            for (int n=0; n<5; n++){
                if (i+n >= len) continue;
                sum += coeffs3[n]*FFTIn[i+n];
            }
            break;
        case 4:
            for (int n=0; n<5; n++){
                if (i+n >= len) continue;
                sum += coeffs4[n]*FFTIn[i+n];
            }
            break;
    }
    
    
    FIRFFTIn[i] = sum;

    return;
}


