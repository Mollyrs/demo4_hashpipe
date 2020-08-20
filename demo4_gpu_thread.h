/**
 * @file demo4_gpu_proc.h
 * Hashpipe Demo3
 *  Top-level header file
 *
 * @author Sparke Pei
 * @date 2017.05.08
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <string.h>     // for memset(), strncpy(), memcpy(), strerror() 
#include <sys/types.h>  // for open() 
#include <sys/stat.h>   // for open() 
#include <fcntl.h>      // for open() 
#include <unistd.h>     // for close() and usleep() 
#include <cpgplot.h>    // for cpg*() 
#include <float.h>      // for FLT_MAX 
#include <getopt.h>     // for option parsing 
#include <assert.h>     // for assert() 
#include <errno.h>      // for errno 
#include <signal.h>     // for signal-handling 
#include <math.h>       // for log10f() in Plot() 
#include <sys/time.h>   // for gettimeofday() 
//#include "demo4_databuf.h"

#define FALSE               0
#define TRUE                1

#define DEF_LEN_IDATA       67108864 
#define DEF_LEN_ODATA       DEF_LEN_IDATA/2   
#define NFFT                67108864 

#define DEF_ACC             1          // default number of spectra to accumulate 

#define FFTPLAN_RANK        1

#define FFTPLAN1_BATCH       1
#define FFTPLAN1_ISTRIDE     1
#define FFTPLAN1_OSTRIDE     1
#define FFTPLAN1_IDIST       DEF_LEN_IDATA
#define FFTPLAN1_ODIST       DEF_LEN_IDATA/2 + 1 
#define FFTPLAN1_ISIZE       FFTPLAN1_IDIST*FFTPLAN1_BATCH
#define FFTPLAN1_OSIZE       FFTPLAN1_ODIST*FFTPLAN1_BATCH

#define FFTPLAN2_BATCH       2 
#define FFTPLAN2_ISTRIDE     1
#define FFTPLAN2_OSTRIDE     1
#define FFTPLAN2_IDIST       DEF_LEN_IDATA/2
#define FFTPLAN2_ODIST       DEF_LEN_IDATA/4 + 1 
#define FFTPLAN2_ISIZE       FFTPLAN2_IDIST*FFTPLAN2_BATCH
#define FFTPLAN2_OSIZE       FFTPLAN2_ODIST*FFTPLAN2_BATCH

#define FFTPLAN3_BATCH       4
#define FFTPLAN3_ISTRIDE     1
#define FFTPLAN3_OSTRIDE     1
#define FFTPLAN3_IDIST       DEF_LEN_IDATA/4
#define FFTPLAN3_ODIST       DEF_LEN_IDATA/8 + 1 
#define FFTPLAN3_ISIZE       FFTPLAN3_IDIST*FFTPLAN3_BATCH
#define FFTPLAN3_OSIZE       FFTPLAN3_ODIST*FFTPLAN3_BATCH

#define FFTPLAN4_BATCH       8
#define FFTPLAN4_ISTRIDE     1
#define FFTPLAN4_OSTRIDE     1
#define FFTPLAN4_IDIST       DEF_LEN_IDATA/8
#define FFTPLAN4_ODIST       DEF_LEN_IDATA/16 + 1 
#define FFTPLAN4_ISIZE       FFTPLAN4_IDIST*FFTPLAN4_BATCH
#define FFTPLAN4_OSIZE       FFTPLAN4_ODIST*FFTPLAN4_BATCH

#define FFTPLAN5_BATCH       16 
#define FFTPLAN5_ISTRIDE     1
#define FFTPLAN5_OSTRIDE     1
#define FFTPLAN5_IDIST       DEF_LEN_IDATA/16
#define FFTPLAN5_ODIST       DEF_LEN_IDATA/32 + 1 
#define FFTPLAN5_ISIZE       FFTPLAN5_IDIST*FFTPLAN5_BATCH
#define FFTPLAN5_OSIZE       FFTPLAN5_ODIST*FFTPLAN5_BATCH

#define USEC2SEC            1e-6

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned char BYTE;

/**
 * Initialises the program.
 */
//int Init(void);

/**
 * Reads all data from the input file and loads it into memory.
 */
//int LoadDataToMem(void);

/**
 * Reads one block (32MB) of data form memory.
 */
//int ReadData(void);

/*
 * Perform polyphase filtering.
 *
 * @param[in]   pc4Data     Input data (raw data read from memory)
 * @param[out]  pf4FFTIn    Output data (input to FFT)
 */

__global__ void CopyDataForFFT(char* pc4Data,
                               float* pf4FFTIn
                               );
int DoFFT(void);
__global__ void Accumulate(float2 *pf4FFTOut,
                           float* pfSumStokes);
/*__global__ void BatchAccumulate(float2 *pf4FFTOut,
                            int numBatch,
                            int sizeBatch,
                            float* pfSumStokes);*/
__global__ void BatchAccumulate(float2 *g_pf4FFTOut1_d,
                float2 *g_pf4FFTOut2_d,
                float2 *g_pf4FFTOut3_d,
                float2 *g_pf4FFTOut4_d,
                float2 *g_pf4FFTOut5_d,
                float* g_sumBatch1,
                float* g_sumBatch2,
                float* g_sumBatch3,
                float* g_sumBatch4,
                float* g_sumBatch5,
                int len
            );

__global__ void FIR(float *FFTIn, 
                    float *FIRFFTIn,
                    int len,
                    int FFTnum);


void CleanUp(void);

#define CUDASafeCallWithCleanUp(iRet)   __CUDASafeCallWithCleanUp(iRet,       \
                                                                  __FILE__,   \
                                                                  __LINE__,   \
                                                                  &CleanUp)

void __CUDASafeCallWithCleanUp(cudaError_t iRet,
                               const char* pcFile,
                               const int iLine,
                               void (*pCleanUp)(void));

#if BENCHMARKING
void PrintBenchmarks(float fAvgPFB,
                     int iCountPFB,
                     float fAvgCpInFFt,
                     int iCountCpInFFT,
                     float fAvgFFT,
                     int iCountFFT,
                     float fAvgAccum,
                     int iCountAccum,
                     float fAvgCpOut,
                     int iCountCpOut);
#endif

/* PGPLOT function declarations */
int InitPlot(void);
void Plot(void);
//int writedata(void);
int RegisterSignalHandlers();
void HandleStopSignals(int iSigNo);
void PrintUsage(const char* pcProgName);



#ifdef __cplusplus
}
#endif


