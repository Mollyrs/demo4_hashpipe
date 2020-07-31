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

#define DEF_LEN_SPEC        8192*64 //2048         // default value for g_iNFFT 

#define DEF_ACC             1 //1024           // default number of spectra to accumulate

#define FFTPLAN_RANK        1
#define FFTPLAN_ISTRIDE     1
#define FFTPLAN_OSTRIDE     1
#define FFTPLAN_IDIST       DEF_LEN_SPEC //2048
#define FFTPLAN_ODIST       DEF_LEN_SPEC/2 + 1 //1025
#define FFTPLAN_BATCH       4 
#define FFTPLAN_ISIZE       FFTPLAN_IDIST*FFTPLAN_BATCH
#define FFTPLAN_OSIZE       FFTPLAN_ODIST*FFTPLAN_BATCH

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
                               float* pf4FFTIn);
int DoFFT(void);
__global__ void Accumulate(float2 *pf4FFTOut,
                           float* pfSumStokes);
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


