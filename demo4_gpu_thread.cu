/*demo4_gpu_thread.c
 *
 * Get two numbers from input databuffer, calculate them and write the sum to output databuffer.
 */
#ifdef __cplusplus
extern "C"{
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <unistd.h>
#include "hashpipe.h"
#include "demo4_databuf.h"
#include "demo4_gpu_thread.h"
#include <cuda.h>
#include <cufft.h>
#include <time.h>
#include <cuda_runtime.h>

int g_iIsDataReadDone = FALSE;
char* g_pc4Data_d = NULL;              /* raw data starting address */
char* g_pc4DataRead_d = NULL;          /* raw data read pointer */
int g_iNFFT = NFFT;
int g_iNFFT1 = NFFT;
int g_iNFFT2 = NFFT/2;
int g_iNFFT3 = NFFT/4;
int g_iNFFT4 = NFFT/8;
int g_iNFFT5 = NFFT/16;
int g_ISIZE1 = FFTPLAN1_ISIZE;
int g_OSIZE1 = FFTPLAN1_OSIZE;
int g_ISIZE2 = FFTPLAN2_ISIZE;
int g_OSIZE2 = FFTPLAN2_OSIZE;
int g_ISIZE3 = FFTPLAN3_ISIZE;
int g_OSIZE3 = FFTPLAN3_OSIZE;
int g_ISIZE4 = FFTPLAN4_ISIZE;
int g_OSIZE4 = FFTPLAN4_OSIZE;
int g_ISIZE5 = FFTPLAN5_ISIZE;
int g_OSIZE5 = FFTPLAN5_OSIZE;
dim3 g_dimBCopy(1, 1, 1);
dim3 g_dimGCopy(1, 1);
dim3 g_dimBAccum(1, 1, 1);
dim3 g_dimGAccum(1, 1);
int g_BatchAccumThreads;
int g_BatchAccumBlocks;
float* g_pf4FFTIn_d = NULL;
float* g_FIRFFTIn1_d = NULL;
float* g_FIRFFTIn2_d = NULL;
float* g_FIRFFTIn3_d = NULL;
float* g_FIRFFTIn4_d = NULL;
float* g_FIRFFTIn5_d = NULL;
float2* g_pf4FFTOut1_d = NULL;
float2* g_pf4FFTOut2_d = NULL;
float2* g_pf4FFTOut3_d = NULL;
float2* g_pf4FFTOut4_d = NULL;
float2* g_pf4FFTOut5_d = NULL;
cufftHandle g_stPlan1 = {0};
cufftHandle g_stPlan2 = {0};
cufftHandle g_stPlan3 = {0};
cufftHandle g_stPlan4 = {0};
cufftHandle g_stPlan5 = {0};
float* g_pf4SumStokes = NULL;
float* g_pf4SumStokes_d = NULL;

float* g_sumBatch1 = NULL; 
float* g_sumBatch2 = NULL;
float* g_sumBatch3 = NULL;
float* g_sumBatch4 = NULL;
float* g_sumBatch5 = NULL;



/* BUG: crash if file size is less than 32MB */
int g_iSizeRead = DEF_LEN_IDATA;

static int Init(hashpipe_thread_args_t * args)
{
    int iDevCount = 0;
    cudaDeviceProp stDevProp = {0};
    int iRet = EXIT_SUCCESS;
    cufftResult iCUFFTRet = CUFFT_SUCCESS;
    int iMaxThreadsPerBlock = 0;

    iRet = RegisterSignalHandlers();
    if (iRet != EXIT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR: Signal-handler registration failed!\n");
        return EXIT_FAILURE;
    }

    /* since CUDASafeCallWithCleanUp() calls cudaGetErrorString(),
       it should not be used here - will cause crash if no CUDA device is
       found */
    (void) cudaGetDeviceCount(&iDevCount);
    if (0 == iDevCount)
    {
        (void) fprintf(stderr, "ERROR: No CUDA-capable device found!\n");
        return EXIT_FAILURE;
    }

    /* just use the first device */
    CUDASafeCallWithCleanUp(cudaSetDevice(0));

    CUDASafeCallWithCleanUp(cudaGetDeviceProperties(&stDevProp, 0));
    iMaxThreadsPerBlock = stDevProp.maxThreadsPerBlock;

    /* allocate memory for data array - 32MB is the block size for the VEGAS
       input buffer */
    //CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_pc4DataRead_d, g_iSizeRead));
    CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_pc4Data_d, g_iSizeRead));
    g_pc4DataRead_d = g_pc4Data_d;

    /* calculate kernel parameters */
    if (DEF_LEN_IDATA < iMaxThreadsPerBlock)
    {
        g_dimBCopy.x = DEF_LEN_IDATA;
        g_dimBAccum.x = DEF_LEN_IDATA;
    }
    else
    {
        g_dimBCopy.x = iMaxThreadsPerBlock;
        g_dimBAccum.x = iMaxThreadsPerBlock;
    }
    g_dimGCopy.x = (DEF_LEN_IDATA) / iMaxThreadsPerBlock;
    g_dimGAccum.x = (DEF_LEN_IDATA) / iMaxThreadsPerBlock;

    if (DEF_LEN_ODATA < iMaxThreadsPerBlock){
        g_BatchAccumThreads = DEF_LEN_ODATA;
    }
    else{
        g_BatchAccumThreads = iMaxThreadsPerBlock;
    }
    g_BatchAccumBlocks = DEF_LEN_ODATA/iMaxThreadsPerBlock;


    CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_pf4FFTIn_d, DEF_LEN_IDATA * sizeof(float)));
    CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_FIRFFTIn1_d, DEF_LEN_IDATA * sizeof(float)));
    CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_FIRFFTIn2_d, DEF_LEN_IDATA * sizeof(float)));
    CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_FIRFFTIn3_d, DEF_LEN_IDATA * sizeof(float)));
    CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_FIRFFTIn4_d, DEF_LEN_IDATA * sizeof(float)));
    CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_FIRFFTIn5_d, DEF_LEN_IDATA * sizeof(float)));
    CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_pf4FFTOut1_d, DEF_LEN_IDATA * sizeof(float2)));
    CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_pf4FFTOut2_d, DEF_LEN_IDATA * sizeof(float2)));
    CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_pf4FFTOut3_d, DEF_LEN_IDATA * sizeof(float2)));
    CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_pf4FFTOut4_d, DEF_LEN_IDATA * sizeof(float2)));
    CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_pf4FFTOut5_d, DEF_LEN_IDATA * sizeof(float2)));

    g_pf4SumStokes = (float *) malloc(DEF_LEN_IDATA * sizeof(float));

    if (NULL == g_pf4SumStokes)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return EXIT_FAILURE;
    }
    CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_pf4SumStokes_d, DEF_LEN_IDATA * sizeof(float)));
    CUDASafeCallWithCleanUp(cudaMemset(g_pf4SumStokes_d, '\0', DEF_LEN_IDATA * sizeof(float)));

    CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_sumBatch1, DEF_LEN_ODATA * sizeof(float)));
    CUDASafeCallWithCleanUp(cudaMemset(g_sumBatch1, '\0', DEF_LEN_ODATA * sizeof(float)));
    CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_sumBatch2, DEF_LEN_ODATA * sizeof(float)));
    CUDASafeCallWithCleanUp(cudaMemset(g_sumBatch2, '\0', DEF_LEN_ODATA * sizeof(float)));
    CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_sumBatch3, DEF_LEN_ODATA * sizeof(float)));
    CUDASafeCallWithCleanUp(cudaMemset(g_sumBatch3, '\0', DEF_LEN_ODATA * sizeof(float)));
    CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_sumBatch4, DEF_LEN_ODATA * sizeof(float)));
    CUDASafeCallWithCleanUp(cudaMemset(g_sumBatch4, '\0', DEF_LEN_ODATA * sizeof(float)));
    CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_sumBatch5, DEF_LEN_ODATA * sizeof(float)));
    CUDASafeCallWithCleanUp(cudaMemset(g_sumBatch5, '\0', DEF_LEN_ODATA * sizeof(float)));

    /* create plan */
    iCUFFTRet = cufftPlanMany(&g_stPlan1,        
                              FFTPLAN_RANK,
                              &g_iNFFT1,
                              &g_ISIZE1,
                              FFTPLAN1_ISTRIDE,
                              FFTPLAN1_IDIST,
                              &g_OSIZE1,
                              FFTPLAN1_OSTRIDE,
                              FFTPLAN1_ODIST,
                              CUFFT_R2C,
                              FFTPLAN1_BATCH);
    if (iCUFFTRet != CUFFT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR: Plan1 creation failed!\n");
        return EXIT_FAILURE;
    }
    
    iCUFFTRet = cufftPlanMany(&g_stPlan2, FFTPLAN_RANK, &g_iNFFT2, &g_ISIZE2, FFTPLAN2_ISTRIDE, FFTPLAN2_IDIST, 
                                     &g_OSIZE2, FFTPLAN2_OSTRIDE, FFTPLAN2_ODIST, CUFFT_R2C, FFTPLAN2_BATCH);                           
    if (iCUFFTRet != CUFFT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR: Plan2 creation failed!\n");
        return EXIT_FAILURE;
    }
    
    iCUFFTRet = cufftPlanMany(&g_stPlan3, FFTPLAN_RANK, &g_iNFFT3, &g_ISIZE3, FFTPLAN3_ISTRIDE, FFTPLAN3_IDIST, 
                                    &g_OSIZE3, FFTPLAN3_OSTRIDE, FFTPLAN3_ODIST, CUFFT_R2C, FFTPLAN3_BATCH);                           
    if (iCUFFTRet != CUFFT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR: Plan3 creation failed!\n");
        return EXIT_FAILURE;
    }

    iCUFFTRet = cufftPlanMany(&g_stPlan4, FFTPLAN_RANK, &g_iNFFT4, &g_ISIZE4, FFTPLAN4_ISTRIDE, FFTPLAN4_IDIST, 
                                    &g_OSIZE4, FFTPLAN4_OSTRIDE, FFTPLAN4_ODIST, CUFFT_R2C, FFTPLAN4_BATCH);                           
    if (iCUFFTRet != CUFFT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR: Plan4 creation failed!\n");
        return EXIT_FAILURE;
    }

    iCUFFTRet = cufftPlanMany(&g_stPlan5, FFTPLAN_RANK, &g_iNFFT5, &g_ISIZE5, FFTPLAN5_ISTRIDE, FFTPLAN5_IDIST, 
                                &g_OSIZE5, FFTPLAN5_OSTRIDE, FFTPLAN5_ODIST, CUFFT_R2C, FFTPLAN5_BATCH);                           
    if (iCUFFTRet != CUFFT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR: Plan5 creation failed!\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

/* function that frees resources */
void CleanUp()
{
    /* free resources */

    if (g_pc4Data_d != NULL)
    {
        (void) cudaFree(g_pc4Data_d);
        g_pc4Data_d = NULL;
    }
    if (g_pf4FFTIn_d != NULL)
    {
        (void) cudaFree(g_pf4FFTIn_d);
        g_pf4FFTIn_d = NULL;
    }
    if (g_FIRFFTIn1_d != NULL){ (void) cudaFree(g_FIRFFTIn1_d); g_FIRFFTIn1_d = NULL;}
    if (g_FIRFFTIn2_d != NULL){ (void) cudaFree(g_FIRFFTIn2_d); g_FIRFFTIn2_d = NULL;}
    if (g_FIRFFTIn3_d != NULL){ (void) cudaFree(g_FIRFFTIn3_d); g_FIRFFTIn3_d = NULL;}
    if (g_FIRFFTIn4_d != NULL){ (void) cudaFree(g_FIRFFTIn4_d); g_FIRFFTIn4_d = NULL;}
    if (g_FIRFFTIn5_d != NULL){ (void) cudaFree(g_FIRFFTIn5_d); g_FIRFFTIn5_d = NULL;}
    if (g_pf4FFTOut1_d != NULL)
    {
        (void) cudaFree(g_pf4FFTOut1_d);
        g_pf4FFTOut1_d = NULL;
    }
    if (g_pf4FFTOut2_d != NULL)
    {
        (void) cudaFree(g_pf4FFTOut2_d);
        g_pf4FFTOut2_d = NULL;
    }
    if (g_pf4FFTOut3_d != NULL)
    {
        (void) cudaFree(g_pf4FFTOut3_d);
        g_pf4FFTOut3_d = NULL;
    }
    if (g_pf4FFTOut4_d != NULL)
    {
        (void) cudaFree(g_pf4FFTOut4_d);
        g_pf4FFTOut4_d = NULL;
    }
    if (g_pf4FFTOut5_d != NULL)
    {
        (void) cudaFree(g_pf4FFTOut5_d);
        g_pf4FFTOut5_d = NULL;
    }
    if (g_pf4SumStokes != NULL)
    {
        free(g_pf4SumStokes);
        g_pf4SumStokes = NULL;
    }
    if (g_pf4SumStokes_d != NULL)
    {
        (void) cudaFree(g_pf4SumStokes_d);
        g_pf4SumStokes_d = NULL;
    }
    if (g_sumBatch2 != NULL)
    {
        (void) cudaFree(g_sumBatch2);
        g_sumBatch2 = NULL;
    }
    if (g_sumBatch1 != NULL)
    {
        (void) cudaFree(g_sumBatch1);
        g_sumBatch1 = NULL;
    }
    if (g_sumBatch3 != NULL)
    {
        (void) cudaFree(g_sumBatch3);
        g_sumBatch3 = NULL;
    }
    if (g_sumBatch4 != NULL)
    {
        (void) cudaFree(g_sumBatch4);
        g_sumBatch4 = NULL;
    }
    if (g_sumBatch5 != NULL)
    {
        (void) cudaFree(g_sumBatch5);
        g_sumBatch5 = NULL;
    }


    /* destroy plan */
    /* TODO: check for plan */
    (void) cufftDestroy(g_stPlan1);
    (void) cufftDestroy(g_stPlan2);
    (void) cufftDestroy(g_stPlan3);
    (void) cufftDestroy(g_stPlan4);
    (void) cufftDestroy(g_stPlan5);
    /* TODO: check if open */
    cpgclos();
    return;
}

/*
 * Registers handlers for SIGTERM and CTRL+C
 */
int RegisterSignalHandlers()
{
    struct sigaction stSigHandler = {{0}};
    int iRet = EXIT_SUCCESS;

    /* register the CTRL+C-handling function */
    stSigHandler.sa_handler = HandleStopSignals;
    iRet = sigaction(SIGINT, &stSigHandler, NULL);
    if (iRet != EXIT_SUCCESS)
    {
        (void) fprintf(stderr,
                       "ERROR: Handler registration failed for signal %d!\n",
                       SIGINT);
        return EXIT_FAILURE;
    }

    /* register the SIGTERM-handling function */
    stSigHandler.sa_handler = HandleStopSignals;
    iRet = sigaction(SIGTERM, &stSigHandler, NULL);
    if (iRet != EXIT_SUCCESS)
    {
        (void) fprintf(stderr,
                       "ERROR: Handler registration failed for signal %d!\n",
                       SIGTERM);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

/*
 * Catches SIGTERM and CTRL+C and cleans up before exiting
 */
void HandleStopSignals(int iSigNo)
{
    /* clean up */
    CleanUp();

    /* exit */
    exit(EXIT_SUCCESS);

    /* never reached */
    return;
}

void __CUDASafeCallWithCleanUp(cudaError_t iRet,
                               const char* pcFile,
                               const int iLine,
                               void (*pCleanUp)(void))
{
    if (iRet != cudaSuccess)
    {
        (void) fprintf(stderr,
                       "ERROR: File <%s>, Line %d: %s\n",
                       pcFile,
                       iLine,
                       cudaGetErrorString(iRet));
        /* free resources */
        (*pCleanUp)();
        exit(EXIT_FAILURE);
    }

    return;
}
/*
 * Prints usage information
 */
void PrintUsage(const char *pcProgName)
{
    (void) printf("Usage: %s [options] <data-file>\n",
                  pcProgName);
    (void) printf("    -h  --help                           ");
    (void) printf("Display this usage information\n");
    (void) printf("    -n  --nfft <value>                   ");
    (void) printf("Number of points in FFT\n");
    (void) printf("    -p  --pfb                            ");
    (void) printf("Enable PFB\n");
    (void) printf("    -a  --nacc <value>                   ");
    (void) printf("Number of spectra to add\n");
    (void) printf("    -s  --fsamp <value>                  ");
    (void) printf("Sampling frequency\n");

    return;
}


static void *run(hashpipe_thread_args_t * args)
{
    // Local aliases to shorten access to args fields
    demo4_input_databuf_t *db_in = (demo4_input_databuf_t *)args->ibuf;
    demo4_output_databuf_t *db_out = (demo4_output_databuf_t *)args->obuf;
    hashpipe_status_t st = args->st;
    const char * status_key = args->thread_desc->skey;

    int rv;
    uint64_t mcnt=0;
    int curblock_in=0;
    int curblock_out=0;
    
    int nhits = 0;
    char *data_raw; // raw data will be feed to gpu thread
    data_raw = (char *)malloc(g_iSizeRead*sizeof(char));

    int n_frames; // number of frames has been processed

    int iRet = EXIT_SUCCESS;
    int iSpecCount = 0;
    int iNumAcc = DEF_ACC;
    //if(iNumAcc > g_iSizeRead/DEF_LEN_IDATA){iNumAcc=g_iSizeRead/DEF_LEN_IDATA;} // if accumulation number larger than data buffer, setit to number spectra frames of buffer
	int n_spec = 0; // number of spectrum
    int iProcData = 0;
    cudaError_t iCUDARet = cudaSuccess;
    struct timeval stStart = {0};
    struct timeval stStop = {0};
    const char *pcProgName = NULL;
    int iNextOpt = 0;
    /* valid short options */
    const char* const pcOptsShort = "hb:n:pa:s:";
    /* valid long options */
    const struct option stOptsLong[] = {
        { "help",           0, NULL, 'h' },
        { "nsub",           1, NULL, 'b' },
        { "nfft",           1, NULL, 'n' },
        { "pfb",            0, NULL, 'p' },
        { "nacc",           1, NULL, 'a' },
        { "fsamp",          1, NULL, 's' },
        { NULL,             0, NULL, 0   }
    };

    while (run_threads()) {

        hashpipe_status_lock_safe(&st);
        hputi4(st.buf, "GPUBLKIN", curblock_in);
        hputs(st.buf, status_key, "waiting");
       	hputi4(st.buf, "GPUBKOUT", curblock_out);
		hputi8(st.buf,"GPUMCNT",mcnt);
        hashpipe_status_unlock_safe(&st);
        n_spec = 0;
        
        // Wait for new output block to be free
        while ((rv=demo4_output_databuf_wait_free(db_out, curblock_out)) != HASHPIPE_OK) {
            if (rv==HASHPIPE_TIMEOUT) {
                hashpipe_status_lock_safe(&st);
                hputs(st.buf, status_key, "blocked gpu out");
                hashpipe_status_unlock_safe(&st);
                continue;
            } else {
                hashpipe_error(__FUNCTION__, "error waiting for free databuf");
                pthread_exit(NULL);
                break;
            }
        }

        while(iSpecCount < iNumAcc){
            // Wait for new input block to be filled
            while ((rv=demo4_input_databuf_wait_filled(db_in, curblock_in)) != HASHPIPE_OK) {
                if (rv==HASHPIPE_TIMEOUT) {
                    hashpipe_status_lock_safe(&st);
                    hputs(st.buf, status_key, "blocked");
                    hashpipe_status_unlock_safe(&st);
                    continue;
                } else {
                    hashpipe_error(__FUNCTION__, "error waiting for filled databuf");
                    pthread_exit(NULL);
                    break;
                }
            }

            // Note processing status
            hashpipe_status_lock_safe(&st);
            hputs(st.buf, status_key, "processing gpu");
            hashpipe_status_unlock_safe(&st);

            //get data from input databuf to local
            memcpy(data_raw,db_in->block[curblock_in].data_block,g_iSizeRead*sizeof(char));
            
            // write new data to the gpu buffer
			CUDASafeCallWithCleanUp(cudaMemcpy(g_pc4Data_d,
                data_raw,
                g_iSizeRead*sizeof(char),
                cudaMemcpyHostToDevice));

            /* whenever there is a read, reset the read pointer to the beginning */
			g_pc4DataRead_d = g_pc4Data_d;

            CopyDataForFFT<<<g_dimGCopy, g_dimBCopy>>>(g_pc4DataRead_d,
                g_pf4FFTIn_d);

            //FIR<<<g_dimGCopy, g_dimBCopy>>>(g_pf4FFTIn_d,g_FIRFFTIn1_d,DEF_LEN_IDATA,1);
            //FIR<<<g_dimGCopy, g_dimBCopy>>>(g_pf4FFTIn_d,g_FIRFFTIn2_d,DEF_LEN_IDATA,2);
            //FIR<<<g_dimGCopy, g_dimBCopy>>>(g_pf4FFTIn_d,g_FIRFFTIn3_d,DEF_LEN_IDATA,3);
            //FIR<<<g_dimGCopy, g_dimBCopy>>>(g_pf4FFTIn_d,g_FIRFFTIn4_d,DEF_LEN_IDATA,4);

            CUDASafeCallWithCleanUp(cudaThreadSynchronize());
            iCUDARet = cudaGetLastError();
            if (iCUDARet != cudaSuccess){
                (void) fprintf(stderr,
                "ERROR: File <%s>, Line %d: %s\n",
                __FILE__,
                __LINE__,
                cudaGetErrorString(iCUDARet));
                CleanUp();
            }

            /* do fft */
            iRet = DoFFT();
            if (iRet != EXIT_SUCCESS){
                (void) fprintf(stderr, "ERROR! FFT failed!\n");
                CleanUp();
            }

            BatchAccumulate<<<g_BatchAccumBlocks, g_BatchAccumThreads>>>(g_pf4FFTOut1_d,
                            1,
                            DEF_LEN_ODATA+1,
                            g_sumBatch1); 
            
            BatchAccumulate<<<g_BatchAccumBlocks, g_BatchAccumThreads>>>(g_pf4FFTOut2_d,
                            2,
                            DEF_LEN_ODATA/2+1,
                            g_sumBatch2);
                       
            BatchAccumulate<<<g_BatchAccumBlocks, g_BatchAccumThreads>>>(g_pf4FFTOut3_d,
                            4,
                            DEF_LEN_ODATA/4+1,
                            g_sumBatch3);

            BatchAccumulate<<<g_BatchAccumBlocks, g_BatchAccumThreads>>>(g_pf4FFTOut4_d,
                            8,
                            DEF_LEN_ODATA/8+1,
                            g_sumBatch4);

            BatchAccumulate<<<g_BatchAccumBlocks, g_BatchAccumThreads>>>(g_pf4FFTOut5_d,
                            16,
                            DEF_LEN_ODATA/16+1,
                            g_sumBatch5);
                
            CUDASafeCallWithCleanUp(cudaThreadSynchronize());
            iCUDARet = cudaGetLastError();
            if (iCUDARet != cudaSuccess)
            {
                (void) fprintf(stderr,
                "ERROR: File <%s>, Line %d: %s\n",
                __FILE__,
                __LINE__,
                cudaGetErrorString(iCUDARet));
                CleanUp();
            }
            ++iSpecCount;
            // Mark input block as free and advance
            demo4_input_databuf_set_free(db_in, curblock_in);
            curblock_in = (curblock_in + 1) % db_in->header.n_block;
        }


        CUDASafeCallWithCleanUp(cudaMemcpy(g_pf4SumStokes,
            g_sumBatch1,
            (DEF_LEN_ODATA
            * sizeof(float)),
            cudaMemcpyDeviceToHost));
                
        CUDASafeCallWithCleanUp(cudaMemcpy(g_pf4SumStokes + DEF_LEN_ODATA,
                        g_sumBatch2,
                        (DEF_LEN_ODATA/2
                        * sizeof(float)),
                        cudaMemcpyDeviceToHost));

        CUDASafeCallWithCleanUp(cudaMemcpy(g_pf4SumStokes + DEF_LEN_ODATA*3/2,
                        g_sumBatch3,
                        (DEF_LEN_ODATA/4
                        * sizeof(float)),
                        cudaMemcpyDeviceToHost));

        CUDASafeCallWithCleanUp(cudaMemcpy(g_pf4SumStokes + DEF_LEN_ODATA*7/4,
                        g_sumBatch4,
                        (DEF_LEN_ODATA/8
                        * sizeof(float)),
                        cudaMemcpyDeviceToHost));

        CUDASafeCallWithCleanUp(cudaMemcpy(g_pf4SumStokes + DEF_LEN_ODATA*15/8,
                        g_sumBatch5,
                        (DEF_LEN_ODATA/16
                        * sizeof(float)),
                        cudaMemcpyDeviceToHost));

        /*
        CUDASafeCallWithCleanUp(cudaMemcpy(g_pf4SumStokes,
                                        g_sumBatch1+DEF_LEN_ODATA/2,
                                        (DEF_LEN_ODATA/2
                                        * sizeof(float)),
                                        cudaMemcpyDeviceToHost));

        CUDASafeCallWithCleanUp(cudaMemcpy(g_pf4SumStokes + DEF_LEN_ODATA/2,
                                            g_sumBatch2+DEF_LEN_ODATA/2,
                                            (DEF_LEN_ODATA/2
                                            * sizeof(float)),
                                            cudaMemcpyDeviceToHost));

        CUDASafeCallWithCleanUp(cudaMemcpy(g_pf4SumStokes + DEF_LEN_ODATA,
                                            g_sumBatch3+DEF_LEN_ODATA/2,
                                            (DEF_LEN_ODATA/2
                                            * sizeof(float)),
                                            cudaMemcpyDeviceToHost));

        CUDASafeCallWithCleanUp(cudaMemcpy(g_pf4SumStokes + DEF_LEN_ODATA*3/2,
                                            g_sumBatch4+DEF_LEN_ODATA/2,
                                            (DEF_LEN_ODATA/2
                                            * sizeof(float)),
                                            cudaMemcpyDeviceToHost));

        CUDASafeCallWithCleanUp(cudaMemcpy(g_pf4SumStokes + DEF_LEN_ODATA*2,
                                            g_sumBatch5+DEF_LEN_ODATA/2,
                                            (DEF_LEN_ODATA/2
                                            * sizeof(float)),
                                            cudaMemcpyDeviceToHost));
                                                */
        memcpy(db_out->block[curblock_out].Stokes_Full+SIZEOF_OUT_STOKES*n_spec,g_pf4SumStokes,SIZEOF_OUT_STOKES*sizeof(float));
            //printf("Stokes to output done!\n");
        n_spec++; 

        /* reset time */
        iSpecCount = 0;
        /* zero accumulators */
        CUDASafeCallWithCleanUp(cudaMemset(g_pf4SumStokes_d,
                                        '\0',
                                        (DEF_LEN_IDATA
                                        * sizeof(float))));
        CUDASafeCallWithCleanUp(cudaMemset(g_sumBatch2,
                                            '\0',
                                            (DEF_LEN_ODATA
                                                * sizeof(float))));
        CUDASafeCallWithCleanUp(cudaMemset(g_sumBatch1,
                                        '\0',
                                        (DEF_LEN_ODATA
                                        * sizeof(float))));
        CUDASafeCallWithCleanUp(cudaMemset(g_sumBatch3,
                                        '\0',
                                        (DEF_LEN_ODATA
                                        * sizeof(float))));
        CUDASafeCallWithCleanUp(cudaMemset(g_sumBatch4,
                                        '\0',
                                        (DEF_LEN_ODATA
                                        * sizeof(float))));
        CUDASafeCallWithCleanUp(cudaMemset(g_sumBatch5,
                                        '\0',
                                        (DEF_LEN_ODATA
                                        * sizeof(float))));
        /* if time to read from input buffer */
        iProcData = 0;
        (void) gettimeofday(&stStop, NULL);
        /*(void) printf("Time taken (barring Init()): %gs\n",
                ((stStop.tv_sec + (stStop.tv_usec * USEC2SEC))
                - (stStart.tv_sec + (stStart.tv_usec * USEC2SEC))));*/

        //return EXIT_SUCCESS;

        //display number of frames in status
        hashpipe_status_lock_safe(&st);
        hputi4(st.buf,"NFRAMES",n_frames);
        hashpipe_status_unlock_safe(&st);


		// Mark output block as full and advance
		demo4_output_databuf_set_filled(db_out, curblock_out);
		curblock_out = (curblock_out + 1) % db_out->header.n_block;

		// Mark input block as free and advance
		//demo4_input_databuf_set_free(db_in, curblock_in);
		//curblock_in = (curblock_in + 1) % db_in->header.n_block;
		mcnt++;
		/* Check for cancel */
		pthread_testcancel();
	}
	CleanUp();
}

static hashpipe_thread_desc_t demo4_gpu_thread = {
    name: "demo4_gpu_thread",
    skey: "GPUSTAT",
    init: Init,
    //init: NULL,
    run:  run,
    ibuf_desc: {demo4_input_databuf_create},
    obuf_desc: {demo4_output_databuf_create}
};

static __attribute__((constructor)) void ctor()
{
	register_hashpipe_thread(&demo4_gpu_thread);
}
#ifdef __cplusplus
}
#endif
