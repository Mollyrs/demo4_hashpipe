#include <stdint.h>
#include <stdio.h>
#include "hashpipe.h"
#include "hashpipe_databuf.h"

//#define TEST_MODE		1	// for test
#define CACHE_ALIGNMENT         256	// cache alignment size	
#define N_INPUT_BLOCKS          3 	// number of input blocks
#define N_OUTPUT_BLOCKS         3	// number of output blocks
#define PAGE_SIZE	      	    1 //16384 //(8*32768)	// number of spectra per memory, define memory size
#define N_CHANS_PER_SPEC	    8192*64 //2048 //4096	// number of FFT channels per spectrum
#define N_BYTES_PER_SAMPLE	    1	// number of bytes per sample
#define N_PKTS_PER_SPEC         512	// number packets per spectrum
#define N_BYTES_HEAD		    8	// number bytes of header in packets
#define N_BYTES_PER_PKT		    1032 //4104	// number bytes per packets
#define N_BYTES_PKT_DATA	    (N_BYTES_PER_PKT-N_BYTES_HEAD)
#define ACC_LEN			        1 //512 // accumulation length
#define SIZEOF_INPUT_DATA_BUF	PAGE_SIZE*N_BYTES_PKT_DATA*N_BYTES_PER_SAMPLE*N_PKTS_PER_SPEC
#define SIZEOF_OUT_STOKES	    PAGE_SIZE*N_CHANS_PER_SPEC/ACC_LEN/2
// Used to pad after hashpipe_databuf_t to maintain cache alignment
typedef uint8_t hashpipe_databuf_cache_alignment[
  CACHE_ALIGNMENT - (sizeof(hashpipe_databuf_t)%CACHE_ALIGNMENT)
];

/* INPUT BUFFER STRUCTURES
  */
typedef struct demo4_input_block_header {
   uint64_t mcnt;                    // mcount of first packet
} demo4_input_block_header_t;

typedef uint8_t demo4_input_header_cache_alignment[
   CACHE_ALIGNMENT - (sizeof(demo4_input_block_header_t)%CACHE_ALIGNMENT)
];

typedef struct demo4_input_block {
   demo4_input_block_header_t header;
   demo4_input_header_cache_alignment padding; // Maintain cache alignment
   char data_block[SIZEOF_INPUT_DATA_BUF*sizeof(char)]; // define input buffer
} demo4_input_block_t;

typedef struct demo4_input_databuf {
   hashpipe_databuf_t header;
   hashpipe_databuf_cache_alignment padding; // Maintain cache alignment
   demo4_input_block_t block[N_INPUT_BLOCKS];
} demo4_input_databuf_t;


/*
  * OUTPUT BUFFER STRUCTURES
  */
typedef struct demo4_output_block_header {
   uint64_t mcnt;
} demo4_output_block_header_t;

typedef uint8_t demo4_output_header_cache_alignment[
   CACHE_ALIGNMENT - (sizeof(demo4_output_block_header_t)%CACHE_ALIGNMENT)
];

typedef struct demo4_output_block {
   demo4_output_block_header_t header;
   demo4_output_header_cache_alignment padding; // Maintain cache alignment
   float Stokes_Full[SIZEOF_OUT_STOKES*sizeof(float)];
} demo4_output_block_t;

typedef struct demo4_output_databuf {
   hashpipe_databuf_t header;
   hashpipe_databuf_cache_alignment padding; // Maintain cache alignment
   demo4_output_block_t block[N_OUTPUT_BLOCKS];
} demo4_output_databuf_t;

/*
 * INPUT BUFFER FUNCTIONS
 */
hashpipe_databuf_t *demo4_input_databuf_create(int instance_id, int databuf_id);

static inline demo4_input_databuf_t *demo4_input_databuf_attach(int instance_id, int databuf_id)
{
    return (demo4_input_databuf_t *)hashpipe_databuf_attach(instance_id, databuf_id);
}

static inline int demo4_input_databuf_detach(demo4_input_databuf_t *d)
{
    return hashpipe_databuf_detach((hashpipe_databuf_t *)d);
}

static inline void demo4_input_databuf_clear(demo4_input_databuf_t *d)
{
    hashpipe_databuf_clear((hashpipe_databuf_t *)d);
}

static inline int demo4_input_databuf_block_status(demo4_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_block_status((hashpipe_databuf_t *)d, block_id);
}

static inline int demo4_input_databuf_total_status(demo4_input_databuf_t *d)
{
    return hashpipe_databuf_total_status((hashpipe_databuf_t *)d);
}

static inline int demo4_input_databuf_wait_free(demo4_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_wait_free((hashpipe_databuf_t *)d, block_id);
}

static inline int demo4_input_databuf_busywait_free(demo4_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_busywait_free((hashpipe_databuf_t *)d, block_id);
}

static inline int demo4_input_databuf_wait_filled(demo4_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_wait_filled((hashpipe_databuf_t *)d, block_id);
}

static inline int demo4_input_databuf_busywait_filled(demo4_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_busywait_filled((hashpipe_databuf_t *)d, block_id);
}

static inline int demo4_input_databuf_set_free(demo4_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_set_free((hashpipe_databuf_t *)d, block_id);
}

static inline int demo4_input_databuf_set_filled(demo4_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_set_filled((hashpipe_databuf_t *)d, block_id);
}

/*
 * OUTPUT BUFFER FUNCTIONS
 */

hashpipe_databuf_t *demo4_output_databuf_create(int instance_id, int databuf_id);

static inline void demo4_output_databuf_clear(demo4_output_databuf_t *d)
{
    hashpipe_databuf_clear((hashpipe_databuf_t *)d);
}

static inline demo4_output_databuf_t *demo4_output_databuf_attach(int instance_id, int databuf_id)
{
    return (demo4_output_databuf_t *)hashpipe_databuf_attach(instance_id, databuf_id);
}

static inline int demo4_output_databuf_detach(demo4_output_databuf_t *d)
{
    return hashpipe_databuf_detach((hashpipe_databuf_t *)d);
}

static inline int demo4_output_databuf_block_status(demo4_output_databuf_t *d, int block_id)
{
    return hashpipe_databuf_block_status((hashpipe_databuf_t *)d, block_id);
}

static inline int demo4_output_databuf_total_status(demo4_output_databuf_t *d)
{
    return hashpipe_databuf_total_status((hashpipe_databuf_t *)d);
}

static inline int demo4_output_databuf_wait_free(demo4_output_databuf_t *d, int block_id)
{
    return hashpipe_databuf_wait_free((hashpipe_databuf_t *)d, block_id);
}

static inline int demo4_output_databuf_busywait_free(demo4_output_databuf_t *d, int block_id)
{
    return hashpipe_databuf_busywait_free((hashpipe_databuf_t *)d, block_id);
}
static inline int demo4_output_databuf_wait_filled(demo4_output_databuf_t *d, int block_id)
{
    return hashpipe_databuf_wait_filled((hashpipe_databuf_t *)d, block_id);
}

static inline int demo4_output_databuf_busywait_filled(demo4_output_databuf_t *d, int block_id)
{
    return hashpipe_databuf_busywait_filled((hashpipe_databuf_t *)d, block_id);
}

static inline int demo4_output_databuf_set_free(demo4_output_databuf_t *d, int block_id)
{
    return hashpipe_databuf_set_free((hashpipe_databuf_t *)d, block_id);
}

static inline int demo4_output_databuf_set_filled(demo4_output_databuf_t *d, int block_id)
{
    return hashpipe_databuf_set_filled((hashpipe_databuf_t *)d, block_id);
}


