/*
 * demo4_net_thread.c
 *
 * This allows you to receive pakets from local ethernet, unpack the packets, and then write data into a shared memory buffer. 
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#include <sched.h>
#include <smmintrin.h>
#include <immintrin.h>
#include "hashpipe.h"
#include "demo4_databuf.h"

#define PKTSOCK_BYTES_PER_FRAME (4096)
#define PKTSOCK_FRAMES_PER_BLOCK (1) //(8)
#define PKTSOCK_NBLOCKS (1) //(20)
#define PKTSOCK_NFRAMES (PKTSOCK_FRAMES_PER_BLOCK * PKTSOCK_NBLOCKS)

#define BYTES_PER_PKT 1032
#define BYTES_PER_PAYLOAD 1024
#define NUM_PACKETS_PER_BUFFER N_PKTS_PER_IBUF

double MJD;

static int init(hashpipe_thread_args_t * args)
{
	// define network params
	char bindhost[80];
	int bindport = 12345;
	hashpipe_status_t st = args->st;
	strcpy(bindhost, "0.0.0.0");
	
	hashpipe_status_lock_safe(&st);
	// Get info from status buffer if present (no change if not present)
	hgets(st.buf, "BINDHOST", 80, bindhost);
	hgeti4(st.buf, "BINDPORT", &bindport);
	// Store bind host/port info etc in status buffer
	hputs(st.buf, "BINDHOST", bindhost);
	hputi4(st.buf, "BINDPORT", bindport);
	hputi8(st.buf, "NPACKETS", 0);
	hputi8(st.buf, "PKTSIZE", BYTES_PER_PKT);
	hashpipe_status_unlock_safe(&st);

	// Set up pktsock 
	struct hashpipe_pktsock *p_ps = (struct hashpipe_pktsock *)
	malloc(sizeof(struct hashpipe_pktsock));
	if(!p_ps) {
		perror(__FUNCTION__);
		return -1;
	}
	/* Make frame_size be a divisor of block size so that frames will be
	contiguous in mapped mempory.  block_size must also be a multiple of
	page_size.  Easiest way is to oversize the frames to be 16384 bytes, which
	is bigger than we need, but keeps things easy. */
	p_ps->frame_size = PKTSOCK_BYTES_PER_FRAME;
	p_ps->nframes = PKTSOCK_NFRAMES;
	p_ps->nblocks = PKTSOCK_NBLOCKS;
	int rv = hashpipe_pktsock_open(p_ps, bindhost, PACKET_RX_RING);
	if (rv!=HASHPIPE_OK) {
	hashpipe_error("demo4_net_thread", "Error opening pktsock.");
	pthread_exit(NULL);
	}
	// Store packet socket pointer in args
	args->user_data = p_ps;
	// Success!
	return 0;
}


typedef struct {
	uint64_t    seq;            // sequence of packets
	unsigned short int source_id;
	unsigned short int spec_id;
} packet_header_t;

static inline void get_header(unsigned char *p_frame, packet_header_t * pkt_header)
{
	uint64_t raw_header;
	raw_header = *(unsigned long long *)PKT_UDP_DATA(p_frame);
	pkt_header->seq        = raw_header  & 0x00ffffffffffffff;
	pkt_header->source_id = (short int)(raw_header >> 56);
	pkt_header->spec_id = (short int)(raw_header & 0x01);

	#ifdef TEST_MODE
	fprintf(stderr,"**Header**\n");
	fprintf(stderr,"Raw Header is : %lu \n",raw_header);
	fprintf(stderr,"seq of Header is :%lu \n ",pkt_header->seq);
	fprintf(stderr,"source id is :%d \n ",pkt_header->source_id);
	fprintf(stderr,"spec id is :%d \n ",pkt_header->spec_id);
	#endif
}

double UTC2JD(double year, double month, double day){
	double jd;
	double a;
	a = floor((14-month)/12);
	year = year+4800-a;
	month = month+12*a-3;
	jd = day + floor((153*month+2)/5)+365*year+floor(year/4)-floor(year/100)+floor(year/400)-32045;
	return jd;
}

static void *run(hashpipe_thread_args_t * args){
	demo4_input_databuf_t *db  = (demo4_input_databuf_t *)args->obuf;
	hashpipe_status_t st = args->st;
	const char * status_key = args->thread_desc->skey;

	int i, rv,input,n;
	uint64_t mcnt = 0;
	int block_idx = 0;
	unsigned long header; // 64 bit counter     
	// unsigned char data_pkt[BYTES_PER_PKT]; // save received packet
	packet_header_t pkt_header;
	unsigned long SEQ=0;
	unsigned long LAST_SEQ=0;
	unsigned long CHANNEL;
	unsigned long n_pkt_rcv; // number of packets has been received
	unsigned long pkt_loss; // number of packets has been lost
	int first_pkt=1;
	double pkt_loss_rate; // packets lost rate
	unsigned int pktsock_pkts = 0;  // Stats counter from socket packet
	unsigned int pktsock_drops = 0; // Stats counter from socket packet
	double Year, Month, Day;
	double jd;
	time_t timep;
	struct tm *p;
	struct timeval currenttime;
	time(&timep);
	p=gmtime(&timep);
	Year=p->tm_year+1900;
	Month=p->tm_mon+1;
	Day=p->tm_mday;
	jd = UTC2JD(Year, Month, Day); 
	MJD=jd+(double)((p->tm_hour-12)/24.0)
                               +(double)(p->tm_min/1440.0)
                               +(double)(p->tm_sec/86400.0)
                               +(double)(currenttime.tv_usec/86400.0/1000000.0)
								-(double)2400000.5;
	printf("MJD time of packets is %lf\n",MJD);

	uint64_t npackets = 0; //number of received packets
	int bindport = 0;
	sleep(1);

	hashpipe_status_lock_safe(&st);
	// Get info from status buffer if present (no change if not present)
	hgeti4(st.buf, "BINDPORT", &bindport);
	hputs(st.buf, status_key, "running");
	hashpipe_status_unlock_safe(&st);

	// Get pktsock from args
	struct hashpipe_pktsock * p_ps = (struct hashpipe_pktsock*)args->user_data;
	pthread_cleanup_push(free, p_ps);
	pthread_cleanup_push((void (*)(void *))hashpipe_pktsock_close, p_ps);

	// Drop all packets to date
	unsigned char *p_frame;
	while(p_frame=hashpipe_pktsock_recv_frame_nonblock(p_ps)) {
		hashpipe_pktsock_release_frame(p_frame);
	}

	// Main loop
	while (run_threads()){		
		hashpipe_status_lock_safe(&st);
		hputs(st.buf, status_key, "waiting");
		hputi4(st.buf, "NETBKOUT", block_idx);
		hputi8(st.buf,"NETMCNT",mcnt);
		hashpipe_status_unlock_safe(&st);

		hashpipe_status_lock_safe(&st);
		hputs(st.buf, status_key, "receiving");
		hashpipe_status_unlock_safe(&st);

		// receiving packets
		for(int i=0;i<NUM_PACKETS_PER_BUFFER;i++){
			do {
				p_frame = hashpipe_pktsock_recv_udp_frame_nonblock(p_ps, bindport);
			} 
			while (!p_frame && run_threads());
			if(!run_threads()) break;
 			

			#ifdef TEST_MODE
				if(npackets == 0){
					get_header(p_frame,&pkt_header);
					SEQ = pkt_header.seq;
					pkt_loss=0;
					LAST_SEQ = (SEQ-1);
				}
				npackets++;
				printf("received %lu packets\n",npackets);
				printf("number of lost packets is : %lu\n",pkt_loss);
			#endif
			hashpipe_pktsock_release_frame(p_frame);
			memcpy(db->block[block_idx].data_block+i*BYTES_PER_PAYLOAD, PKT_UDP_DATA(p_frame), BYTES_PER_PAYLOAD*sizeof(unsigned char));
			pthread_testcancel();
		}

		// Handle variable packet size!
		int packet_size = PKT_UDP_SIZE(p_frame) - 8;
		#ifdef TEST_MODE
		    for (int j=0; j<BYTES_PER_PAYLOAD*NUM_PACKETS_PER_BUFFER; j++){
				printf("Copied data[%d]: %d\n",j, db->block[block_idx].data_block[j]);
			}
			printf("packet size is: %d\n",packet_size);
			printf("packet is udp %d\n", PKT_IS_UDP(p_frame));
			printf("packet dest: %d\n", PKT_UDP_DST(p_frame));
		#endif


		// Mark block as full
		if(demo4_input_databuf_set_filled(db, block_idx) != HASHPIPE_OK) {
			hashpipe_error(__FUNCTION__, "error waiting for databuf filled call");
			pthread_exit(NULL);
		}

		db->block[block_idx].header.mcnt = mcnt;
		block_idx = (block_idx + 1) % db->header.n_block;
		mcnt++;

		/* Will exit if thread has been cancelled */
		pthread_testcancel();
	}
	pthread_cleanup_pop(1); /* Closes push(hashpipe_pktsock_close) */
	pthread_cleanup_pop(1); /* Closes push(free) */
	// Thread success!
	return THREAD_OK;
}

static hashpipe_thread_desc_t demo4_net_thread = {
	name: "demo4_net_thread",
	skey: "NETSTAT",
	init: init,
	run:  run,
	ibuf_desc: {NULL},
	obuf_desc: {demo4_input_databuf_create}
};

static __attribute__((constructor)) void ctor()
{
  register_hashpipe_thread(&demo4_net_thread);
}
