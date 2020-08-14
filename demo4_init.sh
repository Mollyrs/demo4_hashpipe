#!/bin/bash
#./send_packet.py &
#hashpipe -p /home/peix/local/lib/demo4_hashpipe -I 0  demo4_net_thread  demo4_gpu_thread  demo4_output_thread 
#hashpipe -p ./demo4_hashpipe -I 0 -o BINDHOST="eth4" demo4_net_thread  demo4_gpu_thread  demo4_output_thread 
export PGPLOT_DIR=/usr/local/pgplot/
ipcrm -M 0x80010af7
ipcrm -M 0x80010af6
ipcrm -M 0x80010001
ipcrm -M 0x80010002
#hashpipe -p ./demo4_hashpipe -I 0 -o BINDHOST="ens6f0" demo4_net_thread  demo4_gpu_thread  demo4_output_thread 
#hashpipe -p ./demo4_hashpipe -I 0 -o BINDHOST="lo" demo4_net_thread  demo4_gpu_thread  demo4_output_thread
hashpipe -p ./demo4_hashpipe -I 0 -o BINDHOST="ens6f0" demo4_net_thread  demo4_gpu_thread  demo4_output_thread 
