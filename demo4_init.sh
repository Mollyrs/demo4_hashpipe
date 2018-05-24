#!/bin/bash
#./send_packet.py &
#hashpipe -p /home/peix/local/lib/demo4_hashpipe -I 0  demo4_net_thread  demo4_gpu_thread  demo4_output_thread 
#hashpipe -p ./demo4_hashpipe -I 0 -o BINDHOST="eth4" demo4_net_thread  demo4_gpu_thread  demo4_output_thread 
hashpipe -p ./demo4_hashpipe -I 0 -o BINDHOST="eth5" demo4_net_thread  demo4_gpu_thread  demo4_output_thread 
