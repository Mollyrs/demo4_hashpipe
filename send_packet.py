#! /usr/bin/python
import socket
import time,numpy,struct
PKTSIZE = 16
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
addr = ("127.0.0.1",12345)
addr2 = ("127.0.0.2",5009)
fp=open('simdata','r')
data=numpy.fromstring(fp.read(),dtype='b')
for i in range(0,len(data)/(PKTSIZE*2)):
	n=sock.sendto((data[i*PKTSIZE:(i+1)*PKTSIZE]),addr)
	print "send %d bytes of number %d packets to local address! "%(n,i)
	n2=sock.sendto((data[i*PKTSIZE+16:(i+1)*PKTSIZE+16]),addr2)
	print "send %d bytes of number %d packets to other local address! "%(n2,i)
	time.sleep(0.01) #0.000001 sec(100ns) no packets loss
fp.close()
