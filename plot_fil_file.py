import matplotlib.pyplot as plt
import numpy 
n = 1048576*2
fp = open('data_2020-08-03_13-42-34.fil','rb')
#fp = open('5FFT_2Mchannel_128Hz_4Tone.fil')
data = numpy.fromfile(fp,dtype=numpy.float32)
#plt.plot(data[4*n:5*n])
plt.plot(data)
plt.show()
#for i in range(0,len(data)/PKTSIZE):

#while line!='HEADER_END':
 #   print(line)
 #   line=fp.readline()
