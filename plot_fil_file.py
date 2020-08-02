import matplotlib.pyplot as plt
import numpy 

fp = open('data_2020-08-02_15-50-25.fil','rb')
#fp = open('5FFT_2Mchannel_128Hz_4Tone.fil')
data = numpy.fromfile(fp,dtype=numpy.float32)
plt.plot(data)
plt.show()
#for i in range(0,len(data)/PKTSIZE):

#while line!='HEADER_END':
 #   print(line)
 #   line=fp.readline()
