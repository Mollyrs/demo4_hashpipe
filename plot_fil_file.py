import matplotlib.pyplot as plt
import numpy 

fp = open('data_2020-07-31_15-59-58.fil','rb')
data = numpy.fromfile(fp,dtype=numpy.float32)
plt.plot(data)
plt.show()
#for i in range(0,len(data)/PKTSIZE):

#while line!='HEADER_END':
 #   print(line)
 #   line=fp.readline()
