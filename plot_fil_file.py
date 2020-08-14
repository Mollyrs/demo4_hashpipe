import matplotlib.pyplot as plt
import numpy 
n = 1048576*32
fp15=open('data_2020-08-13_18-24-56.fil','rb')
data=numpy.fromfile(fp15,dtype=numpy.float32)
#data15=data[1:n-1]


plt.plot(data)
#labels = [0.00, 6.25, 12.50, 18.75, 25.00, 31.25, 37.50, 43.75, 50.00, 56.25, 62.50, 68.75, 75.00, 81.25, 87.50, 93.75, 100.000]
#plt.xticks(numpy.arange(0, n+1, 2*1048576), labels)
#plt.xlabel('Frequency (MHz)')
#plt.ylim(0,7000000)
#plt.ticklabel_format(axis='y', style='sci',scilimits=(1,2))
#plt.title('100 MHz Spectrum, 15 MHz 100 mVpp Tone')



plt.show()

