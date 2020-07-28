#! /usr/bin/python
import time,numpy,struct,math

SampFreq = 32e6     # 32 MHz
Tone1Freq = 15e6    # 15 MHz
Tone2Freq = 1e6    # 7 MHz
Tone3Freq = 14e6     # 6 Mhz
ScaleFactor = 127   # 8 bit sample range
TimeSamples = 1024   # time samples per frame
realPart = numpy.array([ScaleFactor * 0.1 * math.cos(2 * math.pi * Tone1Freq * i / SampFreq) for i in range(TimeSamples)])
realPart = realPart +  numpy.array([ScaleFactor * 0.2 * math.cos(2 * math.pi * Tone2Freq * i / SampFreq) for i in range(TimeSamples)])
realPart = realPart +  numpy.array([ScaleFactor * 0.2 * math.cos(2 * math.pi * Tone3Freq * i / SampFreq) for i in range(TimeSamples)])
realPart = realPart.astype(numpy.int8)

data = numpy.arange(0,32)
data = data.astype(numpy.int8)

fp=open('simdata','wb')
fp.write(data)
fp.close()
