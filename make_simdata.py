#! /usr/bin/python
import time,numpy,struct,math

SampFreq = 128e6     # 128 MHz
Tone1Freq = 15e6    # 15 MHz
Tone2Freq = 1e6    # 1 MHz
Tone3Freq = 5e5     # 0.5 Mhz
Tone4Freq = 37e6     # 37 Mhz
ScaleFactor = 127   # 8 bit sample range
TimeSamples = 67108864  # time samples per frame
realPart = numpy.array([ScaleFactor * 0.1 * math.cos(2 * math.pi * Tone1Freq * i / SampFreq) for i in range(TimeSamples)])
realPart = realPart +  numpy.array([ScaleFactor * 0.2 * math.cos(2 * math.pi * Tone2Freq * i / SampFreq) for i in range(TimeSamples)])
realPart = realPart +  numpy.array([ScaleFactor * 0.2 * math.cos(2 * math.pi * Tone3Freq * i / SampFreq) for i in range(TimeSamples)])
realPart = realPart +  numpy.array([ScaleFactor * 0.2 * math.cos(2 * math.pi * Tone4Freq * i / SampFreq) for i in range(TimeSamples)])
realPart = realPart.astype(numpy.int8)

data = numpy.arange(0,128)
data = data.astype(numpy.int8)

fp=open('simdata','wb')
fp.write(realPart)
fp.close()
