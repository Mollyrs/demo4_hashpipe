#! /usr/bin/python
import time,numpy,struct,math

SampFreq = 2048e6     
Tone1Freq = 50e6    
Tone2Freq = 100e6    
Tone3Freq = 200e6     
Tone4Freq = 400e6
Tone5Freq = 700e6     

ScaleFactor = 127   # 8 bit sample range
TimeSamples = 67108864  # time samples per frame
realPart = numpy.array([ScaleFactor * 0.2 * math.cos(2 * math.pi * Tone1Freq * i / SampFreq) for i in range(TimeSamples)])
realPart = realPart +  numpy.array([ScaleFactor * 0.2 * math.cos(2 * math.pi * Tone2Freq * i / SampFreq) for i in range(TimeSamples)])
realPart = realPart +  numpy.array([ScaleFactor * 0.2 * math.cos(2 * math.pi * Tone3Freq * i / SampFreq) for i in range(TimeSamples)])
realPart = realPart +  numpy.array([ScaleFactor * 0.2 * math.cos(2 * math.pi * Tone4Freq * i / SampFreq) for i in range(TimeSamples)])
realPart = realPart +  numpy.array([ScaleFactor * 0.2 * math.cos(2 * math.pi * Tone5Freq * i / SampFreq) for i in range(TimeSamples)])
realPart = realPart + numpy.random.random_integers(-64, 64, TimeSamples) #noise
realPart = realPart.astype(numpy.int8)

data = numpy.arange(0,128)
data = data.astype(numpy.int8)

fp=open('simdata','wb')
fp.write(realPart)
fp.close()
