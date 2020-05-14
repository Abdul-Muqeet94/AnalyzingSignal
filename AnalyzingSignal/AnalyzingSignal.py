import matplotlib.pyplot as plt
from decimal import Decimal
import numpy as np
# from scipy.fftpack 
import scipy.signal as sci
from scipy.signal import butter, lfilter, freqz, findfreqs
from numpy.fft import fft,fftfreq,ifft
path = 'sb5.csv'
t_in_s, measurement =[],[]
with open(path, 'r') as analyzingFile:
    anaList=analyzingFile.read().splitlines()
    i=0
    while i< len(anaList)-1:
        i += 1
        val1, val2 = [float(x.strip(' "')) for x in anaList[i].split(',')]
        t_in_s.append(val1)
        measurement.append(val2)



something=sci.deconvolve(measurement,3)
print(something)

plt.figure(4)
plt.subplot(212)
plt.plot(something)


plt.figure(3)
plt.subplot(212)
plt.plot(t_in_s,measurement)

plt.figure(2)
plt.subplot(2,1,2)
powerSpectrum,frequenciesFound,time,imageAxis=plt.specgram(measurement,Fs=1000)

plt.ylabel('measurement')


# b,a =butter(btype='low')
# y=lfilter(b,a,measurement)

plt.figure(1)
plt.subplot(212)
X=fft(measurement)
plt.plot(np.abs(X))


# print (X)
# plt.plot(fr,X_m)


# plt.ylabel('measurement') 20*np.log(abs(fft_result)

# plt.figure(2)
# plt.subplot(212)
# powerSpectrum,frequenciesFound,time,imageAxis=plt.specgram(s1,Fs=100)
# plt.xlabel("time")
# plt.xlabel("Frequency")
# plt.show()
# plt.figure(3)
# plt.subplot(2,1,2)
# plt.plot(measurement+t_in_s)
# plt.xlabel("full signal")
plt.show()

# plt.plot([1,2,3,4])
# plt.ylabel('some numbers')
# plt.show()
