import matplotlib.pyplot as plt
from decimal import Decimal
import numpy as np
from scipy.fftpack import fft,ifft
import scipy.signal as signal

# seed the pseudorandom number generator
from random import seed
from random import random


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

def plot_figure(figure_no, plt, x_data, x_label, y_data=None, y_label=None):
    plt.figure(figure_no)
    plt.subplot(2, 1, 2)
    plt.xlabel(x_label)
    if(y_label):
        plt.ylabel(y_label)
    if(y_data):
        plt.plot(x_data,y_data, 'b-')
    else:
        plt.plot(x_data, 'b-')

seed(1)

i=240
randArray=[]
while(i>0):
    randArray.append(random()-0.5)
    i-=1


def low_pass_filter(measurement):
    N = 3  # Filter order
    Wn = 0.03  # Cutoff frequency
    B, A = signal.butter(N, Wn, output='ba')
    return signal.filtfilt(B, A, measurement)

coef_1 = np.polyfit(t_in_s,measurement,1)
poly_1 =[]
for entity in t_in_s:
    poly_1.append(coef_1[1]+coef_1[0]*entity)

coef_3 = np.polyfit(t_in_s,measurement,3)
poly_3 =[]

for entity in t_in_s:
    poly_3.append(coef_3[3]+coef_3[2]*entity+coef_3[1]*entity*entity+coef_3[0]*entity*entity*entity)

proj_0 = []
i = 0
for m in measurement:
    proj_0.append(m-poly_1[i])
    i = i + 1

fft_p = fft(proj_0)
fft_p_c = np.copy(fft_p)

fft_p
plot_figure(1,plt,fft_p,"fft")
for i in range(3, 238):
    fft_p[i] = 0

sign_rec = np.real(ifft(fft_p))

proj_1 = []
i = 0
for m in proj_0:
    proj_1.append(m-sign_rec[i])
    i = i + 1

fft_p1 = fft(proj_1)

i = 0
real_sig = []
for m in poly_1:
    real_sig.append(m+sign_rec[i])
    i = i + 1

i = 0
noise_sig = []
for m in measurement:
    noise_sig.append(m-real_sig[i])
    i = i + 1

plot_figure(5,plt,abs(fft(noise_sig)),"fft_ noise")
#  plot_figure(1,plt,abs(fft_p1),"fft_proj1")

plot_figure(3,plt,t_in_s,"time",poly_1,"poly_1")
plot_figure(3,plt,t_in_s,"time",real_sig,"real")
plot_figure(4,plt,fft_p,"noise_sig")

# plot_figure(2,plt,t_in_s,"time",proj_0,"poly")
# plot_figure(2, plt,t_in_s,"time", list(sign_rec),'sign rec')

smooth_data = low_pass_filter(proj_0)
# plot_figure(2, plt, smooth_data, 'smooth data')

recovered, remainder = signal.deconvolve(smooth_data, 3)
# plot_figure(3, plt, signal, 'deconvolve')

# print(recovered)
# print(remainder)
signal.fftconvolve(measurement,t_in_s)


ff = fft(smooth_data)
yf = fft(recovered)
xf = fft(remainder)
w, H = signal.freqz(yf)
freq = signal.findfreqs([-1, 1], remainder, 250)

plt.show()
