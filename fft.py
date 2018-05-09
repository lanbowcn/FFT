import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
x = np.random.random(10000)
f, t, Sxx = signal.spectrogram(x, fs=2000, nperseg=200, noverlap=100,mode='magnitude')
sxx=Sxx.reshape(1,1,101,99)
t= np.random.rand(2,3)
print(t)
print(Sxx.shape)
y = np.random.random(100)
print(y)
segment = y[:20]
print(segment)
segment=segment/ (np.sqrt(np.sum(y*y)/len(y)))
print(segment)