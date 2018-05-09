from __future__ import print_function
import sys
import math
from keras.models import load_model
from scipy.io import wavfile
from scipy.io import loadmat
from scipy import signal
import numpy as np

if len(sys.argv) != 2:
    print('missing record id')
    exit(1)

#=======get 101*99 spectrogram features
def get_spectrogram(segment):#, audio):
    segment=segment/ (np.sqrt(np.sum(audio*audio)/len(audio)))
    f, t, Sxx = signal.spectrogram(segment, fs=2000, nperseg=200, noverlap=100,mode='magnitude')
    return Sxx

#load CNN model
cnn_model=load_model('cnn_model.h5')

#read in file
audio = wavfile.read(sys.argv[1] + '.wav')[1].astype('float')
# audio = loadmat('audio_file.mat',squeeze_me=True)
# audio = audio['audio_file']

#predict for each segment
size = len(audio)
seg_num = int(math.floor(size/10000))
start_pos = int(np.ceil(size/2)-5000*seg_num)
vals = np.zeros(seg_num)
labels = np.zeros(seg_num)
count_normal = 0
count_abnormal = 0
for i in range(seg_num):
    segment = audio[start_pos+10000*i:start_pos+10000*(i+1)]
    sxx = get_spectrogram(segment)#, audio)
    sxx=sxx.reshape(1,1,101,99)
    vals[i] = cnn_model.predict(sxx)
# print(vals)
threshold = 0.4
#======generate labels
for i,val in enumerate(vals):
    if val >= threshold: labels[i]=1; count_abnormal+=1
    else: labels[i]=0; count_normal+=1
if count_abnormal > count_normal:
    result = 1
elif count_abnormal < count_normal:
    result = -1
elif np.mean(vals) > threshold:
    result = 1
else:
    result = -1
# print(result)

with open('answers.txt', 'a') as output_file:
    output_file.write(sys.argv[1] + ',' + str(result) + '\n');

