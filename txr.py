import math
import os
import numpy as np
from scipy import signal
from scipy.io import wavfile
import csv
#提取时频特征
def get_spectrogram(segment):#, audio):
    segment=segment/ (np.sqrt(np.sum(audio*audio)/len(audio)))
    f, t, Sxx = signal.spectrogram(segment, fs=2000, nperseg=200, noverlap=100,mode='magnitude')
    return Sxx

dictpath=r'D:\xinyin\xinyin\training\training-a\REFERENCE.csv'
with open(dictpath,'r') as csvfile:
    reader = csv.reader(csvfile)
    rows= [row for row in reader]
    labeldict=dict(rows)

labelpath=r'D:\xinyin\xinyin\training\training-a\REFERENCE2.csv'

wavpath = r'D:\xinyin\xinyin\training\training-a'
for (root, dir, files) in os.walk(wavpath):
    for file in files:
        if os.path.splitext(file)[1]=='.wav':
            #读取wav文件，进行频谱分析，分段存储
            filepath = root + '\\' + file
            filename = os.path.splitext(file)[0]
            audio = wavfile.read(filepath)[1].astype('float')
            size = len(audio)
            seg_num = int(math.floor(size / 10000))
            start_pos = int(np.ceil(size / 2) - 5000 * seg_num)
            for i in range(seg_num):
                segment = audio[start_pos + 10000 * i:start_pos + 10000 * (i + 1)]
                sxx = get_spectrogram(segment)  # , audio)
                # sxx=sxx.reshape(1,1,101,99)
                print(sxx.shape)
                np.savetxt(root+'\\'+'spect'+'\\'+filename+'-'+str(i)+'.csv', sxx, fmt='%f', delimiter=',')

