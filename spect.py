import os
import numpy as np
wavpath = r'D:\xinyin\xinyin\training\training-a\spect'
for (root, dir, files) in os.walk(wavpath):
    for file in files:
        if os.path.splitext(file)[1]=='.csv':
            #读取wav文件，进行频谱分析，分段存储
            filepath = root + '\\' + file
            filename = os.path.splitext(file)[0]
            audio = wavfile.read(filepath)[1].astype('float')
            size = len(audio)
            seg_num = int(math.floor(size / 10000))
            start_pos = int(np.ceil(size / 2) - 5000 * seg_num)
            for i in range(seg_num):
                segment = audio[start_pos + 10000 * i:start_pos + 10000 * (i + 1)]
my_matrix = np.loadtxt(open(datadir,"rb"),delimiter=",",skiprows=0)
print(my_matrix)