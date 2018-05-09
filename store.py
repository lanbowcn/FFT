import numpy as np
import os
import csv


#文件操作：测试提取一定规模的训练样本
filepath=r'D:\xinyin\xinyin\training\training-a\spect'
for(root,dir,files) in os.walk(filepath):
    # 加载 录音-标签 字典
    with open(os.path.join(root,'REFERENCE.csv'), 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
        labedict = dict(rows)
    for i in range(500):
        for file in files:
            matrixpath = os.path.join(root, file)
            my_matrix = np.loadtxt(open(matrixpath, "rb"), delimiter=",", skiprows=0)
            # print(os.path.join(root,file))输出带根目录的文件路径
            wav = file.split('-', 2)[0]
            label = np.array[]

