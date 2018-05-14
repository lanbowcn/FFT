import numpy as np
import os
import csv

# data=[]
# label=[]
# #文件操作：测试提取一定规模的训练样本
# filepath=r'D:\xinyin\xinyin\training\training-a\spect'
# for(root,dir,files) in os.walk(filepath):
#     # 加载 录音-标签 字典
#     with open(os.path.join(r'D:\xinyin\xinyin\training\training-a\REFERENCE.csv'), 'r') as csvfile:
#         reader = csv.reader(csvfile)
#         rows = [row for row in reader]
#         labedict = dict(rows)
#         for file in files[0:50]:
#             matrixpath = os.path.join(root, file)
#             filename = os.path.splitext(file)[0]
#             my_matrix = np.loadtxt(open(matrixpath, "rb"), delimiter=",", skiprows=0)
#             # print(os.path.join(root,file))输出带根目录的文件路径
#
#             data.append(my_matrix)
#
#             wav = file.split('-', 2)[0]
#             label.append(labedict[wav])
#             # print(label[1])
#
# print(label)


def readdata(batchnumber,batchsize):
    # 文件操作：测试提取一定规模的训练样本
    data=[]
    label=[]
    filepath = r'D:\xinyin\xinyin\training\training-a\spect'
    for (root, dir, files) in os.walk(filepath):
        # 加载 录音-标签 字典
        with open(os.path.join(r'D:\xinyin\xinyin\training\training-a\REFERENCE.csv'), 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader]
            labedict = dict(rows)
            for file in files[(batchnumber-1)*batchsize:batchnumber*batchsize]:
                matrixpath = os.path.join(root, file)
                filename = file.split('-', 2)[0]
                my_matrix = np.loadtxt(open(matrixpath, "rb"), delimiter=",", skiprows=0)
                # print(os.path.join(root,file))输出带根目录的文件路径
                data.append(my_matrix)
                label.append(labedict[filename])
    return data,label


for i in range(1,2):
    BATCH_SIZE = 64
    data, label = readdata(i, BATCH_SIZE)
    data = np.asarray(data)
    print(type(data))
    data = np.reshape(data, (64, 101, 99, 1))
    # data.reshape((64, 101, 99 ,1))
    print(data.shape)