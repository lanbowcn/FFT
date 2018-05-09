import csv
import os
import numpy as np
# 字典读取对应标签
filepath = r'D:\xinyin\xinyin\training\training-a\REFERENCE.csv'
with open(filepath,'r') as csvfile:
    reader = csv.reader(csvfile)
    rows= [row for row in reader]
    b=dict(rows)
print(b['a0019'],b['a0037'],b['a0049'])
