import numpy as np

# data = np.array([[1, 2, 4],[4 ,2, 4],[3, 4, 1]])
# perm = np.arange(3)
# np.random.shuffle(perm)
# shuf_data = data[perm]
# print(shuf_data)

def read():
    My_matrix=np.matrix([[1,2,3],[2,3,4],[3,4,5]])
    My_matrix2=np.matrix([[1,2,3],[2,3,4],[3,4,5]])
    b=[]
    b.append(My_matrix)
    b.append(My_matrix2)
    return b

# print(b[0].shape)
for i in range(2):
    a=read()
    print(i)
    print(a)