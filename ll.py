import numpy as np

data = np.array([[1, 2, 4],[4 ,2, 4],[3, 4, 1]])
perm = np.arange(3)
np.random.shuffle(perm)
shuf_data = data[perm]
print(shuf_data)

