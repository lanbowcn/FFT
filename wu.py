import numpy as np
import cv2 as cv
import os
import h5py

np.set_printoptions(threshold=np.inf)

def create_h5_file(h5_save_path, data, label):
    '''
    创建h5格式文件
    :param h5_save_path: h5文件路径
    :param data: ndarray类型数据域
    :param label: ndarray类型标签域
    :return:
    '''
    with h5py.File(h5_save_path, 'w') as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('label', data=label)
        print('finished!')

def generate_h5_file(cover_path, stego_path, train_save_path, test_save_path):
    data = []
    label = []
    # cover读取
    for (root, dir, files) in os.walk(cover_path):
        for file in files:
            filepath = root + '/' + file
            # image = cv.imread(filepath, cv.IMREAD_GRAYSCALE)

            image = cv.imread(filepath)
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            hei, width = image.shape[0], image.shape[1]
            im_input = image.reshape((hei, width, 1))
            im_input = im_input * 1.0
            im_input = im_input.astype('float32')
            # cover类别
            im_label = [0.0, 1.0]

            data.append(im_input)
            label.append(im_label)

    # stego读取
    for (root, dir, files) in os.walk(stego_path):
        for file in files:
            filepath = root + '/' + file
            # image = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
            image = cv.imread(filepath)
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            hei, width = image.shape[0], image.shape[1]
            im_input = image.reshape((hei, width, 1))
            im_input = im_input * 1.0
            im_input = im_input.astype('float32')
            # stagan类别
            im_label = [1.0, 0.0]
            data.append(im_input)
            label.append(im_label)

    data = np.asarray(data)
    label = np.asarray(label)
    # 划分数据集 80%训练 20%测试
    perm = np.arange(data.shape[0])
    np.random.shuffle(perm)
    # create_h5_file(train_save_path, data, label)
    # 训练数据集
    create_h5_file(train_save_path, data[np.ix_(perm[0:16000])], label[np.ix_(perm[0:16000])])
    # 测试数据集
    create_h5_file(test_save_path, data[np.ix_(perm[16000:20000])], label[np.ix_(perm[16000:20000])])



# h5文件读取
def read_h5_file(path):
    with h5py.File(path, 'r') as hf:
        hf_data = hf.get('data')
        data = np.array(hf_data)
        hf_label = hf.get('label')
        label = np.array(hf_label)
        return data, label

# 数据迭代器
def data_iterator(data, label, batch_size):
    num_examples = data.shape[0] // 2
    num_batch = num_examples // batch_size
    num_total = num_batch * batch_size
    while True:
        perm = np.arange(num_examples // 2)
        np.random.shuffle(perm)
        shuf_data = data[perm]
        shuf_data.append(data[perm + 10000])
        shuf_label = label[perm]
        shuf_label.append(label[perm + 10000])

        for i in range(0, num_total // 2, batch_size // 2):
            batch_data = shuf_data[i:i+batch_size]
            batch_label = shuf_label[i:i+batch_size]
            x = i + 10000
            batch_data.append(shuf_data[x:x+batch_size])
            batch_label.append(shuf_label[x:x+batch_size])
            yield batch_data, batch_label


if __name__ == '__main__':
    cover_path = r'../datasets/traindatasets/wow_0.4_256_256_cover'
    stego_path = r'../datasets/traindatasets/wow_0.4_256_256_stego'
    train_save_path = r'../datasets/h5_files/train_256_256_04_cover_stagan_wow_16000.h5'
    test_save_path = r'../datasets/h5_files/test_256_256_04_cover_stagan_wow_4000.h5'
    path = r'../datasets/h5_files/train_256_256.h5'
    # generate_h5_file(cover_path, stego_path, train_save_path, test_save_path)
    data,label = read_h5_file(path)
    # print(da)
    train_mini_batch = data_iterator(data, label, 64)
    mini_batch_train_data, mini_batch_train_label = train_mini_batch.__next__()
    print(mini_batch_train_data.shape)
    # print(data[0])
