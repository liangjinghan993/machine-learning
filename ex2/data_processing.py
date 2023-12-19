# coding=utf-8
import numpy as np
import struct
import os

data_dir = "E:\机器学习\ex2\mnist_data"
train_data_dir = "train-images.idx3-ubyte"
train_label_dir = "train-labels.idx1-ubyte"
test_data_dir = "t10k-images.idx3-ubyte"
test_label_dir = "t10k-labels.idx1-ubyte"


#函数用于从指定文件加载MNIST数据。
def load_mnist(file_dir, is_images='True'):
    # 读取二进制数据
    bin_file = open(file_dir, 'rb')
    bin_data = bin_file.read()
    bin_file.close()
    # 分析文件头部
    if is_images:
        # 读取图像数据
        fmt_header = '>iiii'
        magic, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, 0)
        data_size = num_images * num_rows * num_cols
        mat_data = struct.unpack_from('>' + str(data_size) + 'B', bin_data, struct.calcsize(fmt_header))
        mat_data = np.reshape(mat_data, [num_images, num_rows, num_cols])
    else:
        # 读取标签数据
        fmt_header = '>ii'
        magic, num_images = struct.unpack_from(fmt_header, bin_data, 0)
        mat_data = struct.unpack_from('>' + str(num_images) + 'B', bin_data, struct.calcsize(fmt_header))
        mat_data = np.reshape(mat_data, [num_images])
    print('Load images from %s, number: %d, data shape: %s' % (file_dir, num_images, str(mat_data.shape)))
    return mat_data


# 调用 load_mnist 函数，加载训练集和测试集的图像和标签数据
def load_data():
    print('Loading MNIST data from files...')
    train_images = load_mnist(os.path.join(data_dir, train_data_dir), True)
    train_labels = load_mnist(os.path.join(data_dir, train_label_dir), False)
    test_images = load_mnist(os.path.join(data_dir, test_data_dir), True)
    test_labels = load_mnist(os.path.join(data_dir, test_label_dir), False)
    return train_images, train_labels, test_images, test_labels


# 将图像转换为二进制形式，并生成独热编码的标签。
def data_convert(x, y, m, k):
    x[x <= 40] = 0
    x[x > 40] = 1
    ont_hot_y = np.zeros((m, k))
    for t in range(m):
        ont_hot_y[t, y[t]] = 1
    return x, ont_hot_y


# 用于对图像矩阵进行零填充
def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=(0, 0))
    return X_pad


# 对输入图像进行归一化处理，根据不同的模式进行归一化。
def normalize(image, mode='LeNet5'):
    image -= image.min()
    image = image / image.max()
    if mode == '0p1':
        return image  # range = [0,1]
    elif mode == 'n1p1':
        image = image * 2 - 1  # range = [-1,1]
    elif mode == 'LeNet5':
        image = image * 1.275 - 0.1  # range = [-0.1,1.175]
    return image
