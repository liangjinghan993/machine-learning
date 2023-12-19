import numpy as np
from data_processing import zero_pad


# Numpy version: compute with np.tensordot()
def conv_forward(A_prev, W, b, hyper_parameters):
    """
    输入:
A_prev: 前一层的输出激活值，形状为 (m, n_H_prev, n_W_prev, n_C_prev)。
W: 权重，形状为 (f, f, n_C_prev, n_C)。
b: 偏置，形状为 (1, 1, 1, n_C)。
hyper_parameters: 包含 "stride" 和 "pad" 的字典。
输出:
Z: 卷积输出，形状为 (m, n_H, n_W, n_C)。
cache: 包含 (A_prev, W, b, hyper_parameters) 的元组，用于后续反向传播。
描述:
使用给定的输入和参数计算卷积层的前向传播。
根据 hyper_parameters 中指定的填充方式使用零填充（使用 zero_pad 函数）。
遍历输出体积执行卷积操作，并使用 np.tensordot 计算加权和。
返回结果和后续反向传播所需的缓存。
    """
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape

    stride = hyper_parameters["stride"]
    pad = hyper_parameters["pad"]

    n_H = int((n_H_prev + 2 * pad - f) / stride + 1)
    n_W = int((n_W_prev + 2 * pad - f) / stride + 1)

    # Initialize the output volume Z with zeros.
    Z = np.zeros((m, n_H, n_W, n_C))
    A_prev_pad = zero_pad(A_prev, pad)
    for h in range(n_H):  # loop over vertical axis of the output volume
        for w in range(n_W):  # loop over horizontal axis of the output volume
            # Use the corners to define the (3D) slice of a_prev_pad.
            A_slice_prev = A_prev_pad[:, h * stride:h * stride + f, w * stride:w * stride + f, :]
            # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron.
            Z[:, h, w, :] = np.tensordot(A_slice_prev, W, axes=([1, 2, 3], [0, 1, 2])) + b

    assert (Z.shape == (m, n_H, n_W, n_C))
    cache = (A_prev, W, b, hyper_parameters)
    return Z, cache


# Numpy version: compute with np.dot
def conv_backward(dZ, cache):
    """
    输入:
dZ: 相对于卷积层输出 (Z) 的损失梯度，形状为 (m, n_H, n_W, n_C)。
cache: 从前向传播步骤得到的缓存。
输出:
dA_prev: 相对于卷积层输入 (A_prev) 的损失梯度，形状为 (m, n_H_prev, n_W_prev, n_C_prev)。
dW: 相对于卷积层权重 (W) 的损失梯度，形状为 (f, f, n_C_prev, n_C)。
db: 相对于卷积层偏置 (b) 的损失梯度，形状为 (1, 1, 1, n_C)。
描述:
计算卷积层的反向传播，以获得输入、权重和偏置的梯度。
利用链式法则计算梯度。
遍历输出体积以使用卷积操作的转置计算梯度。
根据需要应用零填充（使用 zero_pad 函数）。
返回计算得到的梯度。
    """
    (A_prev, W, b, hyper_parameters) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    (m, n_H, n_W, n_C) = dZ.shape
    stride = hyper_parameters["stride"]
    pad = hyper_parameters["pad"]

    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    if pad != 0:
        A_prev_pad = zero_pad(A_prev, pad)
        dA_prev_pad = zero_pad(dA_prev, pad)
    else:
        A_prev_pad = A_prev
        dA_prev_pad = dA_prev

    for h in range(n_H):  # loop over vertical axis of the output volume
        for w in range(n_W):  # loop over horizontal axis of the output volume
            # Find the corners of the current "slice"
            vert_start, horiz_start = h * stride, w * stride
            vert_end, horiz_end = vert_start + f, horiz_start + f

            # Use the corners to define the slice from a_prev_pad
            A_slice = A_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end, :]

            # Update gradients for the window and the filter's parameters
            dA_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end, :] += np.transpose(np.dot(W, dZ[:, h, w, :].T), (3, 0, 1, 2))

            dW += np.dot(np.transpose(A_slice, (1, 2, 3, 0)), dZ[:, h, w, :])
            db += np.sum(dZ[:, h, w, :], axis=0)

    # Set dA_prev to the unpadded dA_prev_pad
    dA_prev = dA_prev_pad if pad == 0 else dA_prev_pad[:, pad:-pad, pad:-pad, :]

    # Making sure your output shape is correct
    assert (dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

    return dA_prev, dW, db

"""
输入:
dZ: 相对于卷积层输出 (Z) 的损失梯度，形状为 (m, n_H, n_W, n_C)。
cache: 从前向传播步骤得到的缓存。
输出:
dA_prev: 相对于卷积层输入 (A_prev) 的损失梯度，形状为 (m, n_H_prev, n_W_prev, n_C_prev)。
dW: 相对于卷积层权重 (W) 的损失梯度，形状为 (f, f, n_C_prev, n_C)。
描述:
计算相对于卷积层输入和权重的损失函数的二阶导数（Hessian）。
类似于 conv_backward，但通过在权重更新中使用元素-wise 平方（np.power）来整合二阶导数。
返回计算得到的二阶梯度。
"""
def conv_SDLM(dZ, cache):
    (A_prev, W, b, hparameters) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    (m, n_H, n_W, n_C) = dZ.shape
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    if pad != 0:
        A_prev_pad = zero_pad(A_prev, pad)
        dA_prev_pad = zero_pad(dA_prev, pad)
    else:
        A_prev_pad = A_prev
        dA_prev_pad = dA_prev

    for h in range(n_H):  # loop over vertical axis of the output volume
        for w in range(n_W):  # loop over horizontal axis of the output volume
            # Find the corners of the current "slice"
            vert_start, horiz_start = h * stride, w * stride
            vert_end, horiz_end = vert_start + f, horiz_start + f

            # Use the corners to define the slice from a_prev_pad
            A_slice = A_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end, :]

            # Update gradients for the window and the filter's parameters
            dA_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end, :] += np.transpose(
                np.dot(np.power(W, 2), dZ[:, h, w, :].T), (3, 0, 1, 2))

            dW += np.dot(np.transpose(np.power(A_slice, 2), (1, 2, 3, 0)), dZ[:, h, w, :])
    # Set dA_prev to the unpaded dA_prev_pad
    dA_prev = dA_prev_pad if pad == 0 else dA_prev_pad[:, pad:-pad, pad:-pad, :]
    assert (dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    return dA_prev, dW