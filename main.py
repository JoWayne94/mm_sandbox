"""
Description: main.py

            x A^T = y

Author(s): Jo Wayne Tan, Haitz Sáez de Ocáriz Borde
"""
import math
import numpy as np
import mm
import time
import functions as fn

import torch
from torch import nn

if __name__ == "__main__":

    in_dims = 512  # equals to input features tensor number of columns
    out_dims = 128  # equals to output number of columns
    in_rows = 32  # input features tensor number of rows

    """ Initialise nn.Linear, [rows, cols] for dimensions reference """
    linear_layer = nn.Linear(in_dims, out_dims)
    w = linear_layer.weight  # [out_dims, in_dims]
    b = linear_layer.bias  # [1, out_dims]

    print("Weight tensor: ", w)
    print("Bias tensor: ", b)

    """ Random input features tensor """
    in_features = torch.randn(in_rows, in_dims)
    print("Input tensor: ", in_features)

    """ Actual mechanics, y = x W^T + b """
    w_T = torch.transpose(w, 0, 1)
    result = torch.matmul(in_features, w_T) + b
    print("Manual operations result: ", result)

    startClockTime = time.perf_counter()

    output = linear_layer(in_features)  # [in_rows, out_dims]
    print("Torch nn.Linear result: ", output)

    clockTime = round(time.perf_counter() - startClockTime, 6)
    print("\nTotal execution time for nn.Linear =", clockTime, 's\n')

    """ Random (dense) matrix """
    # a = np.random.random((rows1, cols1))
    # b = np.random.random((rows2, cols2))

    """ Manual matrix insertion """
    # a = np.identity(3)
    # b = np.identity(3)
    # a[0][2] = 5
    # b[1][0] = 3

    print("\nInitialise tensor multiplication test.")

    """ matMul object """
    data = mm.matMul(in_features, w_T)

    startClockTime = time.perf_counter()

    """ Test different methods """
    # print("\nComputed C: ", data.strassen(in_features, w_T) + b)
    naive = data.naive() + b

    """ Print out the execution time """
    clockTime = round(time.perf_counter() - startClockTime, 6)

    print("\nComputed y: ", naive)

    print("\nTotal execution time =", clockTime, 's\n')
    # fn.printMemoryUsageInMB()

    # print("\nCorrect C: ", np.matmul(a, b))
    print("\nSum of errors: ", torch.sum(abs(naive - output)))
