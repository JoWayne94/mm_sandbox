"""
Description: main.py

            A B = C

Author(s): Haitz Sáez de Ocáriz Borde, Jo Wayne Tan
"""
import math
import numpy as np
import mm
import time
import functions as fn

if __name__ == "__main__":
    rows = 4
    cols = 4

    print("Initialise matrices.")

    """ Random (dense) matrix """
    a = np.random.random((rows, cols))
    b = np.random.random((rows, cols))

    """ Manual matrix insertion """
    # a = np.identity(3)
    # b = np.identity(3)
    # a[0][2] = 5
    # b[1][0] = 3

    print("Matrix A: ", a)
    print("Matrix B: ", b)
    print("\nInitialise matrix multiplication test.")

    data = mm.matMul(a, b)

    startClockTime = time.perf_counter()

    """ Test different methods """
    print("\nComputed C: ", data.strassen(a, b))

    """ Print out the execution time """
    clockTime = round(time.perf_counter() - startClockTime, 5)
    print("\nTotal execution time =", clockTime, 's\n')
    # fn.printMemoryUsageInMB()

    print("\nCorrect C: ", np.matmul(a, b))

    print("\nSum of errors: ", np.sum(abs(data.strassen(a, b) - np.matmul(a, b))))
