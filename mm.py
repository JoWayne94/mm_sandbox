"""
Description: Matrix multiplication class

Author(s): Jo Wayne Tan
"""
# from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn.functional as F


class matMul():

    def __init__(self, a, b):
        self.A = a
        self.B = b
        self.A_row_dim = a.size(dim=0)
        self.B_col_dim = b.size(dim=1)
        self.B_row_dim = b.size(dim=0)

    @staticmethod
    def split(matrix):
        """
        Splits a given matrix into quarters.
        :param matrix: nxn matrix
        :return: tuple containing 4 n/2 x n/2 matrices corresponding to a, b, c, d
        """
        row = matrix.size(dim=0)
        col = matrix.size(dim=1)
        row2, col2 = row // 2, col // 2
        return matrix[:row2, :col2], matrix[:row2, col2:], matrix[row2:, :col2], matrix[row2:, col2:]

    def naive(self):
        """
        Naive row by column multiplication
        :return: C
        """
        c = torch.zeros((self.A_row_dim, self.B_col_dim))  # result

        for i in range(self.A_row_dim):
            for k in range(self.B_col_dim):
                for j in range(self.B_row_dim):
                    c[i][k] += self.A[i][j] * self.B[j][k]

        return c

    @staticmethod
    def strassen(A, B):
        """
        Computes matrix product by divide and conquer approach, recursively.
        :return: nxn matrix, product of A and B
        """
        # Base case when size of matrices is 1x1
        if A.size(dim=0) == 1 or B.size(dim=0) == 1:
            return A * B

        n = A.size(dim=0)

        if n % 2 == 1:
            A = F.pad(A, (0, 1), mode='constant')
            B = F.pad(B, (0, 1), mode='constant')

        # Splitting the matrices into quadrants. This will be done recursively until the base case is reached.
        a, b, c, d = matMul.split(A)
        e, f, g, h = matMul.split(B)

        # Computing the 7 products, recursively (p1, p2, ..., p7)
        p1 = matMul.strassen(a, f - h)
        p2 = matMul.strassen(a + b, h)
        p3 = matMul.strassen(c + d, e)
        p4 = matMul.strassen(d, g - e)
        p5 = matMul.strassen(a + d, e + h)
        p6 = matMul.strassen(b - d, g + h)
        p7 = matMul.strassen(a - c, e + f)

        # Computing the values of the 4 quadrants of the final matrix c
        c11 = p5 + p4 - p2 + p6
        c12 = p1 + p2
        c21 = p3 + p4
        c22 = p1 + p5 - p3 - p7

        # Combining the 4 quadrants into a single matrix by stacking horizontally and vertically.
        c = torch.vstack((torch.hstack((c11, c12)), torch.hstack((c21, c22))))

        return c
