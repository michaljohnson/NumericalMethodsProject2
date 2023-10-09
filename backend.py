import numpy as np
import copy

# Task a)
# Implement a method, calculating the LU factorization of A.
# Input: Matrix A - 2D numpy array (e.g. np.array([[1,2],[3,4]]))
# Output: Matrices P, L and U - same shape as A each.


def lu(A):
    P = np.identity(A.shape[0])
    L = np.identity(A.shape[0])
    U = np.copy(A)
    for col_num in range(U.shape[1]):
        index = np.argmax(abs(U[col_num:, col_num])) + col_num
        if col_num != index:
            U[[col_num, index]] = np.copy(U[[index, col_num]])
            P[[col_num, index]] = np.copy(P[[index, col_num]])
            L[[col_num, index]] = np.copy(L[[index, col_num]])
            L[:, [col_num, index]] = np.copy(L[:,[index, col_num]])

        for j in range(col_num + 1, U.shape[0]):
            if (U[col_num, col_num] == 0):
                factor = 0
            else:
                factor = U[j, col_num] / U[col_num, col_num]
            U[j] -= factor * U[col_num]
            L[j, col_num] = factor
    return P, L, U



# Task b)
# Implement a method, calculating the determinant of A.
# Input: Matrix A - 2D numpy array (e.g. np.array([[1,2],[3,4]]))
# Output: The determinant - a floating number
def determinant(A):
    P, L, U = lu(A)

    def zeros_matrix(rows, cols):

        # Creates a matrix filled with zeros.
        M = []
        while len(M) < rows:
            M.append([])
            while len(M[-1]) < cols:
                M[-1].append(0.0)

        return M

    def copy_matrix(M):
        # Creates and returns a copy of a matrix.

        # Section 1: Get matrix dimensions
        rows = len(M)
        cols = len(M[0])

        # Section 2: Create a new matrix of zeros
        MC = zeros_matrix(rows, cols)

        # Section 3: Copy values of M into the copy
        for i in range(rows):
            for j in range(cols):
                MC[i][j] = M[i][j]

        return MC

    def determinant_fast(A):
        # Create an upper triangle matrix using row operations.
        # Then product of diagonal elements is the determinant

        # Section 1: Establish n parameter and copy A
        n = len(A)
        AM = copy_matrix(A)

        # Section 2: Row manipulate A into an upper triangle matrix
        for fd in range(n):  # fd stands for focus diagonal
            if AM[fd][fd] == 0:
                AM[fd][fd] = 1.0e-18  # Cheating by adding zero + ~zero
            for i in range(fd + 1, n):  # skip row with fd in it.
                crScaler = AM[i][fd] / AM[fd][fd]  # cr stands for "current row".
                for j in range(n):  # cr - crScaler * fdRow, one element at a time.
                    AM[i][j] = AM[i][j] - crScaler * AM[fd][j]
        # Section 3: Once AM is in upper triangle form ...
        product = 1.0
        for i in range(n):
            product *= AM[i][i]  # ... product of diagonals is determinant

        return product

    def diagonal(A):
        n = len(A)
        product = 1.0
        for i in range(n):
            product *= A[i][i]

        return product

    n = len(P)
    detL = diagonal(L)  # pivots are all once = 1.0
    detU = diagonal(U)  # uper triangular matrix -> product of diagonal
    detP = determinant_fast(P)
    detA = detP * detL * detU

    return detA