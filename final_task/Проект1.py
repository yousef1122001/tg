import multiprocessing
import numpy as np

def matrix_multiply_worker(data):
    A, B, row = data
    return [sum(a * b for a, b in zip(row, col)) for col in zip(*B)]

def parallel_matrix_multiply(A, B):
    with multiprocessing.Pool() as pool:
        result = pool.map(matrix_multiply_worker, [(A, B, row) for row in A])
    return result

if __name__ == "__main__":
    A = np.random.randint(0, 10, (4, 4))
    B = np.random.randint(0, 10, (4, 4))
    print("Matrix A:")
    print(A)
    print("Matrix B:")
    print(B)
    result = parallel_matrix_multiply(A, B)
    print("Result of A x B:")
    print(np.array(result))