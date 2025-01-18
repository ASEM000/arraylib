import time
import numpy as np
import arraylib as al

N = 50


def timeit(func, *args, **kwargs):
    start_time = time.time()
    for _ in range(N):
        result = func(*args, **kwargs)
    end_time = time.time()
    return (end_time - start_time) / N, result


if __name__ == "__main__":
    size = 1_000
    shape = (size, size)
    shape_1d = (size * size,)

    print("creating arrays...")
    al_array = al.zeros(shape)
    np_array = np.zeros(shape)

    print("\nbenchmarking array creation...")
    al_time, _ = timeit(al.zeros, shape)
    np_time, _ = timeit(np.zeros, shape)
    print(f"al array creation time: {al_time:.6f} seconds")
    print(f"np array creation time: {np_time:.6f} seconds")

    print("\nbenchmarking element-wise addition...")
    al_time, _ = timeit(lambda: al_array + 1.0)
    np_time, _ = timeit(lambda: np_array + 1.0)
    print(f"al array element-wise addition time: {al_time:.6f} seconds")
    print(f"np array element-wise addition time: {np_time:.6f} seconds")

    print("\nbenchmarking array-array addition...")
    al_time, _ = timeit(lambda: al_array + al_array)
    np_time, _ = timeit(lambda: np_array + np_array)
    print(f"al array element-wise addition time: {al_time:.6f} seconds")
    print(f"np array element-wise addition time: {np_time:.6f} seconds")

    print("\nbenchmarking matrix multiplication...")
    al_time, _ = timeit(al.matmul, al_array, al_array)
    np_time, _ = timeit(np.matmul, np_array, np_array)
    print(f"al array matrix multiplication time: {al_time:.6f} seconds")
    print(f"np array matrix multiplication time: {np_time:.6f} seconds")

    print("\nbenchmarking reduction operations...")

    al_time, _ = timeit(al.reduce_sum, al_array, dims=(0,))
    np_time, _ = timeit(np.sum, np_array, axis=(0,))
    print(f"al array sum reduction time: {al_time:.6f} seconds")
    print(f"np array sum reduction time: {np_time:.6f} seconds")

    al_time, _ = timeit(al.reduce_max, al_array, dims=(0,))
    np_time, _ = timeit(np.max, np_array, axis=(0,))
    print(f"al array max reduction time: {al_time:.6f} seconds")
    print(f"np array max reduction time: {np_time:.6f} seconds")

    al_time, _ = timeit(al.reduce_min, al_array, dims=(0,))
    np_time, _ = timeit(np.min, np_array, axis=(0,))
    print(f"al array min reduction time: {al_time:.6f} seconds")
    print(f"np array min reduction time: {np_time:.6f} seconds")

    print("\nbenchmarking reshaping...")
    al_time, _ = timeit(al.reshape, al_array, shape_1d)
    np_time, _ = timeit(np.reshape, np_array, shape_1d)
    print(f"al array reshaping time: {al_time:.6f} seconds")
    print(f"np array reshaping time: {np_time:.6f} seconds")

    print("\nbenchmarking transposing...")
    al_time, _ = timeit(al.transpose, al_array, (1, 0))
    np_time, _ = timeit(np.transpose, np_array, (1, 0))
    print(f"al array transposing time: {al_time:.6f} seconds")
    print(f"np array transposing time: {np_time:.6f} seconds")

    print("\nbenchmarking ravel...")
    al_time, _ = timeit(al.ravel, al_array)
    np_time, _ = timeit(np.ravel, np_array)
    print(f"al array ravel time: {al_time:.6f} seconds")
    print(f"np array ravel time: {np_time:.6f} seconds")

    print("\nbenchmarking dot product...")
    al_array_1d = al.reshape(al_array, shape_1d)
    np_array_1d = np.reshape(np_array, shape_1d)
    al_time, _ = timeit(al.dot, al_array_1d, al_array_1d)
    np_time, _ = timeit(np.dot, np_array_1d, np_array_1d)
    print(f"al array dot product time: {al_time:.6f} seconds")
    print(f"np array dot product time: {np_time:.6f} seconds")
