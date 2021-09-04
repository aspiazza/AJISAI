from icecream import ic
from numba import jit, cuda
import math
import time

'''
# Wrapper function
def timeit(func):
    def measure_time(*args, **kw):
        start_time = time.time()
        result = func(*args, **kw)
        print("Processing time of %s(): %.10f seconds."
              % (func.__qualname__, time.time() - start_time))
        return result

    return measure_time


@timeit
@jit(nopython=True)  # Best performance
def hypot(x, y):
    return math.sqrt(x * x + y * y)


@timeit
def hypot2(x, y):
    return math.sqrt(x * x + y * y)


# Numba function
ic(hypot(3.0, 4.0))
ic(hypot2(3.0, 4.0))

# https://www.youtube.com/watch?v=x58W9A2lnQc&list=PLKLdcrR-hyUYVsIrANIxQzBIRaAzjh1ap&index=2
'''

'''
import numpy as np
from numba import vectorize


@vectorize(['float32(float32, float32)'], target='cuda')
def add(a, b):
    return a + b


# Initialize arrays
N = 100000
A = np.ones(N, dtype=np.float32)
B = np.ones(A.shape, dtype=A.dtype)
C = np.empty_like(A, dtype=A.dtype)

# Add arrays on GPU
C = add(A, B)
'''