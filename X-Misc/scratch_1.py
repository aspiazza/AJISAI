from icecream import ic
from numba import jit, cuda
import math
import time


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
