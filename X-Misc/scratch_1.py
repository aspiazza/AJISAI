from icecream import ic
from numba import jit
import math
import time


def timeit(func):
    def measure_time(*args, **kw):
        start_time = time.time()
        result = func(*args, **kw)
        print("Processing time of %s(): %.9f seconds."
              % (func.__qualname__, time.time() - start_time))
        return result

    return measure_time


@timeit
@jit
def hypot(x, y):
    return math.sqrt(x * x + y * y)


@timeit
def hypot2(x, y):
    return math.sqrt(x * x + y * y)


# Numba function
ic(hypot(3.0, 4.0))
ic(hypot2(3.0, 4.0))

# https://towardsdatascience.com/speed-up-your-algorithms-part-2-numba-293e554c5cc1
# https://www.youtube.com/watch?v=x58W9A2lnQc&list=PLKLdcrR-hyUYVsIrANIxQzBIRaAzjh1ap&index=2
