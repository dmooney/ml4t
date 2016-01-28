import numpy as np

def foo(x):
    return 1 + x ** 2

a = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
b = [0.2, 0.2, 0.2, 0.2, 0.2]
print(foo(b))
print(foo(a))
