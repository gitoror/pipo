import numpy as np


rho = np.array([[2, 3, 4], [5, 6, 7]])
w = np.array([1, 2, 3])
print(np.tensordot(rho, w, axes=0))

a = np.ones((2, 3))
print(a)
