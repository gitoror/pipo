import numpy as np

SIZE_X, SIZE_Y = 2, 3
LATTICE_Q = 4
ex = np.array([1, 2, 3, 4])
u = np.array([[1, 2, 3], [4, 5, 6]])

# print(u)
# print("")
# print(ex)
# print("")
# # print(np.ones((SIZE_X, SIZE_Y, LATTICE_Q)))
# print(np.tensordot(u, ex, axes=0))


b = np.array([1, 2, 3, 4, 5, 6])
u = u.reshape((SIZE_X*SIZE_Y))
b = b.reshape(SIZE_X, SIZE_Y)
print(u)
print("")
print(b)
