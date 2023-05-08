import numpy as np
import time


N = 100
A = np.array(list(range(N)))
# mauvais choix pour faire du calcul car les int sont codés sur plus de bits
L = list(range(N))
# print(A)
# print(type(A))
# print(A.dtype)
# print(type(L[0]))

A + A  # addition terme à terme
A * A  # multiplication terme à terme

T1 = np.zeros(N)
T2 = np.zeros_like(A)  # crée un tableau de 0 de même taille que A
T3 = np.ones(N, dtype=np.float64)

T = list(range(N*N))
T = np.array(T)
T.shape = (N, N)

# print(T)
# print(T[:, 3])

start = time.time()

U = [[N*i+j for j in range(N)] for i in range(N)]
V1 = [[0 for j in range(N)] for i in range(N)]
for i in range(N):
    for j in range(N):
        V1[i][j] = sum(U[i][k]*U[k][j] for k in range(N))

end = time.time()
print("Python", end-start)

start2 = time.time()
V2 = np.zeros_like(T)
for i in range(N):
    for j in range(N):
        V2[i][j] = np.sum(T[i, :]*T[:, j])

end2 = time.time()
print("Numpy naif", end2-start2)

start3 = time.time()
V3 = np.matmul(T, T)
end3 = time.time()
print("Numpy matmul", end3-start3)


# Créer un tableau de taille N, N qui contient des matrice 3x3
SIZE_X = 5
SIZE_Y = 3
LATTICE_Q = 9
N = np.ones((SIZE_X, SIZE_Y, LATTICE_Q), dtype=np.float64)
N[1, 1, :] = np.arange(0, LATTICE_Q)

print(N)
print(' ----- ')

ex = [0, 1, 0, -1, 0, 1, -1, -1, 1]
ey = [0, 0, 1, 0, -1, 1, 1, -1, -1]


def stream(N):
    # return N au temps t+1
    R = np.zeros_like(N)
    for x in range(SIZE_X):
        for y in range(SIZE_Y):
            for q in range(LATTICE_Q):
                xp = (x + ex[q]) % SIZE_X
                yp = (y + ey[q]) % SIZE_Y
                R[xp, yp, q] = N[x, y, q]
    return R


print(stream(N))
