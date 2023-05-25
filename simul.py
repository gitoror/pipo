#!/usr/bin/env python3

import numpy as np
import time
from evtk import hl as vtkhl
import imageio.v2 as imageio

# Domaine
LATTICE_D = 2
LATTICE_Q = 9
# SIZE_X = 60
# SIZE_Y = 40

# Constantes
ex = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
ey = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
lattice_w = np.array([4/9] + [1/9] * 4 + [1/36] * 4)  # poids des directions
cs2 = sum(lattice_w * ex * ex)  # vitesse du son au carré
nu = 0.001  # viscosité cinématique
tau = nu/cs2 + 1/2  # temps de relaxation

# Utilitaires


def IDXY(x, y, q):
    return x*SIZE_Y*LATTICE_Q + y*LATTICE_Q + q


def calc_permutations():
    P = np.zeros(SIZE_X*SIZE_Y*LATTICE_Q, dtype=np.int64)
    for x in range(SIZE_X):
        for y in range(SIZE_Y):
            for q in range(LATTICE_Q):
                xp = (x + ex[q]) % SIZE_X
                yp = (y + ey[q]) % SIZE_Y
                P[IDXY(xp, yp, q)] = IDXY(x, y, q)
    return P


cpt = iter(range(1000000))  # image counter


def save_to_vtk(N, name, path='images'):
    rho, u, v = flow_properties(N)
    u = np.reshape(u, (SIZE_X, SIZE_Y, 1), order='C')
    v = np.reshape(v, (SIZE_X, SIZE_Y, 1), order='C')
    rho = np.reshape(rho, (SIZE_X, SIZE_Y, 1), order='C')
    vtkhl.imageToVTK(f"{path}/{name}_{next(cpt)}",
                     pointData={"pressure": rho, "u": u, "v": v})


# Définition du problème à simuler dans un fichier image
def open_image(filename):
    image = imageio.imread(filename)
    SIZE_X = image.shape[0]
    SIZE_Y = image.shape[1]
    walls = np.array([(i, j) for i in range(SIZE_X) for j in range(
        SIZE_Y) if sum(image[i, j, :]) < 20], dtype=np.int64)
    return SIZE_X, SIZE_Y, walls

# Simulation


def flow_properties(N):
    rho = np.sum(N, axis=2)
    u = np.sum(N * ex, axis=2) / rho
    v = np.sum(N * ey, axis=2) / rho
    return rho, u, v


cs4x2 = 2*cs2**2
cs2x2 = 2*cs2


def equilibrium_distribution1(rho, u, v):
    Neq = np.zeros((SIZE_X, SIZE_Y, LATTICE_Q), dtype=np.float64)
    vsq = u**2 + v**2
    for q in range(LATTICE_Q):
        vci = u * ex[q] + v * ey[q]
        Neq[:, :, q] = rho*lattice_w[q]*(1.+vci/cs2+vci**2/cs4x2-vsq/cs2x2)
    return Neq


def equilibrium_distribution(rho, u, v):
    def p(t2, t1):
        return np.tensordot(t2, t1, axes=0)
    vci = p(u, ex) + p(v, ey)
    rhow = p(rho, lattice_w)
    vsq = p(u**2+v**2, np.ones(LATTICE_Q))
    Neq = rhow*(1 + vci/cs2 + vci**2 /
                (2*cs2**2) - (vsq)/(2*cs2))
    return Neq


# SIZE_X, SIZE_Y = 2, 3

# rho = np.ones((SIZE_X, SIZE_Y), dtype=np.float64)
# u = np.zeros((SIZE_X, SIZE_Y), dtype=np.float64)
# v = np.zeros((SIZE_X, SIZE_Y), dtype=np.float64)
# N = equilibrium_distribution(rho, u, v)
# print(N.shape)
# print(N)
# N2 = equilibrium_distribution(rho, u, v)
# print(N2.shape)

# def stream(N):
#     # return N au temps t+1
# R = np.zeros_like(N)
# for x in range(SIZE_X):
#     for y in range(SIZE_Y):
#         for q in range(LATTICE_Q):
#             xp = (x + ex[q]) % SIZE_X
#             yp = (y + ey[q]) % SIZE_Y
#             R[xp, yp, q] = N[x, y, q]
# return R


def stream(N, P):
    return N.reshape(SIZE_X*SIZE_Y*LATTICE_Q)[P].reshape(SIZE_X, SIZE_Y, LATTICE_Q)

# Collision


def collide(N):
    rho, u, v = flow_properties(N)
    Neq = equilibrium_distribution(rho, u, v)
    return N - (N-Neq)/tau


# Walls
walls = np.array([], dtype=np.int64)
Lwall = 50
for i in range(Lwall):
    walls = np.append(walls, np.array([i, 10], dtype=np.int64))


# Bounce back boundary conditions (parois)
opposite_bb = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)


def bounce_back(N):
    for x, y in walls:
        for q in range(LATTICE_Q):
            qbb = opposite_bb[q]
            xp, yp = (x - ex[q]) % SIZE_X, (y - ey[q]) % SIZE_Y
            N[xp, yp, qbb], N[x, y, q] = N[x, y, q], N[xp, yp, qbb]
    return N


def impose_vel(N, domain, uy):
    # uy << 1 car écoulement faiblement compressible 0.05
    for x, y in domain:
        N[x, y, :] = equilibrium_distribution(1., 0., uy)
    return N

######################################################################
# Conditions limites enviornnement infini vitesse nulles et pressiosns nulles au bord
# pas de réflexion


if __name__ == '__main__':

    # Init
    start_time = time.time()
    SIZE_X, SIZE_Y, walls = open_image('dessin.png')

    rho = np.ones((SIZE_X, SIZE_Y), dtype=np.float64)
    u = np.zeros((SIZE_X, SIZE_Y), dtype=np.float64)
    v = np.zeros((SIZE_X, SIZE_Y), dtype=np.float64)
    P = calc_permutations()
    N = equilibrium_distribution(rho, u, v)
    # Modification de la distribution initiale
    # la dist de chaque pixel (x,y) (de profondeur 9) tq y va 10 à 20 devient la dist d'un point de vitesse 0.05 selon x et de pression 1
    N[:, 10:20, :] = equilibrium_distribution(1.0, 5e-2, 0.0)
    cond_lim = np.array([(j, 0) for j in range(0, SIZE_X)])

    save_to_vtk(N, 'rond')

    # Calcul de la simulation
    # ATTENTION : l'orde dépend de si le bouceback propage après coup ou avant !!!!!!!
    for t in range(300):
        N = collide(N)
        impose_vel(N, cond_lim, 0.05)  # condition aux limites
        N = stream(N, P)  # propagation
        N = bounce_back(N)
        if t % 10 == 0:
            save_to_vtk(N, 'rond')
            print(t)

    # Afficher le temps d'exécution
    print('temps de calcul', time.time() - start_time)
