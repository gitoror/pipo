#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boltzmann sur réseau
--------------------

@author: hippolyte
"""


import numpy as np
import imageio.v2 as imageio
from time import time


###############################################################################
################################ Initialisation ###############################
###############################################################################

#%% Paramètres de simulation

# Nom de la simulation
NAME = 'dessin'

# Nombre de pas de simulation
TIME_STEPS = 10000

# Intervalle d'écriture
DT_WRITE = 10


#%% Paramètres physiques

# Viscosité du gaz
NU = 1e-3


#%% Domaine

def init():
    N = np.zeros((SIZE_X, SIZE_Y, LATTICE_Q), dtype=DTYPE_FLOAT)
    N[:, :, :] = eq_distrib(1.0, 0.0, 0.0)
    N[:, 10:20, :] = eq_distrib(1.0, 1e-1, 0.0)
    return N


def create_walls():
    return open_image("dessin.png")


def create_CLeq():
    return np.array([(x, 0) for x in range(SIZE_X)])
u_CL = 0.
v_CL = 5e-2


###############################################################################
################################## Programme ##################################
###############################################################################

#%% Utils

# Conversion indices matrice->ligne
def IDXY(i, j, q):
    return ((i * SIZE_Y) + j) * LATTICE_Q + q


# Permutations
def permutations():
    P = np.zeros(SIZE_RESHAPE, dtype=DTYPE_INT)
    for i in range(SIZE_X):
        for j in range(SIZE_Y):
            for q in range(LATTICE_Q):
                ijq = IDXY((i+ex[q])%SIZE_X, (j+ey[q])%SIZE_Y, q)
                P[ijq] = IDXY(i, j, q)
    return P


# Définition du problème à simuler dans un fichier image
def open_image(filename):
    image = imageio.imread(filename)
    SIZE_X = image.shape[0]
    SIZE_Y = image.shape[1]
    walls = np.array([(i,j) for i in range(SIZE_X) for j in range(SIZE_Y) if sum(image[i,j,:]) < 20], dtype=DTYPE_INT)
    return SIZE_X, SIZE_Y, walls


#%% Constantes

# Numpy types
DTYPE_INT = np.int64
DTYPE_FLOAT = np.float64

# Réseau
LATTICE_D = 2
LATTICE_Q = 9

# Règles de déplacement
ex = np.array([0,  1,  0, -1,  0,  1, -1, -1,  1], dtype=DTYPE_INT)
ey = np.array([0,  0,  1,  0, -1,  1,  1, -1, -1], dtype=DTYPE_INT)
lattice_w = np.array([4./9.,
                      1./9., 1./9., 1./9., 1./9.,
                      1./36., 1./36., 1./36., 1./36.], dtype=DTYPE_FLOAT)

# Règles de déplacement aux parois
opposite_bb = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=DTYPE_INT)

# Domaine
SIZE_X, SIZE_Y, walls = create_walls()
walls_X, walls_Y = walls[:, 0], walls[:, 1]
walls_Xp = np.array([(walls_X + ex[q]) % SIZE_X for q in range(LATTICE_Q)], dtype=DTYPE_INT)
walls_Yp = np.array([(walls_Y + ey[q]) % SIZE_Y for q in range(LATTICE_Q)], dtype=DTYPE_INT)
CLeq = create_CLeq()
CLeq_X, CLeq_Y = CLeq[:, 0], CLeq[:, 1]
SIZE_RESHAPE = SIZE_X * SIZE_Y * LATTICE_Q

# Vitesse du son
cs2 = 1./3.
cs2x2 = 2. * cs2
cs4x2 = 2. * cs2**2

# Temps caractéristique
TAU = NU/cs2 + 0.5

# Permutations
P = permutations()


#%% Propriétés du fluide

def flow_properties(N, rho, u, v):
    np.sum(N, axis=2, out=rho)
    np.sum(ex * N, axis=2, out=u)
    np.sum(ey * N, axis=2, out=v)
    u /= rho
    v /= rho


#%% Distribution à l'équilibre

def equilibrium_distribution(rho, u, v, Neq):
    vit2 = u**2 + v**2
    for q in range(LATTICE_Q):
        vci = u * ex[q] + v * ey[q]
        Neq[:, :, q] = rho * lattice_w[q] * (1. + vci/cs2 + vci**2/cs4x2 - vit2/cs2x2)


def eq_distrib(rho, u, v):
    vit2 = u**2 + v**2
    vci = u * ex + v * ey
    return rho * lattice_w * (1. + vci/cs2 + vci**2/cs4x2 - vit2/cs2x2)


#%% Collision

def collide(N, rho, u, v, Neq):
    equilibrium_distribution(rho, u, v, Neq)
    N *= (1. - 1./TAU)
    N += Neq / TAU


#%% Propagation

def stream(N):
    N[:] = N.reshape(SIZE_RESHAPE)[P].reshape(SIZE_X, SIZE_Y, LATTICE_Q)


#%% Conditions de paroi

def bounce_back(N):
    for q in range(LATTICE_Q):
        Xp = walls_Xp[q]
        Yp = walls_Yp[q]
        qbb = opposite_bb[q]
        N[Xp, Yp, qbb], N[walls_X, walls_Y, q] = N[walls_X, walls_Y, q], N[Xp, Yp, qbb]


#%% Conditions aux limites

def conditions_limites(N):
    N[CLeq_X, CLeq_Y, :] = eq_distrib(1.0, u_CL, v_CL)


#%% Format EVTK

from evtk import hl as vtkhl

cpt = iter(range(1000000)) #image counter

def save_to_vtk(rho, u, v, name):
    rho[walls_X, walls_Y] = np.nan
    u[walls_X, walls_Y] = np.nan
    v[walls_X, walls_Y] = np.nan
    u = np.reshape(u, (SIZE_X, SIZE_Y, 1), order='C')
    v = np.reshape(v, (SIZE_X, SIZE_Y, 1), order='C')
    rho = np.reshape(rho, (SIZE_X, SIZE_Y, 1), order='C')
    vtkhl.imageToVTK(f"images/{name}_{next(cpt)}",
                     pointData={"pressure": rho, "u": u, "v": v})


#%% Exécution

# Informations sur la simulation
print("\nBoltzmann sur réseau")
print("--------------------")
print(f"Nom de la simulation      {NAME}")
print(f"Nombre de pas de temps    {TIME_STEPS}")
print(f"Intervalle d'écriture     {DT_WRITE}")
print(f"SIZE_X                    {SIZE_X}")
print(f"SIZE_Y                    {SIZE_Y}")
print(f"Viscosité cinématique     {NU}")
print("\nExécution")
print("---------")

# Début du timer
deb = time()

# Initialisation
N = init()
Neq = np.zeros((SIZE_X, SIZE_Y, LATTICE_Q), dtype=DTYPE_FLOAT)
rho = np.zeros((SIZE_X, SIZE_Y), dtype=DTYPE_FLOAT)
u = np.zeros((SIZE_X, SIZE_Y), dtype=DTYPE_FLOAT)
v = np.zeros((SIZE_X, SIZE_Y), dtype=DTYPE_FLOAT)

# Calcul
for t in range(1, TIME_STEPS+1):
    # Affichage du pas de temps tous les multiples de 10
    if np.log10(t) % 1. == 0 or t == TIME_STEPS:
        print(f"Pas de temps    {t}")

    flow_properties(N, rho, u, v)
    collide(N, rho, u, v, Neq)  # Collision
    stream(N)  # Propagation
    conditions_limites(N)
    bounce_back(N)  # Parois
    if t % DT_WRITE == 0:
        save_to_vtk(rho, u, v, NAME)
flow_properties(N, rho, u, v)
save_to_vtk(rho, u, v, NAME)

# Fin du timer
fin = time()
temps = fin - deb

# Informations d'exécution
print("\nInformations")
print("------------")
print(f"Durée de simulation    {int(round(temps, 0))} secondes")
