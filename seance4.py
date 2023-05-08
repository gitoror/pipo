import numpy as np
from evtk import hl as vtkhl

LATTICE_D = 2
LATTICE_Q = 9

SIZE_X = 48
SIZE_Y = 64

ex = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int64)
ey = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int64)
lattice_w = np.array([4/9] + [1/9] * 4 + [1/36] * 4)


opposite_bb = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)

cs2 = sum(lattice_w * ex * ex)

nu = 0.001
tau = nu/cs2 + 1/2

nwalls = 40
walls = np.ones([nwalls, LATTICE_D], dtype=np.int64)
walls[:, 0] = np.arange(0, nwalls) + 5
walls[:, 1] = 40

print(walls)
cpt = iter(range(1000000))  # image counter


def IDXY(x, y, q):
    return ((x * SIZE_Y) + y) * LATTICE_Q + q


def save_to_vtk(N, name):
    rho, u, v = flow_properties(N)
    u = np.reshape(u, (SIZE_X, SIZE_Y, 1), order='C')
    v = np.reshape(v, (SIZE_X, SIZE_Y, 1), order='C')
    rho = np.reshape(rho, (SIZE_X, SIZE_Y, 1), order='C')
    vtkhl.imageToVTK(f"images/{name}_{next(cpt)}",
                     pointData={"pressure": rho, "u": u, "v": v})


def init():
    N = np.ones([SIZE_X, SIZE_Y, LATTICE_Q])
    for q in range(LATTICE_Q):
        N[:, :, q] = lattice_w[q]

    return N


def flow_properties(N):
    rho = np.sum(N, axis=2)
    u = np.sum(N * ex, axis=2) / rho
    v = np.sum(N * ey, axis=2) / rho
    return rho, u, v


def stream(N, P):
    return N.reshape(SIZE_X * SIZE_Y * LATTICE_Q)[P].reshape(SIZE_X, SIZE_Y, LATTICE_Q)


def calc_permutations():
    P = np.zeros([SIZE_X * SIZE_Y * LATTICE_Q], dtype=np.int64)
    for x in range(SIZE_X):
        for y in range(SIZE_Y):
            for q in range(LATTICE_Q):
                xp = (x + ex[q]) % SIZE_X
                yp = (y + ey[q]) % SIZE_Y
                P[IDXY(xp, yp, q)] = IDXY(x, y, q)
    return P


def equilibrium_distribution(rho, u, v):
    def p(t2, t1):
        return np.tensordot(t2, t1, axes=0)
    qones = np.ones(LATTICE_Q)
    vci = p(u, ex) + p(v, ey)
    row = p(rho, lattice_w)
    vsq = p(u*u + v*v, qones)
    Neq = row * (1 + (vci)/cs2 + (vci)**2 / (2*cs2**2) - (vsq)/(2*cs2))
    return Neq


def collide(N):
    rho, u, v = flow_properties(N)
    Neq = equilibrium_distribution(rho, u, v)
    return N - (N - Neq)/tau


def bounceback(N, walls):
    for x, y in walls:
        for q in range(LATTICE_Q):
            qbb = opposite_bb[q]
            xp, yp = (x - ex[q]) % SIZE_X, (y - ey[q]) % SIZE_Y
            N[xp, yp, qbb], N[x, y, q] = N[x, y, q], N[xp, yp, qbb]
    return N


def impose_vel(N, domain, uy):
    for x, y in domain:
        N[x, y, :] = equilibrium_distribution(1.0, 0.0, uy)
    return N

###############################################################################


rho = np.ones([SIZE_X, SIZE_Y])  # X ligne, Y colonnes
u = np.zeros([SIZE_X, SIZE_Y])
v = np.zeros([SIZE_X, SIZE_Y])
N = equilibrium_distribution(rho, u, v)

# N[10, 20, :] = N[10, 20, :] * 1.1

P = calc_permutations()
save_to_vtk(N, "rond")

cond_lim = np.array([(j, 0) for j in range(0, SIZE_X)])

print(N[14, 20, :])

# walls sphere de rayon 10 centrée au milieu de N
walls = np.array([(i, j) for i in range(30-5, 30+5)
                 for j in range(30-5, 30+5) if (i-30)**2 + (j-30)**2 < 5**2])

# partie entière

# vitesse u vers la droite
N[:, 1:10, :] = equilibrium_distribution(1.0, 0.1, 0.0)

# for t in range(200):
#     N = collide(N)
#     N = stream(N, P)
#     impose_vel(N, cond_lim, 0.05)
#     # N = bounceback(N, [])
#     N = bounceback(N, walls)
#     save_to_vtk(N, "rond")


# OpenCL (gratuit, ne dépend pas de la carte graphique (=interopérable))
# Cuda (carte graphique NVIDIA)

# Carte graphique
# divisé en unités de calculs (typiquement 256)
# 1 unité = 1 pixel
# Bonnes cartes 32000 unités
# code qui s'exécute : shader / kerne
# écrit en OpenCL (ressemble au C)
# => calcul arithmétique sur des tableaux
# Envoi/Réception des données à la carte graphique : couteux en temps
# => Enjeu : minimiser le nombre d'envois/réceptions
# on traduit le code sequentiel vers le code paralelle fonction par fonction
