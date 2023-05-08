import numpy as np
import time
from evtk import hl as vtkhl
import imageio

# Hyp écoulement faiblement compressible
LATTICE_D = 2
LATTICE_Q = 9
SIZE_X = 60
SIZE_Y = 40

ex = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
ey = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
lattice_w = np.array([4/9] + [1/9] * 4 + [1/36] * 4)
cs2 = sum(lattice_w * ex * ex)
nu = 0.1
tau = nu/cs2 + 1/2


def init():
    N = np.ones((SIZE_X, SIZE_Y, LATTICE_Q), dtype=np.float64)
    for q in range(LATTICE_Q):
        N[:, :, 0] = lattice_w[0]
    return N


def IDXY(x, y, q):
    return x*SIZE_Y*LATTICE_Q + y*LATTICE_Q + q


def flow_properties(N):
    rho = np.sum(N, axis=2)
    u = np.sum(N * ex, axis=2) / rho
    v = np.sum(N * ey, axis=2) / rho
    return rho, u, v


cpt = iter(range(1000000))  # image counter


def save_to_vtk(N, name):
    rho, u, v = flow_properties(N)
    u = np.reshape(u, (SIZE_X, SIZE_Y, 1), order='C')
    v = np.reshape(v, (SIZE_X, SIZE_Y, 1), order='C')
    rho = np.reshape(rho, (SIZE_X, SIZE_Y, 1), order='C')
    vtkhl.imageToVTK(f"images/{name}_{next(cpt)}",
                     pointData={"pressure": rho, "u": u, "v": v})


def stream(N):
    # return N au temps t+1
    R = np.zeros_like(N)
    for x in range(SIZE_X):
        for y in range(SIZE_Y):
            for q in range(LATTICE_Q):
                xp = (x + ex[q]) % SIZE_X
                yp = (y + ey[q]) % SIZE_Y
                # pb avec ca c'est qu on utilise python et pas numpy
                R[xp, yp, q] = N[x, y, q]
    return R


def stream2(N, P):
    return N.reshape(SIZE_X*SIZE_Y*LATTICE_Q)[P].reshape(SIZE_X, SIZE_Y, LATTICE_Q)


def calc_permutations(N):
    P = np.zeros(SIZE_X*SIZE_Y*LATTICE_Q, dtype=np.int64)
    for x in range(SIZE_X):
        for y in range(SIZE_Y):
            for q in range(LATTICE_Q):
                xp = (x + ex[q]) % SIZE_X
                yp = (y + ey[q]) % SIZE_Y
                P[IDXY(xp, yp, q)] = IDXY(x, y, q)
    return P


# fait office d'initialisation, cette fois avec des vitesses non nulles
def equilibrium_distribution(rho, u, v):
    def p(t2, t1):
        return np.tensordot(t2, t1, axes=0)
    vci = p(u, ex) + p(v, ey)
    rhow = p(rho, lattice_w)
    vsq = p(u**2+v**2, np.ones(LATTICE_Q))
    Neq = rhow*(1 + (vci)/cs2 + (vci)**2 /
                (2*cs2**2) - (vsq)/(2*cs2))
    return Neq


def collide(N):
    rho, u, v = flow_properties(N)
    Neq = equilibrium_distribution(rho, u, v)
    return N - (N-Neq)/tau


walls = np.array([], dtype=np.int64)
Lwall = 50
for i in range(Lwall):
    walls = np.append(walls, np.array([i, 10], dtype=np.int64))

opposite_bb = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)


def bounce_back(N):
    for x, y in walls:
        for q in range(LATTICE_Q):
            qbb = opposite_bb[q]
            xp, yp = (x - ex[q]) % SIZE_X, (y - ey[q]) % SIZE_Y
            N[xp, yp, qbb], N[x, y, q] = N[x, y, q], N[xp, yp, qbb]
    return N

# ux << 1 car écoulement faiblement compressible 0.05


def impose_vel(N, domain, ux):
    for x, y in domain:
        N[x, y, :] = equilibrium_distribution(1, ux, 0)
    return N


def open_image(filename):
    image = imageio.imread(filename)
    SIZE_X = image.shape[0]
    SIZE_Y = image.shape[1]
    murs = [(i, j) for i in range(SIZE_X)
            for j in range(SIZE_Y) if image[i, j, :] < 20]
    murs = np.array(murs)
    return SIZE_X, SIZE_Y, murs


rho = np.ones((SIZE_X, SIZE_Y))
u = np.zeros((SIZE_X, SIZE_Y))
v = np.zeros((SIZE_X, SIZE_Y))
N = equilibrium_distribution(rho, u, v)
print(N.shape)
# N = init()
N[10, 20, :] = N[10, 20, :] * 1.1
save_to_vtk(N, 'rond')

P = calc_permutations(N)
for t in range(200):
    N = stream2(N, P)  # propagation
    N = collide(N)
    N = bounce_back(N)
    save_to_vtk(N, 'rond')

# Pour la prochaine fois : changer le stream
# --> enregirstre les indices à leur prochanes positions
