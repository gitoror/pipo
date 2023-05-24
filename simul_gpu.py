import imageio.v2 as imageio
import pyopencl as cl
import numpy as np
import time
from evtk import hl as vtkhl

# Constantes
SIZE_X, SIZE_Y, LATTICE_Q = 2, 3, 9
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


def equilibrium_distribution(rho, u, v):
    def p(t2, t1):
        return np.tensordot(t2, t1, axes=0)
    vci = p(u, ex) + p(v, ey)
    rhow = p(rho, lattice_w)
    vsq = p(u**2+v**2, np.ones(LATTICE_Q))
    Neq = rhow*(1 + vci/cs2 + vci**2 /
                (2*cs2**2) - (vsq)/(2*cs2))
    return Neq


def stream(N, P):
    return N.reshape(SIZE_X*SIZE_Y*LATTICE_Q)[P].reshape(SIZE_X, SIZE_Y, LATTICE_Q)


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


if __name__ == '__main__':
    pl = cl.get_platforms()[0]  # platform
    dv = pl.get_devices()[1]  # device
    ctx = cl.Context(devices=[dv])  # contexte
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    #
    prog = cl.Program(ctx, open("simul_gpu.cl").read()).build()
    # INIT

    rho = np.ones(SIZE_X * SIZE_Y, dtype=np.float64)
    u = np.zeros(SIZE_X * SIZE_Y, dtype=np.float64)
    v = np.zeros(SIZE_X * SIZE_Y, dtype=np.float64)
    N = np.zeros(SIZE_X * SIZE_Y * LATTICE_Q, dtype=np.float64)
    #
    rho_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=rho)
    u_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=u)
    v_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=v)
    N_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=N)
    #
    prog.equilibrium_distribution(
        queue, (SIZE_X, SIZE_Y, LATTICE_Q), None, N_g, rho_g, u_g, v_g)
    cl.enqueue_copy(queue, N, N_g)
    # Test
    N_test = equilibrium_distribution(
        rho, u, v).reshape(SIZE_X*SIZE_Y*LATTICE_Q)
    print(np.allclose(N, N_test))
    save_to_vtk(N.reshape(SIZE_X, SIZE_Y, LATTICE_Q), "flute", "img")
    # Calcul des permutations gpu
    P = np.zeros(SIZE_X*SIZE_Y*LATTICE_Q, dtype=np.int64)
    P_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=P)
    prog.calc_permutations(queue, (SIZE_X, SIZE_Y, LATTICE_Q), None, P_g)
    cl.enqueue_copy(queue, P, P_g)

    # Test
    P_test = calc_permutations().reshape(SIZE_X*SIZE_Y*LATTICE_Q)
    print(np.allclose(P, P_test))
    print(P)
    print(P_test)

    # Boucle
    # for t in range(300):
    #     prog.collide(queue, (SIZE_X, SIZE_Y, LATTICE_Q), None, N_g)
    #     prog.stream(queue, (SIZE_X, SIZE_Y, LATTICE_Q), None, N_g, P_g)
