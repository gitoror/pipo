import imageio.v2 as imageio
import pyopencl as cl
import numpy as np
import time
from evtk import hl as vtkhl
import pyopencl.cltypes

# Constantes
LATTICE_Q = 9
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
    # print("rho", rho)
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


test = True
if __name__ == '__main__':
    start_time = time.time()
    # OPENCL
    pl = cl.get_platforms()[0]  # platform
    dv = pl.get_devices()[1]  # device
    ctx = cl.Context(devices=[dv])  # contexte
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    # INIT
    # SIZE_X, SIZE_Y, walls = open_image('dessin.png')
    SIZE_X, SIZE_Y = 3, 4
    walls = np.array([(0, 1), (1, 1)])
    # print(walls)

    with open('simul_gpu.cl') as file:
        code = file.read()
    code = code.replace('#define SIZE_X', "#define SIZE_X "+str(SIZE_X))
    code = code.replace('#define SIZE_Y', "#define SIZE_Y "+str(SIZE_Y))
    code = code.replace('#define nu', "#define nu "+str(nu)+'f')

    prog = cl.Program(ctx, code).build()
    # Init cpu variables
    rho = np.ones(SIZE_X * SIZE_Y, dtype=np.float32)
    u = np.zeros(SIZE_X * SIZE_Y, dtype=np.float32)
    v = np.zeros(SIZE_X * SIZE_Y, dtype=np.float32)
    N = np.zeros(SIZE_X * SIZE_Y * LATTICE_Q, dtype=np.float32)
    walls_x = np.array(walls[:, 0], dtype=np.int32)
    walls_y = np.array(walls[:, 1], dtype=np.int32)
    cond_lim = np.array([(j, 0) for j in range(0, SIZE_X)])
    cond_lim_x = np.array(cond_lim[:, 0], dtype=np.int32)
    cond_lim_y = np.array(cond_lim[:, 1], dtype=np.int32)
    # Init gpu variables
    rho_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=rho)
    u_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=u)
    v_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=v)
    N_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=N)
    walls_x_g = cl.Buffer(ctx, mf.READ_WRITE |
                          mf.COPY_HOST_PTR, hostbuf=walls_x)
    walls_y_g = cl.Buffer(ctx, mf.READ_WRITE |
                          mf.COPY_HOST_PTR, hostbuf=walls_y)
    cond_lim_x_g = cl.Buffer(ctx, mf.READ_WRITE |
                             mf.COPY_HOST_PTR, hostbuf=cond_lim_x)
    cond_lim_y_g = cl.Buffer(ctx, mf.READ_WRITE |
                             mf.COPY_HOST_PTR, hostbuf=cond_lim_y)
    # Equilibrium dist
    prog.equilibrium_distribution(
        queue, (SIZE_X, SIZE_Y, LATTICE_Q), None, N_g, rho_g, u_g, v_g)
    # Test equilibrium dist
    if test:
        cl.enqueue_copy(queue, N, N_g)
        N_test = equilibrium_distribution(
            rho.reshape(SIZE_X, SIZE_Y), u.reshape(SIZE_X, SIZE_Y), v.reshape(SIZE_X, SIZE_Y))
        # print(N.reshape(SIZE_X, SIZE_Y, LATTICE_Q))
        # print(N_test)
        print(np.allclose(
            N, N_test.reshape(SIZE_X*SIZE_Y*LATTICE_Q), atol=1e-8), "equilibrium_distribution")

    # Distribution initiale perturbée dans un domaine
    prog.equilibrium_perturbation(queue, (SIZE_X, SIZE_Y, LATTICE_Q), None, N_g, np.float32(1.0),
                                  np.float32(5e-2), np.float32(0.0))
    # Test perturbation
    if test:
        N_test[:, 1:3, :] = equilibrium_distribution(1.0, 5e-2, 0.0)
        cl.enqueue_copy(queue, N, N_g)
        print(np.allclose(
            N, N_test.reshape(SIZE_X*SIZE_Y*LATTICE_Q), atol=1e-8), "equilibrium_perturbation")
        print("N_test")
        print(N_test[:, 1:3, :])
        print("N_gpu")
        print(N.reshape(SIZE_X, SIZE_Y, LATTICE_Q)[:, 1:3, :])
        # for k in range(SIZE_Y):
        #     if not np.allclose(N.reshape(SIZE_X, SIZE_Y, LATTICE_Q)[:, k, :], N_test[:, k, :], atol=1e-8):
        #         print(k)

    # Récupérer N pour save to vtk
    save_to_vtk(N.reshape(SIZE_X, SIZE_Y, LATTICE_Q), "flute_gpu", "img")

    # Calcul des permutations gpu
    P = np.zeros(SIZE_X * SIZE_Y * LATTICE_Q, dtype=np.int32)
    P_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=P)
    prog.calc_permutations(queue, (SIZE_X, SIZE_Y, LATTICE_Q), None, P_g)
    cl.enqueue_copy(queue, P, P_g)
    # Test calcul des permutations
    if test:
        P_test = calc_permutations()
        # print(P.reshape(SIZE_X, SIZE_Y, LATTICE_Q))
        # print(P_test.reshape(SIZE_X, SIZE_Y, LATTICE_Q))
        print(np.allclose(P, P_test, atol=1e-8), "calc_permutations")

    # Boucle
    for t in range(2):
        # Collide
        prog.collide(queue, (SIZE_X, SIZE_Y, LATTICE_Q),
                     None, N_g, rho_g, u_g, v_g)
        # Test collide
        if test:
            cl.enqueue_copy(queue, N, N_g)
            N_test = collide(N_test)
            print(np.allclose(N, N_test.reshape(
                SIZE_X*SIZE_Y*LATTICE_Q), atol=1e-8), "collide", )

        # Condtions aux limites
        # prog.impose_vel(queue, (len(cond_lim), LATTICE_Q),
        #                 None, N_g, cond_lim_x_g, cond_lim_y_g, np.float32(1), np.float32(0), np.float32(0.05))
        cl.enqueue_copy(queue, N, N_g)
        N = N.reshape(SIZE_X, SIZE_Y, LATTICE_Q)
        N = impose_vel(N, cond_lim, 0.05)
        # Test conditions aux limites
        if test:
            # cl.enqueue_copy(queue, N, N_g)
            N_test = impose_vel(N_test, cond_lim, 0.05)
            print(np.allclose(N.reshape(SIZE_X*SIZE_Y*LATTICE_Q), N_test.reshape(
                SIZE_X*SIZE_Y*LATTICE_Q), atol=1e-8), "impose_vel")
            # print(np.allclose(N, N_test.reshape(
            #     SIZE_X*SIZE_Y*LATTICE_Q), atol=1e-8), "impose_vel")
            # print("N_gpu")
            # print(N.reshape(SIZE_X, SIZE_Y, LATTICE_Q))
            # print("N_test")
            # print(N_test)

        # Stream
        N = N.reshape(SIZE_X*SIZE_Y*LATTICE_Q)
        N_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=N)
        prog.stream(queue, (SIZE_X, SIZE_Y, LATTICE_Q), None, N_g, P_g)
        # Test stream
        if test:
            cl.enqueue_copy(queue, N, N_g)
            N_test = stream(N_test, P_test)
            print(np.allclose(N, N_test.reshape(
                SIZE_X*SIZE_Y*LATTICE_Q), atol=1e-8), "stream")

        # Bounce back
        prog.bounce_back(queue, (len(walls_x), LATTICE_Q),
                         None, N_g, walls_x_g, walls_y_g)

        cl.enqueue_copy(queue, N, N_g)

        # Test bounce back
        if test:
            cl.enqueue_copy(queue, N, N_g)
            N_test = bounce_back(N_test)
            print(np.allclose(N, N_test.reshape(
                SIZE_X*SIZE_Y*LATTICE_Q), atol=1e-8), "bounce_back")
            # print("N_gpu")
            # print(N.reshape(SIZE_X, SIZE_Y, LATTICE_Q))
            # print("N_test")
            # print(N_test)
        # Récupérer N pour save to vtk

        if t % 15 == 0:
            cl.enqueue_copy(queue, N, N_g)
            save_to_vtk(N.reshape(SIZE_X, SIZE_Y, LATTICE_Q),
                        "flute_gpu", "img")
            print("t =", t)
    print('temps de calcul', time.time() - start_time)
