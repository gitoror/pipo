#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

typedef long double float64;
typedef long int int64;
#define SIZE_X 2
#define SIZE_Y 3
#define LATTICE_Q 9

constant int ex[LATTICE_Q] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
constant int ey[LATTICE_Q] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
constant float lattice_w[LATTICE_Q] = {4/9.0f, 1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f, 1/36.0f, 1/36.0f, 1/36.0f, 1/36.0f};
constant float64 cs2 = 1/3.0f;
constant float64 nu = 0.001f;
#define tau (nu/cs2 + 0.5f)




kernel int IDXYQ(int x, int y, int q) {
  return (x * SIZE_Y + y ) * LATTICE_Q + q;
}

kernel int IDXY(int x, int y) {
  return x * SIZE_Y + y;
}


kernel void equilibrium_distribution(__global float64 *N, __global float64 *rho, __global float64 *u, __global float64 *v) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int q = get_global_id(2);
  int xyq = IDXYQ(x, y, q);
  int xy = IDXY(x, y);
  float64 vci = u[xy] * ex[q] + v[xy] * ey[q];
  float64 rhow = rho[xy] * lattice_w[q];
  float64 vsq = u[xy] * u[xy] + v[xy] * v[xy];
  N[xyq] = rhow * (1 + vci/cs2 + vci*vci/(2*cs2*cs2) - vsq/(2*cs2));
}

kernel void equilibrium_distribution_inline(float64 *N, __global float64 *rho, __global float64 *u, __global float64 *v) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int q = get_global_id(2);
  int xyq = IDXYQ(x, y, q);
  int xy = IDXY(x, y);
  float64 vci = u[xy] * ex[q] + v[xy] * ey[q];
  float64 rhow = rho[xy] * lattice_w[q];
  float64 vsq = u[xy] * u[xy] + v[xy] * v[xy];
  N[xyq] = rhow * (1 + vci/cs2 + vci*vci/(2*cs2*cs2) - vsq/(2*cs2));
}

// def calc_permutations():
//     P = np.zeros(SIZE_X*SIZE_Y*LATTICE_Q, dtype=np.int64)
//     for x in range(SIZE_X):
//         for y in range(SIZE_Y):
//             for q in range(LATTICE_Q):
//                 xp = (x + ex[q]) % SIZE_X
//                 yp = (y + ey[q]) % SIZE_Y
//                 P[IDXY(xp, yp, q)] = IDXY(x, y, q)
//     return P

int modulo(int dividend, int divisor);
inline int modulo(int dividend, int divisor) {
    int result = dividend % divisor;
    if (result < 0) {
        result += divisor;
    }
    return result;
}

kernel void calc_permutations(__global int *P) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int q = get_global_id(2);
  int xp = modulo(x + ex[q], SIZE_X);
  int yp = modulo(y + ey[q], SIZE_Y);
  P[IDXYQ(xp, yp, q)] = IDXYQ(x, y, q);
}

// def flow_properties(N):
//     rho = np.sum(N, axis=2)
//     u = np.sum(N * ex, axis=2) / rho
//     v = np.sum(N * ey, axis=2) / rho
//     return rho, u, v

// void flow_properties(__global float64 *N, __global float64 *rho, __global float64 *u, __global float64 *v);
kernel void flow_properties(__global float64 *N, __global float64 *rho, __global float64 *u, __global float64 *v) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int xy = IDXY(x, y);
  float64 rhoxy = 0;
  float64 uxy = 0;
  float64 vxy = 0;
  for (int q = 0; q < LATTICE_Q; q++) {
    int xyq = IDXYQ(x, y, q);
    rhoxy += N[xyq];
    uxy += N[xyq] * ex[q];
    vxy += N[xyq] * ey[q];
  }
  rho[xy] = rhoxy;
  u[xy] = uxy / rhoxy;
  v[xy] = vxy / rhoxy;
}


// def collide(N):
//     rho, u, v = flow_properties(N)
//     Neq = equilibrium_distribution(rho, u, v)
//     return N - (N-Neq)/tau

kernel void collide(__global float64 *N, __global float64 *rho, __global float64 *u, __global float64 *v) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int q = get_global_id(2);
  int xyq = IDXYQ(x, y, q);
  flow_properties(N, rho, u, v);
  float64 Neq[SIZE_X*SIZE_Y*LATTICE_Q];
  equilibrium_distribution_inline(Neq,rho, u, v);
  N[xyq] = N[xyq] - (N[xyq] - Neq[xyq])/tau;
}

// def stream(N, P):
//     return N.reshape(SIZE_X*SIZE_Y*LATTICE_Q)[P].reshape(SIZE_X, SIZE_Y, LATTICE_Q)

kernel void stream(__global float64 *N, __global int *P) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int q = get_global_id(2);
  int xyq = IDXYQ(x, y, q);
  int xyq_p = P[xyq];
  N[xyq] = N[xyq_p];
}

