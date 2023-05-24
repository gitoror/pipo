typedef long double float64;
typedef long long int int64;
#define SIZE_X 2
#define SIZE_Y 3
#define LATTICE_Q 9

constant int64 ex[LATTICE_Q] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
constant int64 ey[LATTICE_Q] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
constant float lattice_w[LATTICE_Q] = {4/9.0f, 1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f, 1/36.0f, 1/36.0f, 1/36.0f, 1/36.0f};
constant float64 cs2 = 1/3.0f;


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

// def calc_permutations():
//     P = np.zeros(SIZE_X*SIZE_Y*LATTICE_Q, dtype=np.int64)
//     for x in range(SIZE_X):
//         for y in range(SIZE_Y):
//             for q in range(LATTICE_Q):
//                 xp = (x + ex[q]) % SIZE_X
//                 yp = (y + ey[q]) % SIZE_Y
//                 P[IDXY(xp, yp, q)] = IDXY(x, y, q)
//     return P

kernel void calc_permutations(__global int64 *P) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int q = get_global_id(2);
  // int xyq = IDXYQ(x, y, q);
  // int xp = (x + ex[q]) % SIZE_X;
  // int yp = (y + ey[q]) % SIZE_Y;
  // P[IDXYQ(xp, yp, q)] = IDXYQ(x, y, q);
  P[q] = ex[q];  
}