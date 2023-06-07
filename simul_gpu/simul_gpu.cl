#define SIZE_X 
#define SIZE_Y 
#define LATTICE_Q 9
#define cs2 1/3.0f
#define nu 
#define tau (nu/cs2 + 0.5f)


constant int ex[LATTICE_Q] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
constant int ey[LATTICE_Q] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
constant int opposite_bb[LATTICE_Q] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
constant float lattice_w[LATTICE_Q] = {4/9.0f, 1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f, 1/36.0f, 1/36.0f, 1/36.0f, 1/36.0f};

kernel int IDXYQ(int x, int y, int q) {
  return (x * SIZE_Y + y ) * LATTICE_Q + q;
}

kernel int IDXY(int x, int y) {
  return x * SIZE_Y + y;
}

kernel void equilibrium_distribution(__global float *N, __global float *rho, __global float *u, __global float *v) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int q = get_global_id(2);
  int xyq = IDXYQ(x, y, q);
  int xy = IDXY(x, y);
  float vci = u[xy] * ex[q] + v[xy] * ey[q];
  float rhow = rho[xy] * lattice_w[q];
  float vsq = u[xy] * u[xy] + v[xy] * v[xy];
  N[xyq] = rhow * (1 + vci/cs2 + vci*vci/(2*cs2*cs2) - vsq/(2*cs2));
}

kernel void equilibrium_distribution_in(float *N, __global float *rho, __global float *u, __global float *v) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int q = get_global_id(2);
  int xyq = IDXYQ(x, y, q);
  int xy = IDXY(x, y);
  float vci = u[xy] * ex[q] + v[xy] * ey[q];
  float rhow = rho[xy] * lattice_w[q];
  float vsq = u[xy] * u[xy] + v[xy] * v[xy];
  N[xyq] = rhow * (1 + vci/cs2 + vci*vci/(2*cs2*cs2) - vsq/(2*cs2));
}

kernel void equilibrium_perturbation(__global float *N, float rho, float u, float v) {
  int q = get_global_id(2);
  float vci = u * ex[q] + v * ey[q];
  float rhow = rho * lattice_w[q];
  float vsq = u * u + v * v;
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (y>=1 && y<3) {
    int xyq = IDXYQ(x, y, q);
    N[xyq] = rhow * (1 + vci/cs2 + vci*vci/(2*cs2*cs2) - vsq/(2*cs2));
  }
}

kernel int modulo(int dividend, int divisor) {
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

kernel void flow_properties(__global float *N, __global float *rho, __global float *u, __global float *v) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int xy = IDXY(x, y);
  float rhoxy = 0;
  float uxy = 0;
  float vxy = 0;
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

kernel void collide(__global float *N, __global float *rho, __global float *u, __global float *v) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int q = get_global_id(2);
  int xyq = IDXYQ(x, y, q);
  flow_properties(N, rho, u, v);
  float Neq[SIZE_X*SIZE_Y*LATTICE_Q];
  equilibrium_distribution_in(Neq,rho, u, v);
  N[xyq] = N[xyq] - (N[xyq] - Neq[xyq])/tau;
}

kernel void stream(__global float *N, __global int *P) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int q = get_global_id(2);
  int xyq = IDXYQ(x, y, q);
  int xyq_p = P[xyq];
  N[xyq] = N[xyq_p];
}

kernel void impose_vel(__global float *N, __global int *domain_x, __global int *domain_y,float rho_, float ux, float uy) {
  int point_domain = get_global_id(0);
  int q = get_global_id(1);
  int x = domain_x[point_domain];
  int y = domain_y[point_domain];
  float rho = rho_;
  float u = ux;
  float v = uy;
  float vci = u * ex[q] + v * ey[q];
  float rhow = rho * lattice_w[q];
  float vsq = u * u + v * v;
  int xyq = IDXYQ(x, y, q);
  N[xyq] = rhow * (1 + vci/cs2 + vci*vci/(2*cs2*cs2) - vsq/(2*cs2));
}

kernel void bounce_back(__global float *N, __global int *walls_x, __global int *walls_y) {
  int point_walls = get_global_id(0);
  int q = get_global_id(1);
  int x = walls_x[point_walls];
  int y = walls_y[point_walls];
  int qbb = opposite_bb[q];
  int xp = modulo(x - ex[q], SIZE_X);
  int yp = modulo(y - ey[q], SIZE_Y);
  int xyq = IDXYQ(x, y, q);
  int xyqbb = IDXYQ(xp, yp, qbb);
  float mem = N[xyqbb];
  N[xyqbb] = N[xyq];
  N[xyq] = mem;
}