#define RHO 1.0f;
constant float beta = 1.03f; 
constant int dimension = 3;
constant float omega[3] = {1.0f, 2.0f, 3.0f};
typedef double float64;

kernel void add(__global float *a,__global const float *b, float alpha) {
  int i = get_global_id(0);
  a[i] = a[i] + b[i] + alpha;
  for (int j = 0; j < dimension; j++) {
    a[i] = a[i] + omega[j];
  }
  float sum = 0.0f;
  for (int j = 0; j < dimension; j++) {
    sum += omega[j];
  }
  a[i] -= sum;
}  

kernel void difference(__global float *a,__global const float *b) {
  int i = get_global_id(0);
  a[i] = a[i] - b[i] - beta;
}

kernel void show_float64(float64 a) {
  printf("a = %f\n", a);
}