#include <assert.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define float_t float

typedef struct {
  float_t *data;
  int shape[2];
  int strides[2];
} Matrix;

void updateStrides(Matrix *m) {
  if (m->shape[0] <= m->shape[1]) {
    m->strides[0] = m->shape[1];
    m->strides[1] = 1;
  } else {
    m->strides[0] = m->shape[1];
    m->strides[1] = 1;
  }
}

Matrix matrix(float_t *data, int shape[2]) {
  Matrix m;
  m.data = data;
  m.shape[0] = shape[0];
  m.shape[1] = shape[1];
  updateStrides(&m);
  return m;
}

float_t getMatrix(Matrix m, int i, int j) {
  return m.data[i * m.strides[0] + j * m.strides[1]];
}
void setMatrix(Matrix *m, int i, int j, float_t value) {
  m->data[i * m->strides[0] + j * m->strides[1]] = value;
}
void addEqualMatrix(Matrix *m, int i, int j, float_t value) {
  m->data[i * m->strides[0] + j * m->strides[1]] += value;
}

void printShape(Matrix c) { printf("shape (%d, %d)", c.shape[0], c.shape[1]); }

void unoptimizedmatmul(Matrix a, Matrix b, Matrix out) {
  int m = a.shape[0];
  int inner = b.shape[0];
  int n = b.shape[1];

  for (int k = 0; k < m; k++) {
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < inner; i++) {
        out.data[out.strides[0] * k + out.strides[1] * j] +=
            b.data[b.strides[0] * i + b.strides[1] * j] *
            a.data[a.strides[0] * k + a.strides[1] * i];
      }
    }
  }
}

// optimized out the wazoo
void matmul(Matrix a, Matrix b, Matrix out) {
  int m = a.shape[0];
  int inner = b.shape[0];
  int n = b.shape[1];

#pragma omp parallel for
  for (int k = 0; k < m; k++) {
    for (int j = 0; j < n; j++) {
      float_t sum = 0.0;
#pragma omp simd reduction(+ : sum)
      for (int i = 0; i < inner; i++) {
        sum += b.data[b.strides[0] * i + b.strides[1] * j] *
               a.data[a.strides[0] * k + a.strides[1] * i];
        // addEqualMatrix(out, k, j, getMatrix(b, i, j) * getMatrix(a, k, i));
      }
      out.data[out.strides[0] * k + out.strides[1] * j] += sum;
    }
  }
}

float_t dot(Matrix a, Matrix b) {
  assert(a.shape[1] == b.shape[0]);
  assert(a.shape[0] == 1 && b.shape[1] == 1);
  float_t output[1] = {0.0};
  Matrix out = matrix(output, (int[]){1, 1});
  matmul(a, b, out);
  return out.data[0];
}

void randomData(float_t *out, int length) {
  srand((unsigned int)time(NULL));
  for (int i = 0; i < length; i++) {
    for (int i = 0; i < 20; i++)
      out[i] = ((float_t)rand() / (float_t)(RAND_MAX)) * 1.0;
  }
}

Matrix reshape(Matrix m, int shape[2]) {
  m.shape[0] = shape[0];
  m.shape[1] = shape[1];
  updateStrides(&m);
  return m;
}

void printMatrix(Matrix a) {
  for (int i = 0; i < a.shape[0]; i++) {
    for (int j = 0; j < a.shape[1]; j++) {
      printf("%f ", getMatrix(a, i, j));
    }
    printf("\n");
  }
}

void relu(Matrix m) {
  for (int i = 0; i < m.shape[0] * m.shape[1]; i++) {
    m.data[i] = m.data[i] < 0. ? 0. : m.data[i];
  }
}

#define TOTAL_SIZE 1000000
int main() {
  {
    float_t times = 0.0;
    int runs = 1;
    int dimension = (int)sqrt(TOTAL_SIZE);
    for (int i = 0; i < runs; i++) {
      float_t dataA[TOTAL_SIZE] = {0.0};
      float_t dataB[TOTAL_SIZE] = {0.0};
      float_t output[TOTAL_SIZE] = {0.0};
      randomData(dataA, TOTAL_SIZE);
      randomData(dataB, TOTAL_SIZE);

      int shape[2] = {dimension, dimension};
      Matrix a = matrix(dataA, shape);
      Matrix b = matrix(dataB, shape);
      Matrix out = matrix(output, shape);

      float_t start = omp_get_wtime();
      matmul(a, b, out);
      float_t end = omp_get_wtime();
      times += (end - start);
    }
    printf("OpenMP + SIMD Matmul\n");
    printf("time %f\n", times / runs);
    printf("megaflops %f\n", (pow(dimension, 3) / (times / runs)) / 1e6);
    printf("gigaflops %f", (pow(dimension, 3) / (times / runs)) / 1e9);
  }

  printf("\n\n");

  {
    float_t times = 0.0;
    int runs = 1;
    int dimension = (int)sqrt(TOTAL_SIZE);
    for (int i = 0; i < runs; i++) {
      float_t dataA[TOTAL_SIZE] = {0.0};
      float_t dataB[TOTAL_SIZE] = {0.0};
      float_t output[TOTAL_SIZE] = {0.0};
      randomData(dataA, TOTAL_SIZE);
      randomData(dataB, TOTAL_SIZE);

      int shape[2] = {dimension, dimension};
      Matrix a = matrix(dataA, shape);
      Matrix b = matrix(dataB, shape);
      Matrix out = matrix(output, shape);

      float_t start = omp_get_wtime();
      unoptimizedmatmul(a, b, out);
      float_t end = omp_get_wtime();
      times += (end - start);
    }
    printf("Regular Matmul\n");
    printf("time %f\n", times / runs);
    printf("megaflops %f\n", (pow(dimension, 3) / (times / runs)) / 1e6);
    printf("gigaflops %f", (pow(dimension, 3) / (times / runs)) / 1e9);
  }

  return 0;
}
