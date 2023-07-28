#include <stdio.h>

typedef struct {
  double *data;
  int shape[2];
  int strides[2];
} Matrix;

void updateStrides(Matrix *m) {
  if (m->shape[0] < m->shape[1]) {
    m->strides[0] = m->shape[1];
    m->strides[1] = 1;
  } else {
    m->strides[0] = m->shape[1];
    m->strides[1] = 1;
  }
}

Matrix matrix(double *data, int shape[2]) {
  Matrix m;
  m.data = data;
  m.shape[0] = shape[0];
  m.shape[1] = shape[1];
  updateStrides(&m);
  return m;
}

double getMatrix(Matrix m, int i, int j) {
  return m.data[i * m.strides[0] + j * m.strides[1]];
}

void printShape(Matrix c) { printf("shape (%d, %d)", c.shape[0], c.shape[1]); }

double dot(Matrix a, Matrix b, int length) {
  double summed = 0.0;
  for (int i = 0; i < length; i++) {
    summed += a.data[i] * b.data[i];
  }
  return summed;
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

int main() {
  double data[6] = {0., 1., 2., 3., 4., 5.};

  Matrix a = matrix(data, (int[]){2, 3});
  printShape(a);
  printf("\n");
  printMatrix(a);

  a = reshape(a, (int[]){6, 1});
  printShape(a);
  printf("\n");
  printMatrix(a);

  return 0;
}
