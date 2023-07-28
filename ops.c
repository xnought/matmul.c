#include <assert.h>
#include <stdio.h>

typedef struct {
  double *data;
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
void setMatrix(Matrix *m, int i, int j, double value) {
  m->data[i * m->strides[0] + j * m->strides[1]] = value;
}
void addEqualMatrix(Matrix *m, int i, int j, double value) {
  m->data[i * m->strides[0] + j * m->strides[1]] += value;
}

void printShape(Matrix c) { printf("shape (%d, %d)", c.shape[0], c.shape[1]); }

double dot(Matrix a, Matrix b) {
  assert(a.shape[1] == b.shape[0]);
  double summed = 0.0;
  for (int i = 0; i < a.shape[1]; i++) {
    summed += getMatrix(a, 0, i) * getMatrix(b, i, 0);
  }
  return summed;
}

void matmul(Matrix a, Matrix b, Matrix *out) {
  int m = a.shape[0];
  int inner = b.shape[0];
  int n = b.shape[1];
  for (int k = 0; k < m; k++) {
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < inner; i++) {
        addEqualMatrix(out, k, j, getMatrix(b, i, j) * getMatrix(a, k, i));
      }
    }
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

int main() {
  double data[6] = {0., 1., 2., 3., 4., 5.};
  double output[9] = {0.0};

  Matrix a = matrix(data, (int[]){3, 2});
  Matrix b = matrix(data, (int[]){2, 3});
  Matrix out = matrix(output, (int[]){3, 3});

  printMatrix(a);
  printf("@\n");
  printMatrix(b);
  printf("=\n");
  matmul(a, b, &out);
  printMatrix(out);

  return 0;
}
