#include <stdio.h>

typedef struct {
  double *data;
} Matrix;

double dot(Matrix a, Matrix b, int length) {
  double summed = 0.0;
  for (int i = 0; i < length; i++) {
    summed += a.data[i] * b.data[i];
  }
  return summed;
}

int main() {
  Matrix a;
  Matrix b;
  double data[3] = {1., 2., 4.};

  a.data = data;
  b.data = data;

  for (int i = 0; i < 3; i++) {
    printf("%f ", a.data[i]);
  }

  printf("Dot product with a and b = %f", dot(a, b, 3));

  return 0;
}
