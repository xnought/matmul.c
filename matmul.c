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
  int offset;
} Matrix;

Matrix matrix(float_t *data, int shape[2]) {
  Matrix m;
  m.data = data;

  m.shape[0] = shape[0];
  m.shape[1] = shape[1];

  m.strides[0] = m.shape[1];
  m.strides[1] = 1;

  m.offset = 0;

  return m;
}

int dataIndex(Matrix *m, int i, int j) {
  return m->offset + i * m->strides[0] + j * m->strides[1];
}
float_t getMatrix(Matrix m, int i, int j) {
  return m.data[dataIndex(&m, i, j)];
}

void printShape(Matrix c) { printf("shape (%d, %d)", c.shape[0], c.shape[1]); }

void unoptimizedmatmul(Matrix a, Matrix b, Matrix out) {
  int m = a.shape[0];
  int inner = b.shape[0];
  int n = b.shape[1];

  for (int k = 0; k < m; k++) {
    for (int j = 0; j < n; j++) {
      float_t sum = 0.0;
      for (int i = 0; i < inner; i++) {
        sum += b.data[b.offset + b.strides[0] * i + b.strides[1] * j] *
               a.data[a.offset + a.strides[0] * k + a.strides[1] * i];
      }
      out.data[out.offset + out.strides[0] * k + out.strides[1] * j] = sum;
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
        sum += b.data[b.offset + b.strides[0] * i + b.strides[1] * j] *
               a.data[a.offset + a.strides[0] * k + a.strides[1] * i];
      }
      out.data[out.offset + out.strides[0] * k + out.strides[1] * j] += sum;
    }
  }
}

// using asm under the hood
void asmmatmul(Matrix a, Matrix b, Matrix out) {
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
    out[i] = ((float_t)rand() / (float_t)(RAND_MAX)) * 1.0;
  }
}

Matrix reshape(Matrix m, int shape[2]) {
  m.shape[0] = shape[0];
  m.shape[1] = shape[1];

  m.strides[0] = m.shape[1];
  m.strides[1] = 1;

  return m;
}

Matrix columnSlice(Matrix m, int column) {
  m.strides[0] = m.shape[1];
  m.strides[1] = 1;

  m.shape[0] = m.shape[0];
  m.shape[1] = 1;

  m.offset = column;

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

void columnMulAdd(Matrix a, float_t scaled, Matrix out) {
#pragma omp simd
  for (int i = 0; i < out.shape[0]; i++)
    out.data[out.offset + out.strides[0] * i] +=
        scaled * a.data[a.offset + a.strides[0] * i];
}

void reshapematmul(Matrix a, Matrix b, Matrix out) {
  // Separate into matrix vector mult
#pragma omp parallel for
  for (int i = 0; i < b.shape[1]; i++) {
    Matrix bCol = columnSlice(b, i);
    Matrix outCol = columnSlice(out, i);
    // Separate into scalar times vector summed
    for (int j = 0; j < b.shape[0]; j++) {
      float_t bVal = getMatrix(bCol, j, 0);
      Matrix aCol = columnSlice(a, j);
      columnMulAdd(aCol, bVal, outCol);
    }
  }
}

#define TOTAL_SIZE 1000000
void matmulSpeedTest() {
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
      reshapematmul(a, b, out);
      float_t end = omp_get_wtime();
      times += (end - start);
    }
    printf("\n\nReshape Matmul\n");
    printf("time %f\n", times / runs);
    printf("megaflops %f\n", (pow(dimension, 3) / (times / runs)) / 1e6);
    printf("gigaflops %f", (pow(dimension, 3) / (times / runs)) / 1e9);
  }
}

float performSimdSum(float *inputArray, int arraySize) {
  int simdSize = 4; // 4 floats at a time for SSE.
  float sum = 0.0f;

  // Outer for loop to process elements in SIMD chunks.
  for (int i = 0; i < arraySize; i += simdSize) {
    // Use inline assembly to load and sum the elements in SIMD registers
    // (SSE).
    __asm__("movups (%0), %%xmm0\n"   // Load 4 floats from inputArray into xmm0
                                      // register.
            "haddps %%xmm0, %%xmm0\n" // Sum the first 2 floats, and the last
                                      // 2 floats in xmm0.
            "haddps %%xmm0, %%xmm0\n" // Sum all 4 floats in xmm0.
            "movss %%xmm0, %1\n"      // Store the sum in the sum variable.
            :
            : "r"(&inputArray[i]), "m"(sum)
            : "%xmm0");
  }

  // For any remaining elements (if the array size is not a multiple of
  // simdSize), sum them sequentially using regular scalar operations.
  for (int i = arraySize - (arraySize % simdSize); i < arraySize; i++) {
    sum += inputArray[i];
  }

  return sum;
}

float __asm_dot(float *a, float *b, int arraySize) {
  int simdSize = 4; // 4 floats at a time for SSE.
  float sum[] = {0.0, 0.0, 0.0, 0.0};

  // Outer for loop to process elements in SIMD chunks.
  for (int i = 0; i < arraySize; i += simdSize) {
    // Use inline assembly to load and sum the elements in SIMD registers
    // (SSE).
    __asm__("movups (%1), %%xmm0\n" // Load 4 floats from inputArray into xmm0
            "movups (%2), %%xmm1\n" // Load 4 floats from inputArray into xmm0
            "mulps %%xmm1, %%xmm0\n"
            "addps %%xmm0, %0\n" // Store the sum in the sum variable.
            : "=x"(sum)
            : "r"(&a[i]), "r"(&b[i])
            : "%xmm0", "%xmm1");
  }
  float result = sum[0] + sum[1] + sum[2] + sum[3];
  return result;
}

void asmTest() {
  float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
  float b[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
  int arraySize = sizeof(a) / sizeof(float);

  float sum = __asm_dot(a, b, arraySize);
  printf("Sum: %f\n", sum);
}

int main() {
  matmulSpeedTest();
  return 0;
}
