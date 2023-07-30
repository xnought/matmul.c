# mlops.c

Some educational implementations. Started out with a matrix multiply aided by openmp parallelization + SIMD.

```c
void matmul(Matrix a, Matrix b, Matrix out) {
  int m = a.shape[0];
  int inner = b.shape[0];
  int n = b.shape[1];

  #pragma omp parallel for
  for (int k = 0; k < m; k++) {
    for (int j = 0; j < n; j++) {
      double sum = 0.0;
      #pragma omp simd reduction(+ : sum)
      for (int i = 0; i < inner; i++) {
        sum += b.data[b.strides[0] * i + b.strides[1] * j] *
               a.data[a.strides[0] * k + a.strides[1] * i];
      }
      out.data[out.strides[0] * k + out.strides[1] * j] += sum;
    }
  }
}
```

For a 1 million item matrix multiply (1000x1000) multiplied by (1000x1000) output I get the following average performance on a few runs

```txt
OpenMP + SIMD Matmul
time 0.193359 seconds
megaflops 5171.717172
gigaflops 5.171717

Regular Matmul
time 1.035156 seconds
megaflops 966.037736
gigaflops 0.966038```

Which is really darn fast for my computer! With just OpenMP parallel I get a 3x speedup, with just OpenMP SIMD I get a 2x speedup. Together,as you can see above I get a 5x speedup over regular.

Check what `Matrix` struct looks like in [`matmul.c`](matmul.c).

