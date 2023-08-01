# mlops.c

Some educational implementations. Started out with a matrix multiply aided by openmp parallelization + SIMD.

For a 1 million item matrix multiply (1000x1000) multiplied by (1000x1000) output I get the following average performance on a few runs

```txt
OpenMP + SIMD Matmul
time 0.193359 seconds
megaflops 5171.717172
gigaflops 5.171717

Regular Matmul
time 1.035156 seconds
megaflops 966.037736
gigaflops 0.966038
```

Which is really darn fast for my computer! With just OpenMP parallel I get a 3x speedup, with just OpenMP SIMD I get a 2x speedup. Together,as you can see above I get a 5x speedup over regular.

Check what `Matrix` struct looks like in [`matmul.c`](matmul.c).

However, if I use the `OFast` compiler flag and `march` native for SIMD, the unoptimized (no openmp or anything) beats everything


```txt 
gcc-12 -Ofast -fopenmp -march=native matmul.c  -lm  -o matmul

OpenMP + SIMD Matmul
time 0.189453
megaflops 5278.350515
gigaflops 5.278351

Regular Matmul
time 0.119141
megaflops 8393.442623
gigaflops 8.393443
```

Cool!
