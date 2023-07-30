all: build run 

build:
	@gcc-12 ops.c -fopenmp -O3 -lm

run:
	@./a.out
