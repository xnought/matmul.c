all: build run 

build:
	@gcc-12 ops.c -fopenmp

run:
	@./a.out
