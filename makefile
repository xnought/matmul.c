FILE=matmul.c
CC=gcc-12
all: build-optimized run 

build:
	@$(CC) $(FILE) -fopenmp -lm -o matmul

build-optimized:
	@$(CC) $(FILE) -fopenmp -O3 -lm -o matmul

run:
	@./matmul
