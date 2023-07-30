FILE=matmul.c
CC=gcc-12
all: build-asm run 

build:
	@$(CC) $(FILE) -fopenmp -lm -o matmul

build-optimized:
	@$(CC) $(FILE) -fopenmp -O3 -lm -o matmul

build-asm:
	@$(CC) $(FILE) -fopenmp -lm -o matmul -m64 

run:
	@./matmul
