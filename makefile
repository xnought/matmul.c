FILE=matmul.c
CC=gcc-12
all: build-optimized run 

build:
	@$(CC) $(FILE) -fopenmp -lm -o matmul

build-optimized:
	$(CC) -Ofast -fopenmp -march=native $(FILE)  -lm  -o matmul

build-asm:
	@$(CC) $(FILE) -fopenmp -lm -o matmul -m64 

run:
	@./matmul
