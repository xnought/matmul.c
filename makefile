all: build run 

build:
	@gcc ops.c

run:
	@./a.out
