#CFLAGS := -Og -ggdb3
CFLAGS := -O3 -flto

snet: main.c rng.c
	gcc -o $@ $^ $(CFLAGS) -lm
