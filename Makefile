target: exe

CC=nvcc
CFLAGS= -std=c++11

exe: main.cu
	$(CC) -L./lib $(CFLAGS) $^ -o $@

.PHONY: clean

clean:
	rm -f *.o exe

exec:
	./exe
