target: exe

CC=nvcc
CFLAGS= -std=c++11

main.o: main.cu

exe: main.o
	$(CC) -L./lib $(CFLAGS) $^ -o $@

.PHONY: clean

clean:
	rm -f *.o exe

exec:
	./exe
