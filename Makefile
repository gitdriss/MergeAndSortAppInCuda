target: exe genere

CC=nvcc
CFLAGS= -std=c++11

exe: main.cu
	$(CC) -L./lib $(CFLAGS) -o $@ $^

genere: genere.c
	$(CC) -o $@ $^

.PHONY: clean

clean:
	rm -f *.o exe tmp.txt exemple2.txt

exec:
	./genere
	python doc.py
	./exe
