target: exe genere

CC=nvcc
CFLAGS= -std=c++11

bin/exe: src/main.cu
	$(CC) -L./lib $(CFLAGS) -o $@ $^

script/genere: script/genere.c
	$(CC) -o $@ $^

.PHONY: clean

reset:
	rm -f *.o bin/exe src/tmp.txt bin/exemple2.txt bin/exemple.txt data/price.txt data/kilometer.txt

clean:
	rm -f *.o bin/exe src/tmp.txt

exec:
	./script/genere
	python script/doc.py
	./bin/exe
