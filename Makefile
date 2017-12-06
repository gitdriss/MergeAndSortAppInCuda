target: ./bin/exe ./script/genere

CC=nvcc
CFLAGS= -std=c++11 -lm

./bin/exe: src/main.cu
	rm -f -r bin/
	mkdir bin
	$(CC) -L./lib $(CFLAGS) -o $@ $^

./script/genere: script/genere.c
	$(CC) -o $@ $^

.PHONY: clean

reset:
	rm -f *.o src/tmp.txt bin/exemple2.txt bin/exemple.txt data/price.txt data/kilometer.txt
	rm -f -r bin/

clean:
	rm -f *.o src/tmp.txt
	rm -f -r bin/

exec:
	./script/genere
	python script/doc.py
	cd bin && ./exe
