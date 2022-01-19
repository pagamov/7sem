CC = g++
LD = g++
CCFLAGS = -Wall -pedantic -std=c++14 -O3
LDFLAGS =

main: main.o
	$(CC) $(CCFLAGS) main.o -o main -lsfml-graphics -lsfml-window -lsfml-system
main.o: main.cpp ANN_Lib.hpp
	$(CC) $(CCFLAGS) -c ./main.cpp
make clean:
	rm main main.o
