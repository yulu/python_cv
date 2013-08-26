main:main.o
	g++ main.o `pkg-config --libs opencv` -o main -L /usr/local/cuda-5.5/lib64 -lnppc
main.o:main.cpp
	gcc `pkg-config --cflags opencv` -g -c main.cpp