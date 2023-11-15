all: pgm.o	hough

hough: houghBase.cu pgm.o
    nvcc -arch=sm_35 houghBase.cu pgm.o -o houghBase


pgm.o:	common/pgm.cpp
	g++ -c common/pgm.cpp -o ./pgm.o
