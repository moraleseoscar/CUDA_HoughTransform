all: pgm.o houghCompartida	houghConstante	hough 

hough:	hough.cu pgm.o
	nvcc hough.cu pgm.o -o hough -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -I/usr/include/opencv4 -w

houghConstante: houghConstante.cu pgm.o
	nvcc houghConstante.cu pgm.o -o houghConstante -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -I/usr/include/opencv4 -w

houghCompartida: houghCompartida.cu pgm.o
	nvcc houghCompartida.cu pgm.o -o houghCompartida -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -I/usr/include/opencv4 -w

pgm.o:	common/pgm.cpp
	g++ common/pgm.cpp -o pgm.o -lopencv_core -lopencv_imgproc -lopencv_highgui -w