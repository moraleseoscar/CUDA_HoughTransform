#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "common/pgm.h"

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;

//*****************************************************************
// The CPU function returns a pointer to the accummulator
void CPU_HoughTran (unsigned char *pic, int w, int h, int **acc)
{
  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;  //(w^2 + h^2)/2, radio max equivalente a centro -> esquina
  *acc = new int[rBins * degreeBins];            //el acumulador, conteo depixeles encontrados, 90*180/degInc = 9000
  memset (*acc, 0, sizeof (int) * rBins * degreeBins); //init en ceros
  int xCent = w / 2;
  int yCent = h / 2;
  float rScale = 2 * rMax / rBins;

  for (int i = 0; i < w; i++) //por cada pixel
    for (int j = 0; j < h; j++) //...
      {
        int idx = j * w + i;
        if (pic[idx] > 0) //si pasa thresh, entonces lo marca
          {
            int xCoord = i - xCent;
            int yCoord = yCent - j;  // y-coord has to be reversed
            float theta = 0;         // actual angle
            for (int tIdx = 0; tIdx < degreeBins; tIdx++) //add 1 to all lines in that pixel
              {
                float r = xCoord * cos (theta) + yCoord * sin (theta);
                int rIdx = (r + rMax) / rScale;
                (*acc)[rIdx * degreeBins + tIdx]++; //+1 para este radio r y este theta
                theta += radInc;
              }
          }
      }
}

// GPU kernel. One thread per image pixel is spawned.
// The accummulator memory needs to be allocated by the host in global memory
__global__ void GPU_HoughTran(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale, float *d_Cos, float *d_Sin) {
    int blockID = blockIdx.x;
    int threadID = threadIdx.x;
    int gloID = blockID * blockDim.x + threadID;

    // Definir locID usando los IDs de los hilos del bloque
    int locID = threadIdx.x;

    // Definir un acumulador local en memoria compartida llamado localAcc
    extern __shared__ int localAcc[];
   
    // Inicializar a 0 todos los elementos de este acumulador local
    for (int i = locID; i < degreeBins * rBins; i += blockDim.x) {
        localAcc[i] = 0;
    }

    // Barrera para asegurar que todos los hilos hayan completado la inicialización del acumulador local
    __syncthreads();

    if (gloID < w * h) { // Reemplazar el return por una condición para continuar solo si el hilo está dentro de la imagen
        int xCent = w / 2;
        int yCent = h / 2;
        int xCoord = gloID % w - xCent;
        int yCoord = yCent - gloID / w;

        if (pic[gloID] > 0) {
            for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
                float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
                int rIdx = (r + rMax) / rScale;

                // Coordinar el acceso a memoria y garantizar que la operación de suma sea completada por cada hilo
                atomicAdd(&localAcc[rIdx * degreeBins + tIdx], 1);
            }
        }
    }

    // Barrera para asegurar que todos los hilos hayan completado el proceso de incremento del acumulador local
    __syncthreads();

    // Agregar un loop para sumar los valores del acumulador local localAcc al acumulador global acc
    for (int i = locID; i < degreeBins * rBins; i += blockDim.x) {
        atomicAdd(&acc[i], localAcc[i]);
    }
}


// Se calculan puntos de la línea inicial y final
double getPoints(double val, char op, double angulo) {
    if (op == '+') {
        return val + 1000 * (angulo);
    } else if (op == '-') {
        return val - 1000 * (angulo);
    } else {
        return 0.0; // Valor predeterminado si el operador no es válido
    }
}

// Función para dibujar las líneas más pesadas en la imagen
void drawAllLines(cv::Mat& image, int *h_hough, int w, int h, float rScale, float rMax, int threshold) {
    std::vector<std::pair<cv::Vec2f, int>> linesWithWeights; // Vector para almacenar las líneas con su peso

    for (int r = 0; r < rBins; r++) {
        for (int theta = 0; theta < degreeBins; theta++) {
            int index = r * degreeBins + theta;
            int weight = h_hough[index];
            if (weight > threshold) { // Comprobar si el peso es mayor que el umbral
                float rValue = (r * rScale) - (rMax);
                float thetaValue = theta * radInc;
                linesWithWeights.push_back(std::make_pair(cv::Vec2f(thetaValue, rValue), weight));
            }
        }
    }

    // Ordenar las líneas por peso en orden descendente
    std::sort(linesWithWeights.begin(), linesWithWeights.end(), [](const std::pair<cv::Vec2f, int>& point0, const std::pair<cv::Vec2f, int>& point1) { return point0.second > point1.second;});

    for (int i = 0; i < linesWithWeights.size(); ++i) {
        cv::Vec2f lineParams = linesWithWeights[i].first;
        float theta = lineParams[0], r = lineParams[1];
        double cosTheta = cos(theta), sinTheta = sin(theta);
        double x0 = (w / 2) + (r * cosTheta), y0 = (h / 2) - (r * sinTheta);
        double xA = getPoints(x0, '+', sinTheta), xB = getPoints(x0, '-', sinTheta), 
        yA = getPoints(y0, '+', cosTheta), yB = getPoints(y0, '-', cosTheta);

        cv::line(image, cv::Point(cvRound(xA), cvRound(yA)), cv::Point(cvRound(xB), cvRound(yB)), cv::Scalar(0, 255, 255), 1.75, cv::LINE_AA);
    }

    cv::imwrite("output.png", image);
}


// Función para comparar los resultados y registrar discrepancias
bool compareResults(int* gpuResult, int* cpuResult, int size) {
    bool match = true;
    for (int i = 0; i < size; i++) {
        if (gpuResult[i] != cpuResult[i]) {
            match = false;
            printf("Discrepancia en el índice %d: GPU = %d, CPU = %d\n", i, gpuResult[i], cpuResult[i]);
        }
    }
    return match;
}


//*****************************************************************
int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Uso: " << argv[0] << " <nombre_de_imagen.pgm>" << std::endl;
        return -1;
    }

    // Load the image using OpenCV
    cv::Mat originalImage = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

    if (originalImage.empty()) {
        std::cerr << "Error al cargar la imagen." << std::endl;
        return -1;
    }

    int w = originalImage.cols;
    int h = originalImage.rows;

    float *d_Cos;
    float *d_Sin;

    // CPU calculation
    int *cpuResult;
    CPU_HoughTran(originalImage.data, originalImage.cols, originalImage.rows, &cpuResult);

    cudaMalloc((void **)&d_Cos, sizeof(float) * degreeBins);
    cudaMalloc((void **)&d_Sin, sizeof(float) * degreeBins);

    // pre-compute values to be stored
    float *pcCos = (float *)malloc(sizeof(float) * degreeBins);
    float *pcSin = (float *)malloc(sizeof(float) * degreeBins);
    float rad = 0;
    for (int i = 0; i < degreeBins; i++) {
        pcCos[i] = cos(rad);
        pcSin[i] = sin(rad);
        rad += radInc;
    }

    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
    float rScale = 2 * rMax / rBins;

    cudaMemcpy(d_Cos, pcCos, sizeof(float) * degreeBins, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sin, pcSin, sizeof(float) * degreeBins, cudaMemcpyHostToDevice);

    // setup and copy data from host to device
    unsigned char *d_in, *h_in;
    int *d_hough, *h_hough;

    h_in = originalImage.data; // h_in contiene los pixeles de la imagen
    h_hough = (int *)malloc(degreeBins * rBins * sizeof(int));

    cudaMalloc((void **)&d_in, sizeof(unsigned char) * w * h);
    cudaMalloc((void **)&d_hough, sizeof(int) * degreeBins * rBins);
    cudaMemcpy(d_in, h_in, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
    cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);
    
    // Marcar el inicio del tiempo de ejecución del kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // execution configuration uses a 1-D grid of 1-D blocks, each made of 256 threads
    //Configurar la dimensión del bloque y memoria compartida
    dim3 blockSize(256);
    dim3 gridSize((w * h + blockSize.x - 1) / blockSize.x);
    int sharedMemorySize = degreeBins * rBins * sizeof(int);

    // Llamada al kernel con memoria compartida
    GPU_HoughTran<<<gridSize, blockSize, sharedMemorySize>>>(d_in, w, h, d_hough, rMax, rScale, d_Cos, d_Sin);

    // get results from device
    cudaMemcpy(h_hough, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

    // compare CPU and GPU results
    bool resultsMatch = compareResults(h_hough, cpuResult, degreeBins * rBins);

    if (resultsMatch) {
        printf("Los resultados coinciden entre GPU y CPU.\n");
    } else {
        printf("Los resultados difieren entre GPU y CPU.\n");
    }

    // Crea una copia de la imagen original utilizando OpenCV
    cv::Mat imageWithLines;
    cv::cvtColor(originalImage, imageWithLines, cv::COLOR_GRAY2BGR); // Convierte a imagen en color

    int threshold = 4175; // Define la cantidad máxima de líneas a dibujar
    drawAllLines(imageWithLines, h_hough, w, h, rScale, rMax, threshold);

    // Marcar el final del tiempo de ejecución del kernel
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Tiempo de ejecución del kernel: %f ms\n", milliseconds);

    printf("Done!\n");

    free(pcCos);
    free(pcSin);
    free(h_hough);
    free(cpuResult);
    cudaFree(d_Cos);
    cudaFree(d_Sin);
    cudaFree(d_in);
    cudaFree(d_hough);

    return 0;
}