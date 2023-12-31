# CUDA Hough Transform for Line Detection in Images

This repository presents the CUDA implementations of the linear Hough Transform algorithm for detecting lines in black and white images. The project is structured around three distinct versions, encompassing a base implementation, an optimized implementation utilizing Constant Memory, and a specialized implementation utilizing Shared Memory in CUDA.

## Project Overview

The main objectives of this project were:

- Understand the application of GPU Constant Memory.
- Leverage features of Global, Shared, and Constant memories in a common image analysis problem.

### Files Overview

- **hough.cu**: Contains the initial CUDA implementation of the linear Hough Transform.
- **houghConstante.cu**: Implements the Hough Transform utilizing Constant Memory optimization.
- **houghCompartida.cu**: Implements the Hough Transform utilizing Shared Memory optimization.
- **pgm.cpp**: Handles PGM Image I/O used in the project.
- **pgm.h**: Header file for PGMImage class.
- **Makefile**: Builds the different CUDA implementations.
- **runway.pgm**: Image file provided for testing purposes.

## Usage

### Building the Implementations

To compile the CUDA implementations, use the provided Makefile:

```bash
make all
```
### Running the Implementations

To execute any of the CUDA implementations, use the following command format:

- `./hough <image.pgm>`: Base implementation
- `./houghConstante <image.pgm>`: Constant Memory implementation
- `./houghCompartida <image.pgm>`: Shared Memory implementation

## Result Visualization

![Resultant Image](https://i.postimg.cc/Sx8JjVYL/output.png)


## Notes

- Ensure OpenCV is correctly installed for image handling.
- Adjust thresholds or parameters as needed for optimal line detection.