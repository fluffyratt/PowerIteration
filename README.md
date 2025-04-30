# PowerIteration

This project implements the Power Iteration method using both OpenMP and MPI for parallel computation.

## Description

Power Iteration is an algorithm used to find the dominant eigenvalue and corresponding eigenvector of a matrix. This project demonstrates how to apply parallel and distributed programming techniques to accelerate the Power Iteration process.

- **MPI (Message Passing Interface)** is used to distribute the matrix and computation tasks across multiple processes, enabling execution on a cluster or multi-node system.
- **OpenMP (Open Multi-Processing)** is employed to further parallelize computations within each process across multiple threads.

The implementation is part of the coursework for the subject *"Distributed and Parallel Programming"*.

## Requirements

- MPI library (e.g., MPICH, OpenMPI)
- OpenMP-compatible C/C++ compiler (e.g., GCC)

## Usage

1. Compile the code using an MPI compiler with OpenMP support.
2. Run the executable using `mpirun` or `mpiexec`.

