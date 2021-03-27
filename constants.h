#ifndef CONSTANTS_H_
#define CONSTANTS_H_

#define RANDOM_SEED 101010
#define mintime (0.2)
#define NUM_THREADS 64
#define FFT_CYCLES 500
#define SOR_CYCLES 300
#define MonteCarlo_CYCLES 1000000
#define SparseMatMult_CYCLES 5000
#define LU_CYCLES 200

#define FFT_SIZE 1024
#define SOR_SIZE 100
#define SPARSE_SIZE_M 1000
#define SPARSE_SIZE_nz 5000
#define LU_SIZE 100

#define LG_FFT_SIZE 1048576
#define LG_SOR_SIZE 1000
#define LG_SPARSE_SIZE_M 100000
#define LG_SPARSE_SIZE_nz 1000000
#define LG_LU_SIZE 1000

#define TINY_FFT_SIZE 16
#define TINY_SOR_SIZE 10
#define TINY_SPARSE_SIZE_M 10
#define TINY_SPARSE_SIZE_N 10
#define TINY_SPARSE_SIZE_nz 50
#define TINY_LU_SIZE 10

#endif
