#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

#include "Random.h"
#include "kernel.h"
#include "constants.h"


int main(int argc, char *argv[])
{

    /* run the benchmark */

    double res[6] = {0.0};
    unsigned long num_cycles[6] = {0.0};
    Random R = new_Random_seed(RANDOM_SEED);
    int cores = sysconf(_SC_NPROCESSORS_ONLN);

    printf("**************   SciMark   **************\n");
    printf("system enable cpu num is %d\n", cores);
    printf("Using %10.2f seconds min time per kenel.", mintime);
    printf("\n\n");

    /* print out results  */

    kernel_measureFFT(R, &res[1], &num_cycles[1], cores);
    printf("FFT             Mflops: %8.2f    (N=%d) \n",
           res[1], FFT_SIZE);

    kernel_measureSOR(R, &res[2], &num_cycles[2], cores);
    printf("SOR             Mflops: %8.2f    (%d x %d) \n",
           res[2], SOR_SIZE, SOR_SIZE);

    kernel_measureMonteCarlo(R, &res[3], &num_cycles[3], cores);
    printf("MonteCarlo:     Mflops: %8.2f  \n", res[3]);

    kernel_measureSparseMatMult(R, &res[4], &num_cycles[4], cores);
    printf("Sparse matmult  Mflops: %8.2f    (N=%d, nz=%d)  \n",
           res[4], SPARSE_SIZE_M, SPARSE_SIZE_nz);

    kernel_measureLU(R, &res[5], &num_cycles[5], cores);
    printf("LU              Mflops: %8.2f    (M=%d, N=%d) \n",
           res[5], LU_SIZE, LU_SIZE);

    res[0] = (res[1] + res[2] + res[3] + res[4] + res[5]) / 5;

    printf("\n");
    printf("Composite Score:       %8.2f\n", res[0]);
    printf("\n");

    printf("FFT reps:              %ld\n", num_cycles[1]);
    printf("SOR reps:              %ld\n", num_cycles[2]);
    printf("Montel Carlo reps:     %ld\n", num_cycles[3]);
    printf("Sparse MatMult repss:  %ld\n", num_cycles[4]);
    printf("LU reps:               %ld\n", num_cycles[5]);
    printf("\n");

    Random_delete(R);

    return 0;
}
