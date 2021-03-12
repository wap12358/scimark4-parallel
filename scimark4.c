#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "Random.h"
#include "kernel.h"
#include "constants.h"

int main(int argc, char *argv[])
{
        double min_time = 0.1;

        int FFT_size = FFT_SIZE;
        int SOR_size = SOR_SIZE;
        int Sparse_size_M = SPARSE_SIZE_M;
        int Sparse_size_nz = SPARSE_SIZE_nz;
        int LU_size = LU_SIZE;

        /* run the benchmark */

        double res[6] = {0.0};
        unsigned long num_cycles[6] = {0.0};
        Random R = new_Random_seed(RANDOM_SEED);

        printf("**************   SciMark   **************\n");

        printf("Using %10.2f seconds min time per kenel.", min_time);
        printf("\n\n");

        /* print out results  */

        kernel_measureFFT(FFT_size, min_time, R, &res[1], &num_cycles[1]);
        printf("FFT             Mflops: %8.2f    (N=%d) \n",
               res[1], FFT_size);

        kernel_measureSOR(SOR_size, min_time, R, &res[2], &num_cycles[2]);
        printf("SOR             Mflops: %8.2f    (%d x %d) \n",
               res[2], SOR_size, SOR_size);

        kernel_measureMonteCarlo(min_time, R, &res[3], &num_cycles[3]);
        printf("MonteCarlo:     Mflops: %8.2f  \n", res[3]);

        kernel_measureSparseMatMult(Sparse_size_M,
                                    Sparse_size_nz, min_time, R, &res[4], &num_cycles[4]);
        printf("Sparse matmult  Mflops: %8.2f    (N=%d, nz=%d)  \n",
               res[4], Sparse_size_M, Sparse_size_nz);

        kernel_measureLU(LU_size, min_time, R, &res[5], &num_cycles[5]);
        printf("LU              Mflops: %8.2f    (M=%d, N=%d) \n",
               res[5], LU_size, LU_size);

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
