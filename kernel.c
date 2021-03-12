#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "LU.h"
#include "FFT.h"
#include "SOR.h"
#include "MonteCarlo.h"
#include "Random.h"
#include "Stopwatch.h"
#include "SparseCompRow.h"
#include "array.h"
#include "kernel.h"

void kernel_measureFFT(unsigned int N, double mintime, Random R,
                       double *res, unsigned long *num_cycles)
{
    /* initialize FFT data as complex (N real/img pairs) */

    int twiceN = 2 * N;
    double *x = RandomVector(twiceN, R);
    unsigned long cycles = 0;
    Stopwatch Q = new_Stopwatch();
    int i = 0;
    double result = 0.0;

    while (1)
    {
        Stopwatch_resume(Q);
        for (i = 0; i < 500; i++)
        {
            FFT_transform(twiceN, x); /* forward transform */
            FFT_inverse(twiceN, x);   /* backward transform */
        }
        Stopwatch_stop(Q);
        if (Stopwatch_read(Q) >= mintime)
            break;

        cycles += 500;
    }
    /* approx Mflops */

    result = FFT_num_flops(N) * cycles / Stopwatch_read(Q) * 1.0e-6;
    Stopwatch_delete(Q);

    free(x);

    *res = result;
    *num_cycles = cycles;
}

void kernel_measureSOR(unsigned int N, double min_time, Random R,
                       double *res, unsigned long *num_cycles)
{
    double **G = RandomMatrix(N, N, R);
    double result = 0.0;

    Stopwatch Q = new_Stopwatch();
    int cycles = 0;
    while (1)
    {
        Stopwatch_resume(Q);
        SOR_execute(N, N, 1.25, G, 300);
        Stopwatch_stop(Q);

        if (Stopwatch_read(Q) >= min_time)
            break;

        cycles += 300;
    }
    /* approx Mflops */

    result = SOR_num_flops(N, N, cycles) / Stopwatch_read(Q) * 1.0e-6;

    Array2D_double_delete(N, N, G);
    *res = result;
    *num_cycles = cycles;

    Stopwatch_delete(Q);
}

void kernel_measureMonteCarlo(double min_time, Random R,
                              double *res, unsigned long *num_cycles)
{
    double result = 0.0;
    Stopwatch Q = new_Stopwatch();

    int cycles = 0;
    while (1)
    {
        Stopwatch_resume(Q);
        MonteCarlo_integrate(1000000);
        Stopwatch_stop(Q);
        if (Stopwatch_read(Q) >= min_time)
            break;

        cycles += 1000000;
    }
    /* approx Mflops */
    result = MonteCarlo_num_flops(cycles) / Stopwatch_read(Q) * 1.0e-6;
    Stopwatch_delete(Q);
    *res = result;
    *num_cycles = cycles;
}

void kernel_measureSparseMatMult(unsigned int N, unsigned int nz,
                                 double min_time, Random R,
                                 double *res, unsigned long *num_cycles)
{
    /* initialize vector multipliers and storage for result */
    /* y = A*y;  */

    double *x = RandomVector(N, R);
    double *y = (double *)malloc(sizeof(double) * N);

    double result = 0.0;

    int nr = nz / N;  /* average number of nonzeros per row  */
    int anz = nr * N; /* _actual_ number of nonzeros         */

    double *val = RandomVector(anz, R);
    int *col = (int *)malloc(sizeof(int) * nz);
    int *row = (int *)malloc(sizeof(int) * (N + 1));
    int r = 0;
    int cycles = 0;
    int i = 0;

    Stopwatch Q = new_Stopwatch();

    for (i = 0; i < N; i++)
        y[i] = 0.0;

    row[0] = 0;
    for (r = 0; r < N; r++)
    {
        /* initialize elements for row r */

        int rowr = row[r];
        int step = r / nr;
        int i = 0;

        row[r + 1] = rowr + nr;
        if (step < 1)
            step = 1; /* take at least unit steps */

        for (i = 0; i < nr; i++)
            col[rowr + i] = i * step;
    }

    while (1)
    {
        Stopwatch_resume(Q);
        SparseCompRow_matmult(N, y, val, row, col, x, 5000);
        Stopwatch_stop(Q);
        if (Stopwatch_read(Q) >= min_time)
            break;

        cycles += 5000;
    }
    /* approx Mflops */
    result = SparseCompRow_num_flops(N, nz, cycles) /
             Stopwatch_read(Q) * 1.0e-6;

    Stopwatch_delete(Q);
    free(row);
    free(col);
    free(val);
    free(y);
    free(x);

    *res = result;
    *num_cycles = cycles;
}

void kernel_measureLU(unsigned int N, double min_time, Random R,
                      double *res, unsigned long *num_cycles)
{

    double **A = NULL;
    double **lu = NULL;
    int *pivot = NULL;

    Stopwatch Q = new_Stopwatch();
    double result = 0.0;
    int i = 0, j = 0;
    int cycles = 0;
    int N2 = N / 2;

    if ((A = RandomMatrix(N, N, R)) == NULL)
        exit(1);
    if ((lu = new_Array2D_double(N, N)) == NULL)
        exit(1);
    if ((pivot = (int *)malloc(N * sizeof(int))) == NULL)
        exit(1);

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            lu[i][j] = 0.0;

    /* make sure A is diagonally dominant, to avoid singularity */
    /* set diagonal to be 4 times the absolute value of its row sum */
    for (i = 0; i < N; i++)
    {
        double row_sum = 0.0;
        /* compute row sum of absoluate values  */
        for (j = 0; j < N; j++)
            row_sum += fabs(A[i][j]);
        A[i][i] = 4 * row_sum;
    }

    while (1)
    {
        Stopwatch_resume(Q);
        for (i = 0; i < 200; i++)
        {
            double lu_center = fabs(lu[N2][N2]);
            Array2D_double_copy(N, N, lu, A);

            /* add modification to A, based on previous LU */
            /* to avoid being optimized out. */
            /*   lu_center = max( A_center, old_lu_center) */

            lu[N2][N2] = (A[N2][N2] > lu_center ? A[N2][N2] : lu_center);

            LU_factor(N, N, lu, pivot);
        }
        Stopwatch_stop(Q);
        if (Stopwatch_read(Q) >= min_time)
            break;

        cycles += 200;
    }

    /* approx Mflops */
    result = LU_num_flops(N) * cycles / Stopwatch_read(Q) * 1.0e-6;

    Stopwatch_delete(Q);
    free(pivot);

    Array2D_double_delete(N, N, lu);
    Array2D_double_delete(N, N, A);

    *res = result;
    *num_cycles = cycles;
}
