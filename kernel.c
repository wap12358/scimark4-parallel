#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <pthread.h>
#include "LU.h"
#include "FFT.h"
#include "SOR.h"
#include "MonteCarlo.h"
#include "Random.h"
#include "Stopwatch.h"
#include "SparseCompRow.h"
#include "array.h"
#include "kernel.h"
#include "constants.h"

struct thread_data
{
    Random R;
    int cores;
    double res;
    unsigned long cycles;
};

void *kernel_executeFFT(void *td)
{
    /* initialize FFT data as complex (N real/img pairs) */
    int i = 0;
    double min_time = ((double)(*(struct thread_data *)td).cores * mintime);
    double *x = RandomVector(2 * FFT_SIZE, (*(struct thread_data *)td).R);
    Stopwatch Q = new_Stopwatch();
    (*(struct thread_data *)td).cycles = 0;
    (*(struct thread_data *)td).res = 0.0;

    while (1)
    {
        Stopwatch_resume(Q);
        for (i = 0; i < FFT_CYCLES; i++)
        {
            FFT_transform(2 * FFT_SIZE, x);
            FFT_inverse(2 * FFT_SIZE, x);
        }
        Stopwatch_stop(Q);
        if (Stopwatch_read(Q) >= min_time)
            break;

        (*(struct thread_data *)td).cycles += FFT_CYCLES;
    }

    (*(struct thread_data *)td).res = FFT_num_flops(FFT_SIZE) * (*(struct thread_data *)td).cycles / Stopwatch_read(Q) * 1.0e-6;

    Stopwatch_delete(Q);
    free(x);
    pthread_exit(NULL);
}

void kernel_measureFFT(Random R, double *result,
                       unsigned long *num_cycles, int cores)
{
    /* initialize FFT data as complex (N real/img pairs) */
    int i = 0;
    struct thread_data td[NUM_THREADS];
    unsigned long threads[NUM_THREADS];
    double res = 0.0;
    unsigned long cycles = 0;

    for (i = 0; i < cores; i++)
    {
        td[i].R = R;
        td[i].cores = cores;
        pthread_create(&threads[i], NULL, kernel_executeFFT, (void *)&(td[i]));
    }
    for (i = 0; i < cores; i++)
    {
        pthread_join(threads[i], NULL);
        res += td[i].res;
        cycles += td[i].cycles;
    }
    *result = res;
    *num_cycles = cycles;
}

void *kernel_executeSOR(void *td)
{
    double min_time = ((double)(*(struct thread_data *)td).cores * mintime);
    double **G = RandomMatrix(SOR_SIZE, SOR_SIZE, (*(struct thread_data *)td).R);
    Stopwatch Q = new_Stopwatch();
    (*(struct thread_data *)td).cycles = 0;
    (*(struct thread_data *)td).res = 0.0;
    while (1)
    {
        Stopwatch_resume(Q);
        SOR_execute(SOR_SIZE, SOR_SIZE, 1.25, G, SOR_CYCLES);
        Stopwatch_stop(Q);

        if (Stopwatch_read(Q) >= min_time)
            break;

        (*(struct thread_data *)td).cycles += SOR_CYCLES;
    }

    (*(struct thread_data *)td).res = SOR_num_flops(SOR_SIZE, SOR_SIZE, (*(struct thread_data *)td).cycles) / Stopwatch_read(Q) * 1.0e-6;

    Array2D_double_delete(SOR_SIZE, SOR_SIZE, G);
    Stopwatch_delete(Q);
    pthread_exit(NULL);
}

void kernel_measureSOR(Random R, double *result,
                       unsigned long *num_cycles, int cores)
{
    /* initialize FFT data as complex (N real/img pairs) */
    int i = 0;
    struct thread_data td[NUM_THREADS];
    unsigned long threads[NUM_THREADS];
    double res = 0.0;
    unsigned long cycles = 0;

    for (i = 0; i < cores; i++)
    {
        td[i].R = R;
        td[i].cores = cores;
        pthread_create(&threads[i], NULL, kernel_executeSOR, (void *)&(td[i]));
    }
    for (i = 0; i < cores; i++)
    {
        pthread_join(threads[i], NULL);
        res += td[i].res;
        cycles += td[i].cycles;
    }
    *result = res;
    *num_cycles = cycles;
}

void *kernel_executeMonteCarlo(void *td)
{
    double min_time = ((double)(*(struct thread_data *)td).cores * mintime);
    Stopwatch Q = new_Stopwatch();
    (*(struct thread_data *)td).cycles = 0;
    (*(struct thread_data *)td).res = 0.0;
    while (1)
    {
        Stopwatch_resume(Q);
        MonteCarlo_integrate(MonteCarlo_CYCLES);
        Stopwatch_stop(Q);
        if (Stopwatch_read(Q) >= min_time)
            break;
        (*(struct thread_data *)td).cycles += MonteCarlo_CYCLES;
    }
    (*(struct thread_data *)td).res = MonteCarlo_num_flops((*(struct thread_data *)td).cycles) / Stopwatch_read(Q) * 1.0e-6;
    Stopwatch_delete(Q);
    pthread_exit(NULL);
}

void kernel_measureMonteCarlo(Random R, double *result,
                              unsigned long *num_cycles, int cores)
{
    /* initialize FFT data as complex (N real/img pairs) */
    int i = 0;
    struct thread_data td[NUM_THREADS];
    unsigned long threads[NUM_THREADS];
    double res = 0.0;
    unsigned long cycles = 0;

    for (i = 0; i < cores; i++)
    {
        td[i].R = R;
        td[i].cores = cores;
        pthread_create(&threads[i], NULL, kernel_executeMonteCarlo, (void *)&(td[i]));
    }
    for (i = 0; i < cores; i++)
    {
        pthread_join(threads[i], NULL);
        res += td[i].res;
        cycles += td[i].cycles;
    }
    *result = res;
    *num_cycles = cycles;
}

void *kernel_executeSparseMatMult(void *td)
{
    double min_time = ((double)(*(struct thread_data *)td).cores * mintime);
    double *x = RandomVector(SPARSE_SIZE_M, (*(struct thread_data *)td).R);
    double *y = (double *)malloc(sizeof(double) * SPARSE_SIZE_M);
    int nr = SPARSE_SIZE_nz / SPARSE_SIZE_M;
    int anz = nr * SPARSE_SIZE_M;
    double *val = RandomVector(anz, (*(struct thread_data *)td).R);
    int *col = (int *)malloc(sizeof(int) * SPARSE_SIZE_nz);
    int *row = (int *)malloc(sizeof(int) * (SPARSE_SIZE_M + 1));
    int r = 0;
    int i = 0;
    Stopwatch Q = new_Stopwatch();
    (*(struct thread_data *)td).cycles = 0;
    (*(struct thread_data *)td).res = 0.0;
    for (i = 0; i < SPARSE_SIZE_M; i++)
        y[i] = 0.0;

    row[0] = 0;
    for (r = 0; r < SPARSE_SIZE_M; r++)
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
        SparseCompRow_matmult(SPARSE_SIZE_M, y, val, row, col, x, SparseMatMult_CYCLES);
        Stopwatch_stop(Q);
        if (Stopwatch_read(Q) >= min_time)
            break;

        (*(struct thread_data *)td).cycles += SparseMatMult_CYCLES;
    }

    (*(struct thread_data *)td).res = SparseCompRow_num_flops(SPARSE_SIZE_M, SPARSE_SIZE_nz, (*(struct thread_data *)td).cycles) / Stopwatch_read(Q) * 1.0e-6;
    free(row);
    free(col);
    free(val);
    free(y);
    free(x);
    Stopwatch_delete(Q);
    pthread_exit(NULL);
}

void kernel_measureSparseMatMult(Random R, double *result,
                                 unsigned long *num_cycles, int cores)
{
    /* initialize FFT data as complex (N real/img pairs) */
    int i = 0;
    struct thread_data td[NUM_THREADS];
    unsigned long threads[NUM_THREADS];
    double res = 0.0;
    unsigned long cycles = 0;

    for (i = 0; i < cores; i++)
    {
        td[i].R = R;
        td[i].cores = cores;
        pthread_create(&threads[i], NULL, kernel_executeSparseMatMult, (void *)&(td[i]));
    }
    for (i = 0; i < cores; i++)
    {
        pthread_join(threads[i], NULL);
        res += td[i].res;
        cycles += td[i].cycles;
    }
    *result = res;
    *num_cycles = cycles;
}

void kernel_measureLU(Random R, double *res,
                      unsigned long *num_cycles, int cores)
{

    double **A = NULL;
    double **lu = NULL;
    int *pivot = NULL;

    Stopwatch Q = new_Stopwatch();
    double result = 0.0;
    int i = 0, j = 0;
    int cycles = 0;

    if ((A = RandomMatrix(LU_SIZE, LU_SIZE, R)) == NULL)
        exit(1);
    if ((lu = new_Array2D_double(LU_SIZE, LU_SIZE)) == NULL)
        exit(1);
    if ((pivot = (int *)malloc(LU_SIZE * sizeof(int))) == NULL)
        exit(1);

    for (i = 0; i < LU_SIZE; i++)
        for (j = 0; j < LU_SIZE; j++)
            lu[i][j] = 0.0;

    /* make sure A is diagonally dominant, to avoid singularity */
    /* set diagonal to be 4 times the absolute value of its row sum */
    for (i = 0; i < LU_SIZE; i++)
    {
        double row_sum = 0.0;
        /* compute row sum of absoluate values  */
        for (j = 0; j < LU_SIZE; j++)
            row_sum += fabs(A[i][j]);
        A[i][i] = 4 * row_sum;
    }

    while (1)
    {
        Stopwatch_resume(Q);
        for (i = 0; i < LU_CYCLES; i++)
        {
            double lu_center = fabs(lu[LU_SIZE / 2][LU_SIZE / 2]);
            Array2D_double_copy(LU_SIZE, LU_SIZE, lu, A);

            /* add modification to A, based on previous LU */
            /* to avoid being optimized out. */
            /*   lu_center = max( A_center, old_lu_center) */

            lu[LU_SIZE / 2][LU_SIZE / 2] = (A[LU_SIZE / 2][LU_SIZE / 2] > lu_center ? A[LU_SIZE / 2][LU_SIZE / 2] : lu_center);

            LU_factor(LU_SIZE, LU_SIZE, lu, pivot);
        }
        Stopwatch_stop(Q);
        if (Stopwatch_read(Q) >= mintime)
            break;

        cycles += LU_CYCLES;
    }

    /* approx Mflops */
    result = LU_num_flops(LU_SIZE) * cycles / Stopwatch_read(Q) * 1.0e-6;

    Stopwatch_delete(Q);
    free(pivot);

    Array2D_double_delete(LU_SIZE, LU_SIZE, lu);
    Array2D_double_delete(LU_SIZE, LU_SIZE, A);

    *res = result;
    *num_cycles = cycles;
}
