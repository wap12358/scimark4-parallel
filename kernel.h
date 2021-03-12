#ifndef KERNEL_H
#define KERNEL_H

void kernel_measureFFT(Random R, double *res, unsigned long *num_cyles, int cores);
void kernel_measureSOR(Random R, double *res, unsigned long *num_cyles, int cores);
void kernel_measureMonteCarlo(Random R, double *res, unsigned long *num_cyles, int cores);
void kernel_measureSparseMatMult(Random R, double *res, unsigned long *num_cyles, int cores);
void kernel_measureLU(Random R, double *res, unsigned long *num_cyles, int cores);

#endif
