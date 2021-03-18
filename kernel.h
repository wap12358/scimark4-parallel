#ifndef KERNEL_H
#define KERNEL_H

void kernel_measureFFT(Random R, double *res, unsigned long *num_cyles);
void kernel_measureSOR(Random R, double *res, unsigned long *num_cyles);
void kernel_measureMonteCarlo(Random R, double *res, unsigned long *num_cyles);
void kernel_measureSparseMatMult(Random R, double *res, unsigned long *num_cyles);
void kernel_measureLU(Random R, double *res, unsigned long *num_cyles);

#endif
