/* Author: Markjoe Olunna UBA
 * Purpose: ImageInLife project - 4D Image Segmentation Methods
 * Language:  C */
#pragma once
#include <stdbool.h>
#include <stddef.h>

bool _allocateMem_mcf_mpi();

bool _deallocateMem_mcf_mpi();

double _l2norm_mcf_mpi(double *dataArray4DPtr1, double *dataArray4DPtr2);

bool _gs_coefficients_mcf_mpi();

bool _mcfTimeStep_mpi();

bool _set4DBoundaryToExactValues_mcf_mpi(double t);

void _exactSolution_mcf_mpi(double t);

void _copy4DdataFromSrcToDest_mcf_mpi(double *srcDataPtr, double *destDataPtr);

int ijkl(int i, int j, int k, int l);

