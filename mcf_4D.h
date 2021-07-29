/* Author: Markjoe Olunna UBA
 * Purpose: ImageInLife project - 4D Image Segmentation Methods
 * Language:  C */
#pragma once
#include <stdbool.h>
#include <stddef.h>

bool _allocateMem_mcf();

bool _deallocateMem_mcf();

double _l2norm_mcf(double *dataArray3DPtr1, double *dataArray3DPtr2);

bool _gs_coefficients_mcf();

bool _mcfTimeStep();

bool _set4DBoundaryToExactValues_mcf(double t);

void _exactSolution_mcf(double t);

void _copy4DdataFromSrcToDest_mcf(double *srcDataPtr, double *destDataPtr);

int ijkl(int i, int j, int k, int l);