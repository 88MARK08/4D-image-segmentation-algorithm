/* Author: Markjoe Olunna UBA
 * Purpose: ImageInLife project - 4D Image Segmentation Methods
 * Language:  C */
#include <stdio.h> 
#include <stdlib.h>
#include <time.h>
#include <math.h> 
#include <stdbool.h>
#include "mcf_4D.h"

// Image Dimensions
int kMax, iMax, jMax, lMax; // Absolute dimensions plus 2 in each direction e.g., n1 == iMax - 2

double *u; //  Pointer to the numerical solution
double *prev_u; // Pointer to the Previous solution
double *exact_u; // Pointer to the exact solution

double *e; // East coefficient pointer
double *w; // West coefficient pointer
double *n; // North coefficient pointer
double *s; // South coefficient pointer
double *t; // Top coefficient pointer
double *b; // Bottom coefficient pointer
double *fw; // Forward coefficient pointer
double *bw; // Backward coefficient pointer

int maxNoGSIteration;// Maximum number of Gauss-Seidel iterations
double eps2; // epsilon is the regularization factor (Evans-Spruck)
int numberOfTimeStep;// Number of current time step
int maxNoOfTimeSteps;// Maximum number of time step
double mcfTol; // Tolerance for stopping of the evolution process
double tau, h, omega_c, gsTol; /* h is the Grid size, tau is time step for the evolution process,
omega_c is the relaxation parameter in SOR implementation using Gauss-Seidel, gsTol is the acceptable
tolerance for Gauss-Seidel iterations*/

int ijkl(int i, int j, int k, int l)
{
	return l * iMax * jMax * kMax + k * iMax * jMax + j * iMax + i;
	// or equivalently return ((l * kMax + k) * jMax + j) * iMax + i
}

bool _allocateMem_mcf()
{
	int dim4D = iMax * jMax *kMax * lMax;
	prev_u = (double *)malloc(sizeof(double) * dim4D);
	u = (double *)malloc(sizeof(double) * dim4D);
	exact_u = (double *)malloc(sizeof(double) * dim4D);

	e = (double *)malloc(sizeof(double) * dim4D);
	w = (double *)malloc(sizeof(double) * dim4D);
	n = (double *)malloc(sizeof(double) * dim4D);
	s = (double *)malloc(sizeof(double) * dim4D);
	t = (double *)malloc(sizeof(double) * dim4D);
	b = (double *)malloc(sizeof(double) * dim4D);
	fw = (double *)malloc(sizeof(double) * dim4D);
	bw = (double *)malloc(sizeof(double) * dim4D);

	//checks if the memory was allocated
	if (exact_u == NULL || prev_u == NULL || u == NULL || e == NULL || w == NULL || n == NULL ||
		s == NULL || t == NULL || b == NULL || fw == NULL || bw == NULL)
		return false;

	return true;
}
bool _deallocateMem_mcf()
{
	free(prev_u);
	free(exact_u);
	free(u);

	free(e);
	free(w);
	free(n);
	free(s);
	free(t);
	free(b);
	free(fw);
	free(bw);
	return true;
}
double _l2norm_mcf(double *dataArray3DPtr1, double *dataArray3DPtr2)
{
	double l2norm, sumPower = 0;
	double hhhh = h * h * h * h;

	//checks if the memory was allocated
	if (dataArray3DPtr1 == NULL || dataArray3DPtr2 == NULL)
		return false;

	/*Computation of norm*/
	int i, j, k, l;
	for (l = 0; l < lMax; l++)
		for (k = 0; k < kMax; k++)
			for (j = 0; j < jMax; j++)
				for (i = 0; i < iMax; i++)
					sumPower += (pow(dataArray3DPtr1[ijkl(i, j, k, l)] - dataArray3DPtr2[ijkl(i, j, k, l)], 2) * hhhh);

	l2norm = sqrt(sumPower);
	return l2norm;
}

bool _mcfTimeStep()
{
	double  time = tau * numberOfTimeStep;
	int z = 0; // Steps counter
	int i, j, k, l;
	double hh = h * h;
	double coef_tauh = tau / hh;
	double mean_square_residue;
	double hhhh = h * h * h * h;

	//Copy current solution to previous solution
	_copy4DdataFromSrcToDest_mcf(u, prev_u);

	//Calculation of coefficient
	_gs_coefficients_mcf();

	_set4DBoundaryToExactValues_mcf(time);//set boundary values to ensure Dirichlet boundary condition.

	//Solving of the linear system
	do
	{
		z = z + 1;
		/*Iterations for RED elements */
		for (l = 1; l < lMax - 1; l++)
			for (k = 1; k < kMax - 1; k++)
				for (j = 1; j < jMax - 1; j++)
					for (i = 1; i < iMax - 1; i++)
					{
						if ((i + j + k + l) % 2 == 0)
						{
							// Begin Gauss-Seidel Formula Evaluation
							double gauss_seidel = (prev_u[ijkl(i, j, k, l)] + coef_tauh * (
								(e[ijkl(i, j, k, l)] * u[ijkl(i + 1, j, k, l)])
								+ (w[ijkl(i, j, k, l)] * u[ijkl(i - 1, j, k, l)])
								+ (s[ijkl(i, j, k, l)] * u[ijkl(i, j - 1, k, l)])
								+ (n[ijkl(i, j, k, l)] * u[ijkl(i, j + 1, k, l)])
								+ (b[ijkl(i, j, k, l)] * u[ijkl(i, j, k - 1, l)])
								+ (t[ijkl(i, j, k, l)] * u[ijkl(i, j, k + 1, l)])
								+ (bw[ijkl(i, j, k, l)] * u[ijkl(i, j, k, l - 1)])
								+ (fw[ijkl(i, j, k, l)] * u[ijkl(i, j, k, l + 1)]))) /
								(1 + coef_tauh * (e[ijkl(i, j, k, l)] + w[ijkl(i, j, k, l)] + n[ijkl(i, j, k, l)]
									+ s[ijkl(i, j, k, l)] + t[ijkl(i, j, k, l)] + b[ijkl(i, j, k, l)] + fw[ijkl(i, j, k, l)]
									+ bw[ijkl(i, j, k, l)]));

							// SOR implementation using Gauss-Seidel
							u[ijkl(i, j, k, l)] = u[ijkl(i, j, k, l)] + omega_c * (gauss_seidel - u[ijkl(i, j, k, l)]);
						}
					}

		/* Iterations for BLACK elements */
		for (l = 1; l < lMax - 1; l++)
			for (k = 1; k < kMax - 1; k++)
				for (j = 1; j < jMax - 1; j++)
					for (i = 1; i < iMax - 1; i++)
					{
						if ((i + j + k + l) % 2 == 1)
						{
							// Begin Gauss-Seidel Formula Evaluation
							double gauss_seidel = (prev_u[ijkl(i, j, k, l)] + coef_tauh * (
								(e[ijkl(i, j, k, l)] * u[ijkl(i + 1, j, k, l)])
								+ (w[ijkl(i, j, k, l)] * u[ijkl(i - 1, j, k, l)])
								+ (s[ijkl(i, j, k, l)] * u[ijkl(i, j - 1, k, l)])
								+ (n[ijkl(i, j, k, l)] * u[ijkl(i, j + 1, k, l)])
								+ (b[ijkl(i, j, k, l)] * u[ijkl(i, j, k - 1, l)])
								+ (t[ijkl(i, j, k, l)] * u[ijkl(i, j, k + 1, l)])
								+ (bw[ijkl(i, j, k, l)] * u[ijkl(i, j, k, l - 1)])
								+ (fw[ijkl(i, j, k, l)] * u[ijkl(i, j, k, l + 1)]))) /
								(1 + coef_tauh * (e[ijkl(i, j, k, l)] + w[ijkl(i, j, k, l)] + n[ijkl(i, j, k, l)]
									+ s[ijkl(i, j, k, l)] + t[ijkl(i, j, k, l)] + b[ijkl(i, j, k, l)] + fw[ijkl(i, j, k, l)]
									+ bw[ijkl(i, j, k, l)]));

							// SOR implementation using Gauss-Seidel
							u[ijkl(i, j, k, l)] = u[ijkl(i, j, k, l)] + omega_c * (gauss_seidel - u[ijkl(i, j, k, l)]);
						}
					}

		// Error Evaluation
		mean_square_residue = 0.0;
		for (l = 1; l < lMax - 1; l++)
			for (k = 1; k < kMax - 1; k++)
				for (j = 1; j < jMax - 1; j++)
					for (i = 1; i < iMax - 1; i++)
					{
						mean_square_residue += (pow(u[ijkl(i, j, k, l)] * (1 + coef_tauh * (e[ijkl(i, j, k, l)]
							+ w[ijkl(i, j, k, l)] + n[ijkl(i, j, k, l)] + s[ijkl(i, j, k, l)]
							+ t[ijkl(i, j, k, l)] + b[ijkl(i, j, k, l)] + fw[ijkl(i, j, k, l)]
							+ bw[ijkl(i, j, k, l)]))
							- coef_tauh * (
							(e[ijkl(i, j, k, l)] * u[ijkl(i + 1, j, k, l)])
								+ (w[ijkl(i, j, k, l)] * u[ijkl(i - 1, j, k, l)])
								+ (s[ijkl(i, j, k, l)] * u[ijkl(i, j - 1, k, l)])
								+ (n[ijkl(i, j, k, l)] * u[ijkl(i, j + 1, k, l)])
								+ (b[ijkl(i, j, k, l)] * u[ijkl(i, j, k - 1, l)])
								+ (t[ijkl(i, j, k, l)] * u[ijkl(i, j, k + 1, l)])
								+ (bw[ijkl(i, j, k, l)] * u[ijkl(i, j, k, l - 1)])
								+ (fw[ijkl(i, j, k, l)] * u[ijkl(i, j, k, l + 1)]))
							- prev_u[ijkl(i, j, k, l)], 2) * hhhh);
					}
	} while (mean_square_residue > gsTol && z < maxNoGSIteration);
	printf("Residuum is %e\n", mean_square_residue);
	printf("Step is %d\n", numberOfTimeStep);

	return true;
}

/*function that calculate GS coefficients*/
bool _gs_coefficients_mcf()
{
	int i, j, k, l;
	double quotient = 4.0 * h;
	//calculation of coefficients 
	for (l = 1; l < lMax - 1; l++)
		for (k = 1; k < kMax - 1; k++)
			for (j = 1; j < jMax - 1; j++)
				for (i = 1; i < iMax - 1; i++)
				{
					//calculation of coefficients in the original image data
					// Calculation of coefficients in east direction 
					double _ux = (u[ijkl(i + 1, j, k, l)] - u[ijkl(i, j, k, l)]) / h;

					double _uy = ((u[ijkl(i + 1, j + 1, k, l)] + u[ijkl(i, j + 1, k, l)])
						- (u[ijkl(i + 1, j - 1, k, l)] + u[ijkl(i, j - 1, k, l)])) / quotient;

					double _uz = ((u[ijkl(i + 1, j, k + 1, l)] + u[ijkl(i, j, k + 1, l)])
						- (u[ijkl(i + 1, j, k - 1, l)] + u[ijkl(i, j, k - 1, l)])) / quotient;

					double _uw = ((u[ijkl(i + 1, j, k, l + 1)] + u[ijkl(i, j, k, l + 1)])
						- (u[ijkl(i + 1, j, k, l - 1)] + u[ijkl(i, j, k, l - 1)])) / quotient;
					double _e = sqrt((_ux * _ux) + (_uy * _uy) + (_uz * _uz) + (_uw * _uw) + eps2);

					// Calculation of coefficients in west direction  
					_ux = (u[ijkl(i - 1, j, k, l)] - u[ijkl(i, j, k, l)]) / h;

					_uy = ((u[ijkl(i, j + 1, k, l)] + u[ijkl(i - 1, j + 1, k, l)])
						- (u[ijkl(i, j - 1, k, l)] + u[ijkl(i - 1, j - 1, k, l)])) / quotient;

					_uz = ((u[ijkl(i, j, k + 1, l)] + u[ijkl(i - 1, j, k + 1, l)])
						- (u[ijkl(i, j, k - 1, l)] + u[ijkl(i - 1, j, k - 1, l)])) / quotient;

					_uw = ((u[ijkl(i, j, k, l + 1)] + u[ijkl(i - 1, j, k, l + 1)])
						- (u[ijkl(i, j, k, l - 1)] + u[ijkl(i - 1, j, k, l - 1)])) / quotient;
					double _w = sqrt((_ux * _ux) + (_uy * _uy) + (_uz * _uz) + (_uw * _uw) + eps2);

					// Calculation of coefficients in north direction  
					_ux = ((u[ijkl(i + 1, j + 1, k, l)] + u[ijkl(i + 1, j, k, l)])
						- (u[ijkl(i - 1, j + 1, k, l)] + u[ijkl(i - 1, j, k, l)])) / quotient;

					_uy = (u[ijkl(i, j + 1, k, l)] - u[ijkl(i, j, k, l)]) / h;

					_uz = ((u[ijkl(i, j + 1, k + 1, l)] + u[ijkl(i, j, k + 1, l)])
						- (u[ijkl(i, j + 1, k - 1, l)] + u[ijkl(i, j, k - 1, l)])) / quotient;

					_uw = ((u[ijkl(i, j + 1, k, l + 1)] + u[ijkl(i, j, k, l + 1)])
						- (u[ijkl(i, j + 1, k, l - 1)] + u[ijkl(i, j, k, l - 1)])) / quotient;
					double _n = sqrt((_ux * _ux) + (_uy * _uy) + (_uz * _uz) + (_uw * _uw) + eps2);

					// Calculation of coefficients in south direction  
					_ux = ((u[ijkl(i + 1, j - 1, k, l)] + u[ijkl(i + 1, j, k, l)])
						- (u[ijkl(i - 1, j - 1, k, l)] + u[ijkl(i - 1, j, k, l)])) / quotient;

					_uy = (u[ijkl(i, j - 1, k, l)] - u[ijkl(i, j, k, l)]) / h;

					_uz = ((u[ijkl(i, j - 1, k + 1, l)] + u[ijkl(i, j, k + 1, l)])
						- (u[ijkl(i, j - 1, k - 1, l)] + u[ijkl(i, j, k - 1, l)])) / quotient;

					_uw = ((u[ijkl(i, j - 1, k, l + 1)] + u[ijkl(i, j, k, l + 1)])
						- (u[ijkl(i, j - 1, k, l - 1)] + u[ijkl(i, j, k, l - 1)])) / quotient;
					double _s = sqrt((_ux * _ux) + (_uy * _uy) + (_uz * _uz) + (_uw * _uw) + eps2);

					// Calculation of coefficients in top direction  
					_ux = ((u[ijkl(i + 1, j, k + 1, l)] + u[ijkl(i + 1, j, k, l)])
						- (u[ijkl(i - 1, j, k + 1, l)] + u[ijkl(i - 1, j, k, l)])) / quotient;

					_uy = ((u[ijkl(i, j + 1, k + 1, l)] + u[ijkl(i, j + 1, k, l)])
						- (u[ijkl(i, j - 1, k + 1, l)] + u[ijkl(i, j - 1, k, l)])) / quotient;

					_uz = (u[ijkl(i, j, k + 1, l)] - u[ijkl(i, j, k, l)]) / h;

					_uw = ((u[ijkl(i, j, k + 1, l + 1)] + u[ijkl(i, j, k, l + 1)])
						- (u[ijkl(i, j, k + 1, l - 1)] + u[ijkl(i, j, k, l - 1)])) / quotient;
					double _t = sqrt((_ux * _ux) + (_uy * _uy) + (_uz * _uz) + (_uw * _uw) + eps2);

					// Calculation of coefficients in bottom direction  
					_ux = ((u[ijkl(i + 1, j, k - 1, l)] + u[ijkl(i + 1, j, k, l)])
						- (u[ijkl(i - 1, j, k - 1, l)] + u[ijkl(i - 1, j, k, l)])) / quotient;

					_uy = ((u[ijkl(i, j + 1, k - 1, l)] + u[ijkl(i, j + 1, k, l)])
						- (u[ijkl(i, j - 1, k - 1, l)] + u[ijkl(i, j - 1, k, l)])) / quotient;

					_uz = (u[ijkl(i, j, k - 1, l)] - u[ijkl(i, j, k, l)]) / h;

					_uw = ((u[ijkl(i, j, k - 1, l + 1)] + u[ijkl(i, j, k, l + 1)])
						- (u[ijkl(i, j, k - 1, l - 1)] + u[ijkl(i, j, k, l - 1)])) / quotient;
					double _b = sqrt((_ux * _ux) + (_uy * _uy) + (_uz * _uz) + (_uw * _uw) + eps2);

					// Calculation of coefficients in forward direction 
					_ux = ((u[ijkl(i + 1, j, k, l + 1)] + u[ijkl(i + 1, j, k, l)])
						- (u[ijkl(i - 1, j, k, l + 1)] + u[ijkl(i - 1, j, k, l)])) / quotient;

					_uy = ((u[ijkl(i, j + 1, k, l + 1)] + u[ijkl(i, j + 1, k, l)])
						- (u[ijkl(i, j - 1, k, l + 1)] + u[ijkl(i, j - 1, k, l)])) / quotient;

					_uz = ((u[ijkl(i, j, k + 1, l + 1)] + u[ijkl(i, j, k + 1, l)])
						- (u[ijkl(i, j, k - 1, l + 1)] + u[ijkl(i, j, k - 1, l)])) / quotient;

					_uw = (u[ijkl(i, j, k, l + 1)] - u[ijkl(i, j, k, l)]) / h;
					double _fw = sqrt((_ux * _ux) + (_uy * _uy) + (_uz * _uz) + (_uw * _uw) + eps2);


					// Calculation of coefficients in backward direction
					_ux = ((u[ijkl(i + 1, j, k, l - 1)] + u[ijkl(i + 1, j, k, l)])
						- (u[ijkl(i - 1, j, k, l - 1)] + u[ijkl(i - 1, j, k, l)])) / quotient;

					_uy = ((u[ijkl(i, j + 1, k, l - 1)] + u[ijkl(i, j + 1, k, l)])
						- (u[ijkl(i, j - 1, k, l - 1)] + u[ijkl(i, j - 1, k, l)])) / quotient;

					_uz = ((u[ijkl(i, j, k + 1, l - 1)] + u[ijkl(i, j, k + 1, l)])
						- (u[ijkl(i, j, k - 1, l - 1)] + u[ijkl(i, j, k - 1, l)])) / quotient;

					_uw = (u[ijkl(i, j, k, l - 1)] - u[ijkl(i, j, k, l)]) / h;
					double _bw = sqrt((_ux * _ux) + (_uy * _uy) + (_uz * _uz) + (_uw * _uw) + eps2);


					//Evaluation of norm of gradient of image at each voxel
					double average_face_coef = ((_e + _w + _n + _s + _t + _b + _fw + _bw) / 8.0);

					double voxel_coef = sqrt(pow(average_face_coef, 2) + eps2);

					/*Evaluation of norm of gradient of image at each voxel and 
					reciprocal of norm of gradient of image at each voxel face*/
					e[ijkl(i, j, k, l)] = voxel_coef * (1.0 / _e);//east coefficient
					w[ijkl(i, j, k, l)] = voxel_coef * (1.0 / _w);//west coefficient
					n[ijkl(i, j, k, l)] = voxel_coef * (1.0 / _n);//north coefficient
					s[ijkl(i, j, k, l)] = voxel_coef * (1.0 / _s);//south coefficient
					t[ijkl(i, j, k, l)] = voxel_coef * (1.0 / _t);//top coefficient
					b[ijkl(i, j, k, l)] = voxel_coef * (1.0 / _b);//bottom coefficient
					fw[ijkl(i, j, k, l)] = voxel_coef * (1.0 / _fw);//forward coefficient
					bw[ijkl(i, j, k, l)] = voxel_coef * (1.0 / _bw);//backward coefficient
				}
	return true;
}

bool _set4DBoundaryToExactValues_mcf(double t)
{
	int k, i, j, l;
	double x, y, z, w;

	// X Direction
	for (l = 0; l < lMax; l++)
		for (k = 0; k < kMax; k++)
			for (j = 0; j < jMax; j++)
			{
				x = -(h / 2.) - 1.25;
				y = (j - 1) * h + (h / 2.) - 1.25;
				z = (k - 1) * h + (h / 2.) - 1.25;
				w = (l - 1) * h + (h / 2.) - 1.25;
				u[ijkl(0, j, k, l)] = ((x * x + y * y + z * z + w * w - 1.0) / 6.0) + t;
			}
	// X Direction
	for (l = 0; l < lMax; l++)
		for (k = 0; k < kMax; k++)
			for (j = 0; j < jMax; j++)
			{
				x = (h / 2.) + 1.25;
				y = (j - 1) * h + (h / 2.) - 1.25;
				z = (k - 1) * h + (h / 2.) - 1.25;
				w = (l - 1) * h + (h / 2.) - 1.25;
				u[ijkl(iMax - 1, j, k, l)] = ((x * x + y * y + z * z + w * w - 1.0) / 6.0) + t;
			}

	// Y Direction
	for (l = 0; l < lMax; l++)
		for (k = 0; k < kMax; k++)
			for (i = 0; i < iMax; i++)
			{
				x = (i - 1) * h + (h / 2.) - 1.25;
				y = -(h / 2.) - 1.25;
				z = (k - 1) * h + (h / 2.) - 1.25;
				w = (l - 1) * h + (h / 2.) - 1.25;
				u[ijkl(i, 0, k, l)] = ((x * x + y * y + z * z + w * w - 1.0) / 6.0) + t;
			}
	// Y Direction
	for (l = 0; l < lMax; l++)
		for (k = 0; k < kMax; k++)
			for (i = 0; i < iMax; i++)
			{
				x = (i - 1) * h + (h / 2.) - 1.25;
				y = (h / 2.) + 1.25;
				z = (k - 1) * h + (h / 2.) - 1.25;
				w = (l - 1) * h + (h / 2.) - 1.25;
				u[ijkl(i, jMax - 1, k, l)] = ((x * x + y * y + z * z + w * w - 1.0) / 6.0) + t;
			}

	// Z Direction
	for (l = 0; l < lMax; l++)
		for (j = 0; j < jMax; j++)
			for (i = 0; i < iMax; i++)
			{
				x = (i - 1) * h + (h / 2.) - 1.25;
				y = (j - 1) * h + (h / 2.) - 1.25;
				z = -(h / 2.) - 1.25;
				w = (l - 1) * h + (h / 2.) - 1.25;
				u[ijkl(i, j, 0, l)] = ((x * x + y * y + z * z + w * w - 1.0) / 6.0) + t;
			}

	// Z Direction
	for (l = 0; l < lMax; l++)
		for (j = 0; j < jMax; j++)
			for (i = 0; i < iMax; i++)
			{
				x = (i - 1) * h + (h / 2.) - 1.25;
				y = (j - 1) * h + (h / 2.) - 1.25;
				z = (h / 2.) + 1.25;
				w = (l - 1) * h + (h / 2.) - 1.25;
				u[ijkl(i, j, kMax - 1, l)] = ((x * x + y * y + z * z + w * w - 1.0) / 6.0) + t;
			}

	// W Direction
	for (k = 0; k < kMax; k++)
		for (i = 0; i < iMax; i++)
			for (j = 0; j < jMax; j++)
			{
				x = (i - 1) * h + (h / 2.) - 1.25;
				y = (j - 1) * h + (h / 2.) - 1.25;
				z = (k - 1) * h + (h / 2.) - 1.25;
				w = -(h / 2.) - 1.25;
				u[ijkl(i, j, k, 0)] = ((x * x + y * y + z * z + w * w - 1.0) / 6.0) + t;
			}

	// W Direction
	for (k = 0; k < kMax; k++)
		for (i = 0; i < iMax; i++)
			for (j = 0; j < jMax; j++)
			{
				x = (i - 1) * h + (h / 2.) - 1.25;
				y = (j - 1) * h + (h / 2.) - 1.25;
				z = (k - 1) * h + (h / 2.) - 1.25;
				w = (h / 2.) + 1.25;
				u[ijkl(i, j, k, lMax - 1)] = ((x * x + y * y + z * z + w * w - 1.0) / 6.0) + t;
			}
	return true;
}
void _exactSolution_mcf(double t)
{
	int l, k, i, j;
	double x, y, z, w;
	// Contruction of exact condition.
	for (l = 0; l < lMax; l++)
		for (k = 0; k < kMax; k++)
			for (j = 0; j < jMax; j++)
				for (i = 0; i < iMax; i++)
				{
					x = ((i - 1) * h) + (h / 2.) - 1.25;
					y = ((j - 1) * h) + (h / 2.) - 1.25;
					z = ((k - 1) * h) + (h / 2.) - 1.25;
					w = ((l - 1) * h) + (h / 2.) - 1.25;
					// Fill Value
					exact_u[ijkl(i, j, k, l)] = ((x * x + y * y + z * z + w * w - 1.0) / 6.0) + t;
				}

}
void _copy4DdataFromSrcToDest_mcf(double *srcDataPtr, double *destDataPtr)
{
	int i, j, k, l;
	for (l = 0; l < lMax; l++)
		for (k = 0; k < kMax; k++)
			for (j = 0; j < jMax; j++)
				for (i = 0; i < iMax; i++)
					destDataPtr[ijkl(i, j, k, l)] = srcDataPtr[ijkl(i, j, k, l)];
}

int main(int argc, char **argv)
{
	double difference_btw_exact_and_numerical_sol;
	double CPUT1, CPUT2;

	iMax = 162;
	jMax = 162;
	kMax = 162;
	lMax = 162;

	maxNoGSIteration = 1000;
	maxNoOfTimeSteps = 256;
	mcfTol = 1.0e-12;
	omega_c = 1.8;
	gsTol = 1.0e-12;
	h = 2.5 / (iMax - 2);
	tau = (h * h);
	eps2 = h * h;

	_allocateMem_mcf();
	
	//Note the initial time
	CPUT1 = clock() / (double)(CLOCKS_PER_SEC);
	//Construct initial condition
	_exactSolution_mcf(0.);

	//Copy exact solution to current solution
	_copy4DdataFromSrcToDest_mcf(exact_u, u);
	   
	//loop for mcf time steps
	int tCount = 1;
	double time;
	do
	{
		numberOfTimeStep = tCount;
		time = numberOfTimeStep * tau;
		// Call to function that will evolve mcf in each discrete time step
		_mcfTimeStep();
		
		_exactSolution_mcf(time);
		//Compute the L2 norm of the difference between the exact and numerical solutions
		difference_btw_exact_and_numerical_sol = _l2norm_mcf(exact_u, u);
		printf("l2 norm of error at time %lf is %e\n", time, difference_btw_exact_and_numerical_sol);
		tCount++;
	} while ((tCount <= maxNoOfTimeSteps));

	//Note the final time
	CPUT2 = clock() / (double)(CLOCKS_PER_SEC);
	printf("CPU time is: %e secs\n", CPUT2 - CPUT1);
	_deallocateMem_mcf();
	return 0;
}