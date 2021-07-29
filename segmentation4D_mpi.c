/* Author: Markjoe Olunna UBA
 * Purpose: ImageInLife project - 4D Image Segmentation Methods
 * Language:  C */
#include <stdio.h> 
#include <stdlib.h>
#include <time.h>
#include <math.h> 
#include <string.h> 
#include <stdbool.h>
#include <ctype.h>
#include <mpi.h>
#include "segmentation_4D_mpi.h"

 // Image Dimensions
int kMax, iMax, jMax, lMax; // Absolute dimensions plus 2 in each direction e.g., n1 == iMax - 2
int no_of_processes, process_ID;
int procs_lMax; // procs_lMax = (int)ceil((lMax - 2) / no_of_processes);

double *u; //  Pointer to the numerical solution
double *prev_u; // Pointer to the Previous solution
double *i0; // Pointer to the exact solution
double *Mmask;// Pointer to the mask for checking whether a doxel in initial condition is updated or not

double *e; // East coefficient pointer
double *w; // West coefficient pointer
double *n; // North coefficient pointer
double *s; // South coefficient pointer
double *t; // Top coefficient pointer
double *b; // Bottom coefficient pointer
double *fw; // Forward coefficient pointer
double *bw; // Backward coefficient pointer

double *g_e; // East coefficient pointer
double *g_w; // West coefficient pointer
double *g_n; // North coefficient pointer
double *g_s; // South coefficient pointer
double *g_t; // Top coefficient pointer
double *g_b; // Bottom coefficient pointer
double *g_fw; // Forward coefficient pointer
double *g_bw; // Backward coefficient pointer

int *no_of_centers;
double **center_x, **center_y, **center_z;
double spacing_x, spacing_y, spacing_z;
double origin_x, origin_y, origin_z;

int maxNoGSIteration;// Maximum number of Gauss-Seidel iterations
double eps2; // epsilon is the regularization factor (Evans-Spruck)
double K; // constant K in the Perona-Malik function G for the image
int numberOfCurrentTimeStep;// Number of current time step
int maxNoOfTimeSteps;// Maximum number of time step
double segTol; // Tolerance for stopping of the evolution process
double tau, h, omega_c, omega_sigma, gsTol; /* h is the Grid size, tau is time step for the evolution process,
omega_c is the relaxation parameter in SOR implementation using Gauss-Seidel, gsTol is the acceptable
tolerance for Gauss-Seidel iterations*/
double theta_img, rho_thr, alpha, beta;
double sigma; //sigma is time step for presmoothing
double maxNoGSIteration_sigma;// Maximum number of Gauss-Seidel iterations for presmoothing
int timeStepsNum;
double ballRadius;

FILE *input_file, *input_center_file, *output_file;
int ijkl(int i, int j, int k, int l)
{
	return l * iMax * jMax * kMax + k * iMax * jMax + j * iMax + i;
	// or equivalently return ((l * kMax + k) * jMax + j) * iMax + i
}

double _gFunction_mpi_(double value, double coef)
{
	return 1.0 / (1 + coef * value);
}

bool _allocateMem_mpi_()
{
	int i, j, k, l;
	int dim4D = iMax * jMax *kMax * (procs_lMax + 2);

	prev_u = (double *)malloc(sizeof(double) * dim4D);
	u = (double *)malloc(sizeof(double) * dim4D);
	i0 = (double *)malloc(sizeof(double) * dim4D);
	Mmask = (double *)malloc(sizeof(double) * dim4D);

	e = (double *)malloc(sizeof(double) * dim4D);
	w = (double *)malloc(sizeof(double) * dim4D);
	n = (double *)malloc(sizeof(double) * dim4D);
	s = (double *)malloc(sizeof(double) * dim4D);
	t = (double *)malloc(sizeof(double) * dim4D);
	b = (double *)malloc(sizeof(double) * dim4D);
	fw = (double *)malloc(sizeof(double) * dim4D);
	bw = (double *)malloc(sizeof(double) * dim4D);

	g_e = (double *)malloc(sizeof(double) * dim4D);
	g_w = (double *)malloc(sizeof(double) * dim4D);
	g_n = (double *)malloc(sizeof(double) * dim4D);
	g_s = (double *)malloc(sizeof(double) * dim4D);
	g_t = (double *)malloc(sizeof(double) * dim4D);
	g_b = (double *)malloc(sizeof(double) * dim4D);
	g_fw = (double *)malloc(sizeof(double) * dim4D);
	g_bw = (double *)malloc(sizeof(double) * dim4D);


	//checks if the memory was allocated
	if (i0 == NULL || prev_u == NULL || u == NULL || e == NULL || w == NULL || n == NULL ||
		s == NULL || t == NULL || b == NULL || fw == NULL || bw == NULL || g_e == NULL || g_w == NULL ||
		g_n == NULL || g_s == NULL || g_t == NULL || g_b == NULL || g_fw == NULL || g_bw == NULL)
		return false;

	//Initialize all arrays to zero
	for (l = 0; l <= (procs_lMax + 1); l++)
		for (k = 0; k <= kMax - 1; k++)
			for (j = 0; j <= jMax - 1; j++)
				for (i = 0; i <= iMax - 1; i++)
				{
					u[ijkl(i, j, k, l)] = 0.;
					i0[ijkl(i, j, k, l)] = 0.;
					prev_u[ijkl(i, j, k, l)] = 0.;
					Mmask[ijkl(i, j, k, l)] = 0.;

					e[ijkl(i, j, k, l)] = 0.;
					w[ijkl(i, j, k, l)] = 0.;
					n[ijkl(i, j, k, l)] = 0.;
					s[ijkl(i, j, k, l)] = 0.;
					t[ijkl(i, j, k, l)] = 0.;
					b[ijkl(i, j, k, l)] = 0.;
					fw[ijkl(i, j, k, l)] = 0.;
					bw[ijkl(i, j, k, l)] = 0.;

					g_e[ijkl(i, j, k, l)] = 0.;
					g_w[ijkl(i, j, k, l)] = 0.;
					g_n[ijkl(i, j, k, l)] = 0.;
					g_s[ijkl(i, j, k, l)] = 0.;
					g_t[ijkl(i, j, k, l)] = 0.;
					g_b[ijkl(i, j, k, l)] = 0.;
					g_fw[ijkl(i, j, k, l)] = 0.;
					g_bw[ijkl(i, j, k, l)] = 0.;
				}

	center_x = (double **)malloc((procs_lMax + 2) * sizeof(double*));
	center_y = (double **)malloc((procs_lMax + 2) * sizeof(double*));
	center_z = (double **)malloc((procs_lMax + 2) * sizeof(double*));
	//checks if the memory was allocated
	if (center_x == NULL || center_y == NULL || center_z == NULL)
		return false;

	for (l = 0; l <= (procs_lMax + 1); l++)
	{
		center_x[l] = (double *)malloc(no_of_centers[l] * sizeof(double));
		center_y[l] = (double *)malloc(no_of_centers[l] * sizeof(double));
		center_z[l] = (double *)malloc(no_of_centers[l] * sizeof(double));
		//checks if the memory was allocated
		if (center_x[l] == NULL || center_y[l] == NULL || center_z[l] == NULL)
			return false;
	}


	return true;
}
bool _deallocateMem_mpi_()
{
	int l;
	free(prev_u);
	free(i0);
	free(u);
	free(Mmask);

	free(e);
	free(w);
	free(n);
	free(s);
	free(t);
	free(b);
	free(fw);
	free(bw);

	free(g_e);
	free(g_w);
	free(g_n);
	free(g_s);
	free(g_t);
	free(g_b);
	free(g_fw);
	free(g_bw);

	for (l = 0; l <= (procs_lMax + 1); l++)
	{
		free(center_x[l]);
		free(center_y[l]);
		free(center_z[l]);
	}
	free(center_x);
	free(center_y);
	free(center_z);

	return true;
}

double _l2norm_mpi_(double *dataArray3DPtr1, double *dataArray3DPtr2)
{
	//checks if the memory was allocated
	if (dataArray3DPtr1 == NULL || dataArray3DPtr2 == NULL)
		return false;
	double localSumPower = 0, globalSumPower;
	double hhhh = h * h * h * h;
	int i, j, k, l, l_start, l_stop;

	//checks if the memory was allocated
	if (dataArray3DPtr1 == NULL || dataArray3DPtr2 == NULL)
		return false;

	if (process_ID == 0)
	{
		l_start = 0;
		l_stop = procs_lMax;
	}

	if (process_ID == no_of_processes - 1)
	{
		l_start = 1;
		l_stop = procs_lMax + 1;
	}

	if (process_ID > 0 && process_ID < no_of_processes - 1)
	{
		l_start = 1;
		l_stop = procs_lMax;
	}
	/*Computation of norm*/
	for (l = l_start; l <= l_stop; l++)
		for (k = 0; k <= kMax - 1; k++)
			for (j = 0; j <= jMax - 1; j++)
				for (i = 0; i <= iMax - 1; i++)
					localSumPower += (pow(dataArray3DPtr1[ijkl(i, j, k, l)] - dataArray3DPtr2[ijkl(i, j, k, l)], 2) * hhhh);
	
	//Get the global sum of the powers
	MPI_Allreduce(&localSumPower, &globalSumPower, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	return sqrt(globalSumPower);
}

double _mass4D_mpi_(double *dataArray3DPtr1)
{
	//checks if the memory was allocated
	if (dataArray3DPtr1 == NULL)
		return false;

	int i, j, k, l, l_start, l_stop;
	double localSumMass = 0, globalSumMass;
	double hhhh = h * h * h * h;

	if (process_ID == 0)
	{
		l_start = 0;
		l_stop = procs_lMax;
	}

	if (process_ID == no_of_processes - 1)
	{
		l_start = 1;
		l_stop = procs_lMax + 1;
	}	
	
	if (process_ID > 0 && process_ID < no_of_processes - 1)
	{
		l_start = 1;
		l_stop = procs_lMax;
	}
	/*Computation of mass*/
	for (l = l_start; l <= l_stop; l++)
		for (k = 0; k <= kMax - 1; k++)	
			for (j = 0; j <= jMax - 1; j++)
				for (i = 0; i <= iMax - 1; i++)
					localSumMass += (dataArray3DPtr1[ijkl(i, j, k, l)] * hhhh);
	
	//Get the global sum of the powers
	MPI_Allreduce(&localSumMass, &globalSumMass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	return globalSumMass;
}


bool _segTimeStep_mpi_()
{
	int z = 0; // Steps counter
	int i, j, k, l;
	double coef_tauh = tau / (h * h);
	double local_mean_square_residue, global_mean_square_residue;
	double hhhh = h * h * h * h;
	int N = (int)ceil((lMax - 2) / no_of_processes);
	 
	//Copy current solution to previous solution
	_copy4DdataFromSrcToDest_mpi_(u, prev_u);

	//Calculation of coefficient
	_gs_coefficients_mpi_();

	//Solving of the linear system
	do
	{
		z = z + 1;
		/*Iterations for RED elements */
		for (l = 1; l <= procs_lMax; l++)
			for (k = 1; k <= kMax - 2; k++)
				for (j = 1; j <= jMax - 2; j++)
					for (i = 1; i <= iMax - 2; i++)
					{
						if ((N * process_ID + i + j + k + l) % 2 == 0)
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

		/* Communication*/
		//Exchange volumes between processors
		_communication_mpi_(u);

		/* Iterations for BLACK elements */
		for (l = 1; l <= procs_lMax; l++)
			for (k = 1; k <= kMax - 2; k++)
				for (j = 1; j <= jMax - 2; j++)
					for (i = 1; i <= iMax - 2; i++)
					{
						if ((N * process_ID + i + j + k + l) % 2 == 1)
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

		/* Communication*/
		//Exchange volumes between processors
		_communication_mpi_(u);

		// Error Evaluation
		local_mean_square_residue = 0.0;
		for (l = 1; l <= procs_lMax; l++)
			for (k = 1; k <= kMax - 2; k++)
				for (j = 1; j <= jMax - 2; j++)
					for (i = 1; i <= iMax - 2; i++)
					{
						local_mean_square_residue += (pow(u[ijkl(i, j, k, l)] * (1 + coef_tauh * (e[ijkl(i, j, k, l)]
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

		MPI_Allreduce(&local_mean_square_residue, &global_mean_square_residue, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	} while (global_mean_square_residue > gsTol && z < maxNoGSIteration);
	if (process_ID == 0)
	{
		printf("Step is %d\n", numberOfCurrentTimeStep);
		printf("Mean square residue is %e and total number of RBSOR is %d\n", global_mean_square_residue, z);
	}
	
	_rescale4DdataToZeroOne_mpi_(u);

	return true;
}

bool _presmoothStep_mpi_(double *u)
{
	int z = 0; // Steps counter
	int i, j, k, l;
	double coef_tauh = sigma / (h * h);
	double local_mean_square_residue, global_mean_square_residue;
	double hhhh = h * h * h * h;
	int N = (int)ceil((lMax - 2) / no_of_processes);
	
	//Copy current solution to previous solution
	_copy4DdataFromSrcToDest_mpi_(u, prev_u);

	//set boundary values to ensure zero Dirichlet boundary condition.
	_reflection4Ddata_mpi_(u);
	
	//Exchange boundary volumes
	_communication_mpi_(u);

	//Solving of the linear system
	do
	{
		z = z + 1;
		/*Iterations for RED elements */
		for (l = 1; l <= procs_lMax; l++)
			for (k = 1; k <= kMax - 2; k++)
				for (j = 1; j <= jMax - 2; j++)
					for (i = 1; i <= iMax - 2; i++)
					{
						if ((N * process_ID + i + j + k + l) % 2 == 0)
						{
							// Begin Gauss-Seidel Formula Evaluation
							double gauss_seidel = (prev_u[ijkl(i, j, k, l)] + coef_tauh * (
								u[ijkl(i + 1, j, k, l)] + u[ijkl(i - 1, j, k, l)] + u[ijkl(i, j - 1, k, l)] + u[ijkl(i, j + 1, k, l)]
								+ u[ijkl(i, j, k - 1, l)] + u[ijkl(i, j, k + 1, l)] + u[ijkl(i, j, k, l - 1)] + u[ijkl(i, j, k, l + 1)])) /
								(1 + coef_tauh * 8);

							// SOR implementation using Gauss-Seidel
							u[ijkl(i, j, k, l)] = u[ijkl(i, j, k, l)] + omega_sigma * (gauss_seidel - u[ijkl(i, j, k, l)]);
						}
					}

		/* Communication*/
		//Exchange volumes between processors
		_communication_mpi_(u);

		/* Iterations for BLACK elements */
		for (l = 1; l <= procs_lMax; l++)
			for (k = 1; k <= kMax - 2; k++)
				for (j = 1; j <= jMax - 2; j++)
					for (i = 1; i <= iMax - 2; i++)
					{
						if ((N * process_ID + i + j + k + l) % 2 == 1)
						{
							// Begin Gauss-Seidel Formula Evaluation
							double gauss_seidel = (prev_u[ijkl(i, j, k, l)] + coef_tauh * (
								u[ijkl(i + 1, j, k, l)] + u[ijkl(i - 1, j, k, l)] + u[ijkl(i, j - 1, k, l)] + u[ijkl(i, j + 1, k, l)]
								+ u[ijkl(i, j, k - 1, l)] + u[ijkl(i, j, k + 1, l)] + u[ijkl(i, j, k, l - 1)] + u[ijkl(i, j, k, l + 1)])) /
								(1 + coef_tauh * 8);

							// SOR implementation using Gauss-Seidel
							u[ijkl(i, j, k, l)] = u[ijkl(i, j, k, l)] + omega_sigma * (gauss_seidel - u[ijkl(i, j, k, l)]);
						}
					}

		/* Communication*/
		//Exchange volumes between processors
		_communication_mpi_(u);

		// Error Evaluation
		local_mean_square_residue = 0.0;
		for (l = 1; l <= procs_lMax; l++)
			for (k = 1; k <= kMax - 2; k++)
				for (j = 1; j <= jMax - 2; j++)
					for (i = 1; i <= iMax - 2; i++)
					{
						local_mean_square_residue += (pow(u[ijkl(i, j, k, l)] * (1 + coef_tauh * 8) - coef_tauh * (
							u[ijkl(i + 1, j, k, l)] + u[ijkl(i - 1, j, k, l)] + u[ijkl(i, j - 1, k, l)] + u[ijkl(i, j + 1, k, l)]
							+ u[ijkl(i, j, k - 1, l)] + u[ijkl(i, j, k + 1, l)] + u[ijkl(i, j, k, l - 1)] + u[ijkl(i, j, k, l + 1)])
							- prev_u[ijkl(i, j, k, l)], 2) * hhhh);
					}
		MPI_Allreduce(&local_mean_square_residue, &global_mean_square_residue, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	} while (global_mean_square_residue > gsTol && z < maxNoGSIteration_sigma);
	if (process_ID == 0)
		printf("Presmoothing mean square residue is %e and total number of RBSOR is %d\n", global_mean_square_residue, z);
	return true;
}
void _communication_mpi_(double *u)
{
	int req1, req2, req3, req4;
	MPI_Status status;

	/* Communication*/
	//Exchange volumes between processors
	if (process_ID == 0)
	{
		//send 3D array
		MPI_Isend(&(u[ijkl(0, 0, 0, procs_lMax)]), iMax * jMax * kMax, MPI_DOUBLE, process_ID + 1, 1, MPI_COMM_WORLD, &req1);

		//send 3D array
		MPI_Irecv(&(u[ijkl(0, 0, 0, procs_lMax + 1)]), iMax * jMax * kMax, MPI_DOUBLE, process_ID + 1, 1, MPI_COMM_WORLD, &req2);
	}
	if ((process_ID > 0) && (process_ID < no_of_processes - 1))
	{
		//send 3D array
		MPI_Isend(&(u[ijkl(0, 0, 0, 1)]), iMax * jMax * kMax, MPI_DOUBLE, process_ID - 1, 1, MPI_COMM_WORLD, &req1);
		MPI_Isend(&(u[ijkl(0, 0, 0, procs_lMax)]), iMax * jMax * kMax, MPI_DOUBLE, process_ID + 1, 1, MPI_COMM_WORLD, &req2);

		//receive 3D array
		MPI_Irecv(&(u[ijkl(0, 0, 0, 0)]), iMax * jMax * kMax, MPI_DOUBLE, process_ID - 1, 1, MPI_COMM_WORLD, &req3);
		MPI_Irecv(&(u[ijkl(0, 0, 0, procs_lMax + 1)]), iMax * jMax * kMax, MPI_DOUBLE, process_ID + 1, 1, MPI_COMM_WORLD, &req4);
	}
	if (process_ID == no_of_processes - 1)
	{
		//send 3D array
		MPI_Isend(&(u[ijkl(0, 0, 0, 1)]), iMax * jMax * kMax, MPI_DOUBLE, process_ID - 1, 1, MPI_COMM_WORLD, &req1);

		//receive 3D array
		MPI_Irecv(&(u[ijkl(0, 0, 0, 0)]), iMax * jMax * kMax, MPI_DOUBLE, process_ID - 1, 1, MPI_COMM_WORLD, &req2);
	}

	MPI_Wait(&req1, &status);
	MPI_Wait(&req2, &status);
	if (process_ID > 0 && process_ID < no_of_processes - 1)
	{
		MPI_Wait(&req3, &status);
		MPI_Wait(&req4, &status);
	}
}

void _presmoothStepEXP_mpi_(double *u)
{
	int i, j, k, l, t;
	double hh = h * h;
	double coeff = sigma / hh;
	//Copy current solution to previous solution
	_copy4DdataFromSrcToDest_mpi_(u, prev_u);

	//set boundary values to ensure zero Dirichlet boundary condition.
	_reflection4Ddata_mpi_(u);

	//Exchange volumes between processors
	_communication_mpi_(u);

	// The Explicit Scheme Evaluation
	for (t = 0; t < timeStepsNum; t++)
	{

		for (l = 1; l <= procs_lMax; l++)
			for (k = 1; k <= kMax - 2; k++)
				for (j = 1; j <= jMax - 2; j++)
					for (i = 1; i < iMax - 2; i++)
					{
						// Explicit formula
						u[ijkl(i, j, k, l)] = (1.0 - 8.0 * coeff) * prev_u[ijkl(i, j, k, l)]
							+ coeff * (prev_u[ijkl(i + 1, j, k, l)]
								+ prev_u[ijkl(i - 1, j, k, l)]
								+ prev_u[ijkl(i, j + 1, k, l)]
								+ prev_u[ijkl(i, j - 1, k, l)]
								+ prev_u[ijkl(i, j, k + 1, l)]
								+ prev_u[ijkl(i, j, k - 1, l)]
								+ prev_u[ijkl(i, j, k, l + 1)]
								+ prev_u[ijkl(i, j, k, l - 1)]);
					}

		_reflection4Ddata_mpi_(u);
	}
}
/*function that calculate GS coefficients*/
bool _gs_coefficients_mpi_()
{
	int i, j, k, l;
	double quotient = 4.0 * h;
	
	//calculation of coefficients 
	for (l = 1; l <= procs_lMax; l++)
		for (k = 1; k <= kMax - 2; k++)
			for (j = 1; j <= jMax - 2; j++)
				for (i = 1; i <= iMax - 2; i++)
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
					e[ijkl(i, j, k, l)] = voxel_coef * g_e[ijkl(i, j, k, l)] * (1.0 / _e);//east coefficient
					w[ijkl(i, j, k, l)] = voxel_coef * g_w[ijkl(i, j, k, l)] * (1.0 / _w);//west coefficient
					n[ijkl(i, j, k, l)] = voxel_coef * g_n[ijkl(i, j, k, l)] * (1.0 / _n);//north coefficient
					s[ijkl(i, j, k, l)] = voxel_coef * g_s[ijkl(i, j, k, l)] * (1.0 / _s);//south coefficient
					t[ijkl(i, j, k, l)] = voxel_coef * g_t[ijkl(i, j, k, l)] * (1.0 / _t);//top coefficient
					b[ijkl(i, j, k, l)] = voxel_coef * g_b[ijkl(i, j, k, l)] * (1.0 / _b);//bottom coefficient
					fw[ijkl(i, j, k, l)] = voxel_coef * g_fw[ijkl(i, j, k, l)] * (1.0 / _fw);//forward coefficient
					bw[ijkl(i, j, k, l)] = voxel_coef * g_bw[ijkl(i, j, k, l)] * (1.0 / _bw);//backward coefficient
				}
	return true;
}

bool _gf_inputImage_plus_thresholdedImage_mpi_(double *i0, double *i0_thr)
{
	int i, j, k, l;
	double quotient = 4.0 * h;

	_presmoothStep_mpi_(i0);
	_presmoothStep_mpi_(i0_thr);

	//calculation of coefficients 
	for (l = 1; l <= procs_lMax; l++)
		for (k = 1; k <= kMax - 2; k++)
			for (j = 1; j <= jMax - 2; j++)
				for (i = 1; i <= iMax - 2; i++)
				{
					//calculation of coefficients in the original image data
					// Calculation of coefficients in east direction 
					//input image contribution
					double _ux = (i0[ijkl(i + 1, j, k, l)] - i0[ijkl(i, j, k, l)]) / h;

					double _uy = ((i0[ijkl(i + 1, j + 1, k, l)] + i0[ijkl(i, j + 1, k, l)])
						- (i0[ijkl(i + 1, j - 1, k, l)] + i0[ijkl(i, j - 1, k, l)])) / quotient;

					double _uz = ((i0[ijkl(i + 1, j, k + 1, l)] + i0[ijkl(i, j, k + 1, l)])
						- (i0[ijkl(i + 1, j, k - 1, l)] + i0[ijkl(i, j, k - 1, l)])) / quotient;

					double _uw = ((i0[ijkl(i + 1, j, k, l + 1)] + i0[ijkl(i, j, k, l + 1)])
						- (i0[ijkl(i + 1, j, k, l - 1)] + i0[ijkl(i, j, k, l - 1)])) / quotient;

					//thresholded image contribution
					double _ux_thr = (i0_thr[ijkl(i + 1, j, k, l)] - i0_thr[ijkl(i, j, k, l)]) / h;

					double _uy_thr = ((i0_thr[ijkl(i + 1, j + 1, k, l)] + i0_thr[ijkl(i, j + 1, k, l)])
						- (i0_thr[ijkl(i + 1, j - 1, k, l)] + i0_thr[ijkl(i, j - 1, k, l)])) / quotient;

					double _uz_thr = ((i0_thr[ijkl(i + 1, j, k + 1, l)] + i0_thr[ijkl(i, j, k + 1, l)])
						- (i0_thr[ijkl(i + 1, j, k - 1, l)] + i0_thr[ijkl(i, j, k - 1, l)])) / quotient;

					double _uw_thr = ((i0_thr[ijkl(i + 1, j, k, l + 1)] + i0_thr[ijkl(i, j, k, l + 1)])
						- (i0_thr[ijkl(i + 1, j, k, l - 1)] + i0_thr[ijkl(i, j, k, l - 1)])) / quotient;

					g_e[ijkl(i, j, k, l)] = _gFunction_mpi_(theta_img * ((_ux * _ux) + (_uy * _uy) + (_uz * _uz) + (_uw * _uw)) +
						rho_thr * ((_ux_thr * _ux_thr) + (_uy_thr * _uy_thr) + (_uz_thr * _uz_thr) + (_uw_thr * _uw_thr)), K);

					// Calculation of coefficients in west direction
					//input image contribution  
					_ux = (i0[ijkl(i - 1, j, k, l)] - i0[ijkl(i, j, k, l)]) / h;

					_uy = ((i0[ijkl(i, j + 1, k, l)] + i0[ijkl(i - 1, j + 1, k, l)])
						- (i0[ijkl(i, j - 1, k, l)] + i0[ijkl(i - 1, j - 1, k, l)])) / quotient;

					_uz = ((i0[ijkl(i, j, k + 1, l)] + i0[ijkl(i - 1, j, k + 1, l)])
						- (i0[ijkl(i, j, k - 1, l)] + i0[ijkl(i - 1, j, k - 1, l)])) / quotient;

					_uw = ((i0[ijkl(i, j, k, l + 1)] + i0[ijkl(i - 1, j, k, l + 1)])
						- (i0[ijkl(i, j, k, l - 1)] + i0[ijkl(i - 1, j, k, l - 1)])) / quotient;

					//thresholded image contribution  
					_ux_thr = (i0_thr[ijkl(i - 1, j, k, l)] - i0_thr[ijkl(i, j, k, l)]) / h;

					_uy_thr = ((i0_thr[ijkl(i, j + 1, k, l)] + i0_thr[ijkl(i - 1, j + 1, k, l)])
						- (i0_thr[ijkl(i, j - 1, k, l)] + i0_thr[ijkl(i - 1, j - 1, k, l)])) / quotient;

					_uz_thr = ((i0_thr[ijkl(i, j, k + 1, l)] + i0_thr[ijkl(i - 1, j, k + 1, l)])
						- (i0_thr[ijkl(i, j, k - 1, l)] + i0_thr[ijkl(i - 1, j, k - 1, l)])) / quotient;

					_uw_thr = ((i0_thr[ijkl(i, j, k, l + 1)] + i0_thr[ijkl(i - 1, j, k, l + 1)])
						- (i0_thr[ijkl(i, j, k, l - 1)] + i0_thr[ijkl(i - 1, j, k, l - 1)])) / quotient;

					g_w[ijkl(i, j, k, l)] = _gFunction_mpi_(theta_img * ((_ux * _ux) + (_uy * _uy) + (_uz * _uz) + (_uw * _uw)) +
						rho_thr * ((_ux_thr * _ux_thr) + (_uy_thr * _uy_thr) + (_uz_thr * _uz_thr) + (_uw_thr * _uw_thr)), K);

					// Calculation of coefficients in north direction 
					//input image contribution 
					_ux = ((i0[ijkl(i + 1, j + 1, k, l)] + i0[ijkl(i + 1, j, k, l)])
						- (i0[ijkl(i - 1, j + 1, k, l)] + i0[ijkl(i - 1, j, k, l)])) / quotient;

					_uy = (i0[ijkl(i, j + 1, k, l)] - i0[ijkl(i, j, k, l)]) / h;

					_uz = ((i0[ijkl(i, j + 1, k + 1, l)] + i0[ijkl(i, j, k + 1, l)])
						- (i0[ijkl(i, j + 1, k - 1, l)] + i0[ijkl(i, j, k - 1, l)])) / quotient;

					_uw = ((i0[ijkl(i, j + 1, k, l + 1)] + i0[ijkl(i, j, k, l + 1)])
						- (i0[ijkl(i, j + 1, k, l - 1)] + i0[ijkl(i, j, k, l - 1)])) / quotient;

					//thresholded image contribution 
					_ux_thr = ((i0_thr[ijkl(i + 1, j + 1, k, l)] + i0_thr[ijkl(i + 1, j, k, l)])
						- (i0_thr[ijkl(i - 1, j + 1, k, l)] + i0_thr[ijkl(i - 1, j, k, l)])) / quotient;

					_uy_thr = (i0_thr[ijkl(i, j + 1, k, l)] - i0_thr[ijkl(i, j, k, l)]) / h;

					_uz_thr = ((i0_thr[ijkl(i, j + 1, k + 1, l)] + i0_thr[ijkl(i, j, k + 1, l)])
						- (i0_thr[ijkl(i, j + 1, k - 1, l)] + i0_thr[ijkl(i, j, k - 1, l)])) / quotient;

					_uw_thr = ((i0_thr[ijkl(i, j + 1, k, l + 1)] + i0_thr[ijkl(i, j, k, l + 1)])
						- (i0_thr[ijkl(i, j + 1, k, l - 1)] + i0_thr[ijkl(i, j, k, l - 1)])) / quotient;

					g_n[ijkl(i, j, k, l)] = _gFunction_mpi_(theta_img * ((_ux * _ux) + (_uy * _uy) + (_uz * _uz) + (_uw * _uw)) +
						rho_thr * ((_ux_thr * _ux_thr) + (_uy_thr * _uy_thr) + (_uz_thr * _uz_thr) + (_uw_thr * _uw_thr)), K);

					// Calculation of coefficients in south direction
					//input image contribution  
					_ux = ((i0[ijkl(i + 1, j - 1, k, l)] + i0[ijkl(i + 1, j, k, l)])
						- (i0[ijkl(i - 1, j - 1, k, l)] + i0[ijkl(i - 1, j, k, l)])) / quotient;

					_uy = (i0[ijkl(i, j - 1, k, l)] - i0[ijkl(i, j, k, l)]) / h;

					_uz = ((i0[ijkl(i, j - 1, k + 1, l)] + i0[ijkl(i, j, k + 1, l)])
						- (i0[ijkl(i, j - 1, k - 1, l)] + i0[ijkl(i, j, k - 1, l)])) / quotient;

					_uw = ((i0[ijkl(i, j - 1, k, l + 1)] + i0[ijkl(i, j, k, l + 1)])
						- (i0[ijkl(i, j - 1, k, l - 1)] + i0[ijkl(i, j, k, l - 1)])) / quotient;

					//thresholded image contribution  
					_ux_thr = ((i0_thr[ijkl(i + 1, j - 1, k, l)] + i0_thr[ijkl(i + 1, j, k, l)])
						- (i0_thr[ijkl(i - 1, j - 1, k, l)] + i0_thr[ijkl(i - 1, j, k, l)])) / quotient;

					_uy_thr = (i0_thr[ijkl(i, j - 1, k, l)] - i0_thr[ijkl(i, j, k, l)]) / h;

					_uz_thr = ((i0_thr[ijkl(i, j - 1, k + 1, l)] + i0_thr[ijkl(i, j, k + 1, l)])
						- (i0_thr[ijkl(i, j - 1, k - 1, l)] + i0_thr[ijkl(i, j, k - 1, l)])) / quotient;

					_uw_thr = ((i0_thr[ijkl(i, j - 1, k, l + 1)] + i0_thr[ijkl(i, j, k, l + 1)])
						- (i0_thr[ijkl(i, j - 1, k, l - 1)] + i0_thr[ijkl(i, j, k, l - 1)])) / quotient;

					g_s[ijkl(i, j, k, l)] = _gFunction_mpi_(theta_img * ((_ux * _ux) + (_uy * _uy) + (_uz * _uz) + (_uw * _uw)) +
						rho_thr * ((_ux_thr * _ux_thr) + (_uy_thr * _uy_thr) + (_uz_thr * _uz_thr) + (_uw_thr * _uw_thr)), K);

					// Calculation of coefficients in top direction  
					//input image contribution
					_ux = ((i0[ijkl(i + 1, j, k + 1, l)] + i0[ijkl(i + 1, j, k, l)])
						- (i0[ijkl(i - 1, j, k + 1, l)] + i0[ijkl(i - 1, j, k, l)])) / quotient;

					_uy = ((i0[ijkl(i, j + 1, k + 1, l)] + i0[ijkl(i, j + 1, k, l)])
						- (i0[ijkl(i, j - 1, k + 1, l)] + i0[ijkl(i, j - 1, k, l)])) / quotient;

					_uz = (i0[ijkl(i, j, k + 1, l)] - i0[ijkl(i, j, k, l)]) / h;

					_uw = ((i0[ijkl(i, j, k + 1, l + 1)] + i0[ijkl(i, j, k, l + 1)])
						- (i0[ijkl(i, j, k + 1, l - 1)] + i0[ijkl(i, j, k, l - 1)])) / quotient;

					//thresholded image contribution
					_ux_thr = ((i0_thr[ijkl(i + 1, j, k + 1, l)] + i0_thr[ijkl(i + 1, j, k, l)])
						- (i0_thr[ijkl(i - 1, j, k + 1, l)] + i0_thr[ijkl(i - 1, j, k, l)])) / quotient;

					_uy_thr = ((i0_thr[ijkl(i, j + 1, k + 1, l)] + i0_thr[ijkl(i, j + 1, k, l)])
						- (i0_thr[ijkl(i, j - 1, k + 1, l)] + i0_thr[ijkl(i, j - 1, k, l)])) / quotient;

					_uz_thr = (i0_thr[ijkl(i, j, k + 1, l)] - i0_thr[ijkl(i, j, k, l)]) / h;

					_uw_thr = ((i0_thr[ijkl(i, j, k + 1, l + 1)] + i0_thr[ijkl(i, j, k, l + 1)])
						- (i0_thr[ijkl(i, j, k + 1, l - 1)] + i0_thr[ijkl(i, j, k, l - 1)])) / quotient;

					g_t[ijkl(i, j, k, l)] = _gFunction_mpi_(theta_img * ((_ux * _ux) + (_uy * _uy) + (_uz * _uz) + (_uw * _uw)) +
						rho_thr * ((_ux_thr * _ux_thr) + (_uy_thr * _uy_thr) + (_uz_thr * _uz_thr) + (_uw_thr * _uw_thr)), K);

					// Calculation of coefficients in bottom direction  
					//input image contribution
					_ux = ((i0[ijkl(i + 1, j, k - 1, l)] + i0[ijkl(i + 1, j, k, l)])
						- (i0[ijkl(i - 1, j, k - 1, l)] + i0[ijkl(i - 1, j, k, l)])) / quotient;

					_uy = ((i0[ijkl(i, j + 1, k - 1, l)] + i0[ijkl(i, j + 1, k, l)])
						- (i0[ijkl(i, j - 1, k - 1, l)] + i0[ijkl(i, j - 1, k, l)])) / quotient;

					_uz = (i0[ijkl(i, j, k - 1, l)] - i0[ijkl(i, j, k, l)]) / h;

					_uw = ((i0[ijkl(i, j, k - 1, l + 1)] + i0[ijkl(i, j, k, l + 1)])
						- (i0[ijkl(i, j, k - 1, l - 1)] + i0[ijkl(i, j, k, l - 1)])) / quotient;

					//thresholded image contribution
					_ux_thr = ((i0_thr[ijkl(i + 1, j, k - 1, l)] + i0_thr[ijkl(i + 1, j, k, l)])
						- (i0_thr[ijkl(i - 1, j, k - 1, l)] + i0_thr[ijkl(i - 1, j, k, l)])) / quotient;

					_uy_thr = ((i0_thr[ijkl(i, j + 1, k - 1, l)] + i0_thr[ijkl(i, j + 1, k, l)])
						- (i0_thr[ijkl(i, j - 1, k - 1, l)] + i0_thr[ijkl(i, j - 1, k, l)])) / quotient;

					_uz_thr = (i0_thr[ijkl(i, j, k - 1, l)] - i0_thr[ijkl(i, j, k, l)]) / h;

					_uw_thr = ((i0_thr[ijkl(i, j, k - 1, l + 1)] + i0_thr[ijkl(i, j, k, l + 1)])
						- (i0_thr[ijkl(i, j, k - 1, l - 1)] + i0_thr[ijkl(i, j, k, l - 1)])) / quotient;

					g_b[ijkl(i, j, k, l)] = _gFunction_mpi_(theta_img * ((_ux * _ux) + (_uy * _uy) + (_uz * _uz) + (_uw * _uw)) +
						rho_thr * ((_ux_thr * _ux_thr) + (_uy_thr * _uy_thr) + (_uz_thr * _uz_thr) + (_uw_thr * _uw_thr)), K);

					// Calculation of coefficients in forward direction 
					//input image contribution
					_ux = ((i0[ijkl(i + 1, j, k, l + 1)] + i0[ijkl(i + 1, j, k, l)])
						- (i0[ijkl(i - 1, j, k, l + 1)] + i0[ijkl(i - 1, j, k, l)])) / quotient;

					_uy = ((i0[ijkl(i, j + 1, k, l + 1)] + i0[ijkl(i, j + 1, k, l)])
						- (i0[ijkl(i, j - 1, k, l + 1)] + i0[ijkl(i, j - 1, k, l)])) / quotient;

					_uz = ((i0[ijkl(i, j, k + 1, l + 1)] + i0[ijkl(i, j, k + 1, l)])
						- (i0[ijkl(i, j, k - 1, l + 1)] + i0[ijkl(i, j, k - 1, l)])) / quotient;

					_uw = (i0[ijkl(i, j, k, l + 1)] - i0[ijkl(i, j, k, l)]) / h;

					//thresholded image contribution
					_ux_thr = ((i0_thr[ijkl(i + 1, j, k, l + 1)] + i0_thr[ijkl(i + 1, j, k, l)])
						- (i0_thr[ijkl(i - 1, j, k, l + 1)] + i0_thr[ijkl(i - 1, j, k, l)])) / quotient;

					_uy_thr = ((i0_thr[ijkl(i, j + 1, k, l + 1)] + i0_thr[ijkl(i, j + 1, k, l)])
						- (i0_thr[ijkl(i, j - 1, k, l + 1)] + i0_thr[ijkl(i, j - 1, k, l)])) / quotient;

					_uz_thr = ((i0_thr[ijkl(i, j, k + 1, l + 1)] + i0_thr[ijkl(i, j, k + 1, l)])
						- (i0_thr[ijkl(i, j, k - 1, l + 1)] + i0_thr[ijkl(i, j, k - 1, l)])) / quotient;

					_uw_thr = (i0_thr[ijkl(i, j, k, l + 1)] - i0_thr[ijkl(i, j, k, l)]) / h;

					g_fw[ijkl(i, j, k, l)] = _gFunction_mpi_(theta_img * ((_ux * _ux) + (_uy * _uy) + (_uz * _uz) + (_uw * _uw)) +
						rho_thr * ((_ux_thr * _ux_thr) + (_uy_thr * _uy_thr) + (_uz_thr * _uz_thr) + (_uw_thr * _uw_thr)), K);


					// Calculation of coefficients in backward direction
					//input image contribution
					_ux = ((i0[ijkl(i + 1, j, k, l - 1)] + i0[ijkl(i + 1, j, k, l)])
						- (i0[ijkl(i - 1, j, k, l - 1)] + i0[ijkl(i - 1, j, k, l)])) / quotient;

					_uy = ((i0[ijkl(i, j + 1, k, l - 1)] + i0[ijkl(i, j + 1, k, l)])
						- (i0[ijkl(i, j - 1, k, l - 1)] + i0[ijkl(i, j - 1, k, l)])) / quotient;

					_uz = ((i0[ijkl(i, j, k + 1, l - 1)] + i0[ijkl(i, j, k + 1, l)])
						- (i0[ijkl(i, j, k - 1, l - 1)] + i0[ijkl(i, j, k - 1, l)])) / quotient;

					_uw = (i0[ijkl(i, j, k, l - 1)] - i0[ijkl(i, j, k, l)]) / h;

					//thresholded image contribution
					_ux_thr = ((i0_thr[ijkl(i + 1, j, k, l - 1)] + i0_thr[ijkl(i + 1, j, k, l)])
						- (i0_thr[ijkl(i - 1, j, k, l - 1)] + i0_thr[ijkl(i - 1, j, k, l)])) / quotient;

					_uy_thr = ((i0_thr[ijkl(i, j + 1, k, l - 1)] + i0_thr[ijkl(i, j + 1, k, l)])
						- (i0_thr[ijkl(i, j - 1, k, l - 1)] + i0_thr[ijkl(i, j - 1, k, l)])) / quotient;

					_uz_thr = ((i0_thr[ijkl(i, j, k + 1, l - 1)] + i0_thr[ijkl(i, j, k + 1, l)])
						- (i0_thr[ijkl(i, j, k - 1, l - 1)] + i0_thr[ijkl(i, j, k - 1, l)])) / quotient;

					_uw_thr = (i0_thr[ijkl(i, j, k, l - 1)] - i0_thr[ijkl(i, j, k, l)]) / h;

					g_bw[ijkl(i, j, k, l)] = _gFunction_mpi_(theta_img * ((_ux * _ux) + (_uy * _uy) + (_uz * _uz) + (_uw * _uw)) +
						rho_thr * ((_ux_thr * _ux_thr) + (_uy_thr * _uy_thr) + (_uz_thr * _uz_thr) + (_uw_thr * _uw_thr)), K);
				}
	return true;
}

bool _reflection4Ddata_mpi_(double *u)
{
	int k, i, j, l;

	// X Direction
	for (l = 0; l <= procs_lMax + 1; l++)
		for (k = 0; k <= kMax - 1; k++)
			for (j = 0; j <= jMax - 1; j++)
			{
				u[ijkl(0, j, k, l)] = u[ijkl(1, j, k, l)];
				u[ijkl(iMax - 1, j, k, l)] = u[ijkl(iMax - 2, j, k, l)];
			}
	// Y Direction
	for (l = 0; l <= procs_lMax + 1; l++)
		for (k = 0; k < kMax; k++)
			for (i = 0; i < iMax; i++)
			{
				u[ijkl(i, 0, k, l)] = u[ijkl(i, 1, k, l)];
				u[ijkl(i, jMax - 1, k, l)] = u[ijkl(i, jMax - 2, k, l)];
			}
	// Z Direction
	for (l = 0; l <= procs_lMax + 1; l++)
		for (j = 0; j < jMax; j++)
			for (i = 0; i < iMax; i++)
			{
				u[ijkl(i, j, 0, l)] = u[ijkl(i, j, 1, l)];
				u[ijkl(i, j, kMax - 1, l)] = u[ijkl(i, j, kMax - 2, l)];
			}
	// W Direction
	if (process_ID == 0)
	{
		for (k = 0; k < kMax; k++)
			for (j = 0; j < jMax; j++)
				for (i = 0; i < iMax; i++)
					u[ijkl(i, j, k, 0)] = u[ijkl(i, j, k, 1)];
	}

	if (process_ID == no_of_processes - 1)
	{
		for (k = 0; k < kMax; k++)
			for (j = 0; j < jMax; j++)
				for (i = 0; i < iMax; i++)
					u[ijkl(i, j, k, procs_lMax + 1)] = u[ijkl(i, j, k, procs_lMax)];
	}
	return true;
}

void _copy4DdataFromSrcToDest_mpi_(double *srcDataPtr, double *destDataPtr)
{
	int i, j, k, l;
	for (l = 0; l <= procs_lMax + 1; l++)
		for (k = 0; k <= kMax - 1; k++)
			for (j = 0; j <= jMax - 1; j++)
				for (i = 0; i <= iMax - 1; i++)
					destDataPtr[ijkl(i, j, k, l)] = srcDataPtr[ijkl(i, j, k, l)];
}


double _get4DMax_mpi_(double *imagePtr)
{
	//check if the memory was allocated successfully
	if (imagePtr == NULL)
		exit(1);

	int i, j, k, l;
	double localMax = 0, globalMax;

	//Determine minimum and maximum value
	for (l = 1; l <= procs_lMax; l++)
		for (k = 1; k <= kMax - 2; k++)
			for (j = 1; j <= jMax - 2; j++)
				for (i = 1; i <= iMax - 2; i++)
				{
					if (imagePtr[ijkl(i, j, k, l)] > localMax)
						localMax = imagePtr[ijkl(i, j, k, l)];
				}
	MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	return globalMax;
}

double _get3DMinWithinR_mpi_(double *imagePtr, int l, double center_x, double center_y, double center_z, double R)
{
	//check if the memory was allocated successfully
	if (imagePtr == NULL)
		exit(1);

	int i, j, k;
	double dz, dx, dy, norm, min = 100000;
	int k_init, k_final, j_init, j_final, i_init, i_final;

	k_init 	= (int)(center_z - R - 0.5);
	k_final = (int)(center_z + R + 0.5);
	j_init 	= (int)(center_y - R - 0.5);
	j_final = (int)(center_y + R + 0.5);
	i_init 	= (int)(center_x - R - 0.5);
	i_final = (int)(center_x + R + 0.5);

	//Determine minimum and maximum value
	for (k = k_init; k <= k_final; k++)
		for (j = j_init; j <= j_final; j++)
			for (i = i_init; i <= i_final; i++)
			{
				dx = i - center_x;
				dy = j - center_y;
				dz = k - center_z;

				// computes norm of each point and the center
				norm = sqrt((dx * dx) + (dy * dy) + (dz * dz));
				if (norm <= R)
				{
					if (imagePtr[ijkl(i, j, k, l)] < min)
						min = imagePtr[ijkl(i, j, k, l)];
				}
			}
	return min;
}

double _get3DMaxWithinR_mpi_(double *imagePtr, int l, double center_x, double center_y, double center_z, double R)
{
	//check if the memory was allocated successfully
	if (imagePtr == NULL)
		exit(1);

	int i, j, k;
	double dz, dx, dy, norm, max = 0;
	int k_init, k_final, j_init, j_final, i_init, i_final;

	k_init 	= (int)(center_z - R - 0.5);
	k_final = (int)(center_z + R + 0.5);
	j_init 	= (int)(center_y - R - 0.5);
	j_final = (int)(center_y + R + 0.5);
	i_init 	= (int)(center_x - R - 0.5);
	i_final = (int)(center_x + R + 0.5);

	//Determine minimum and maximum value
	for (k = k_init; k <= k_final; k++)
		for (j = j_init; j <= j_final; j++)
			for (i = i_init; i <= i_final; i++)
			{
				dx = i - center_x;
				dy = j - center_y;
				dz = k - center_z;

				// computes norm of each point and the center
				norm = sqrt((dx * dx) + (dy * dy) + (dz * dz));
				if (norm <= R)
				{
					if (imagePtr[ijkl(i, j, k, l)] > max)
						max = imagePtr[ijkl(i, j, k, l)];
				}
			}
	return max;
}

bool _initialize4dArray_mpi_(double *array4DPtr, double value)
{
	int l, k, i, j;

	//checks if the memory was allocated
	if (array4DPtr == NULL)
		return false;

	// filling the 4d array with number=value
	for (l = 0; l <= procs_lMax + 1; l++)
		for (k = 0; k <= kMax - 1; k++)
			for (j = 0; j <= jMax - 1; j++)
				for (i = 0; i <= iMax - 1; i++)
					array4DPtr[ijkl(i, j, k, l)] = value;
	return true;
}

bool _locallyThreshold4Ddata_mpi_(double *inputImagePtr, double *thresholdedImagePtr)
{
	//check if the memory was allocated successfully
	if (inputImagePtr == NULL || thresholdedImagePtr == NULL)
		return false;

	int i, j, k, l, c;
	double tresholdValue, globalMax;
	double max, min, norm, dx, dy, dz;
	int k_init, k_final, j_init, j_final, i_init, i_final;

	globalMax = _get4DMax_mpi_(inputImagePtr);
	 
	for (l = 1; l <= procs_lMax; l++)
		for (c = 0; c < no_of_centers[l]; c++)
		{
			max = _get3DMaxWithinR_mpi_(inputImagePtr, l, center_x[l][c], center_y[l][c], center_z[l][c], ballRadius);
			min = _get3DMinWithinR_mpi_(inputImagePtr, l, center_x[l][c], center_y[l][c], center_z[l][c], ballRadius);
			
			tresholdValue = alpha * min + beta * max;// (3 * min + max) / 4;
			   
			k_init 	= (int)(center_z[l][c] - ballRadius - 0.5);
			k_final = (int)(center_z[l][c] + ballRadius + 0.5);
			j_init 	= (int)(center_y[l][c] - ballRadius - 0.5);
			j_final = (int)(center_y[l][c] + ballRadius + 0.5);
			i_init 	= (int)(center_x[l][c] - ballRadius - 0.5);
			i_final = (int)(center_x[l][c] + ballRadius + 0.5);


			//Perform local thresholding
			for (k = k_init; k <= k_final; k++)
				for (j = j_init; j <= j_final; j++)
					for (i = i_init; i <= i_final; i++)
					{
						dx = i - center_x[l][c];
						dy = j - center_y[l][c];
						dz = k - center_z[l][c];

						// computes norm of each point and the center
						norm = sqrt((dx * dx) + (dy * dy) + (dz * dz));
						if (norm <= ballRadius)
						{
							if (inputImagePtr[ijkl(i, j, k, l)] > tresholdValue)
								thresholdedImagePtr[ijkl(i, j, k, l)] = globalMax;
						}
					}
		}
	
	//Exchange volumes between processors
	_communication_mpi_(thresholdedImagePtr);

	return true;
}

bool _rescale4DdataToZeroOne_mpi_(double *imagePtr)
{
	//check if the memory was allocated successfully
	if (imagePtr == NULL)
		return false;

	int i, j, k, l, c;
	double max, min, quotient, offset, dz, dy, dx, norm_of_distance, new_value;
	int k_init, k_final, j_init, j_final, i_init, i_final;
	 
	for (l = 1; l <= procs_lMax; l++)
		for (c = 0; c < no_of_centers[l]; c++)
		{
			k_init 	= (int)(center_z[l][c] - ballRadius - 0.5);
			k_final = (int)(center_z[l][c] + ballRadius + 0.5);
			j_init 	= (int)(center_y[l][c] - ballRadius - 0.5);
			j_final = (int)(center_y[l][c] + ballRadius + 0.5);
			i_init 	= (int)(center_x[l][c] - ballRadius - 0.5);
			i_final = (int)(center_x[l][c] + ballRadius + 0.5);

			max = 0, min = 100000;

			//Determine minimum and maximum value
			for (k = k_init; k <= k_final; k++)
				for (j = j_init; j <= j_final; j++)
					for (i = i_init; i <= i_final; i++)
					{
						dx = i - center_x[l][c];
						dy = j - center_y[l][c];
						dz = k - center_z[l][c];

						//Find local minimum and maximum
						norm_of_distance = sqrt((dx * dx) + (dy * dy) + (dz * dz));
						if (norm_of_distance <= ballRadius)
						{
							if (imagePtr[ijkl(i, j, k, l)] < min)
								min = imagePtr[ijkl(i, j, k, l)];
							if (imagePtr[ijkl(i, j, k, l)] > max)
								max = imagePtr[ijkl(i, j, k, l)];
						}
					}

			quotient = 1. / (max - min);
			offset = min * quotient;
			//Rescale values to interval (0, 1)
			for (k = k_init; k <= k_final; k++)
				for (j = j_init; j <= j_final; j++)
					for (i = i_init; i <= i_final; i++)
					{
						dx = i - center_x[l][c];
						dy = j - center_y[l][c];
						dz = k - center_z[l][c];

						// recaling values
						norm_of_distance = sqrt((dx * dx) + (dy * dy) + (dz * dz));
						new_value = (quotient * imagePtr[ijkl(i, j, k, l)]) - offset;

						if (c == 0)
						{
							if (norm_of_distance <= ballRadius)
								imagePtr[ijkl(i, j, k, l)] = new_value;
						}
						else
						{
							if (norm_of_distance <= ballRadius)
							{
								if (imagePtr[ijkl(i, j, k, l)] < new_value)
									imagePtr[ijkl(i, j, k, l)] = new_value;
							}
						}
					}

		}
	 
	for (l = 1; l <= procs_lMax; l++)
		for (k = 1; k <= kMax - 2; k++)
			for (j = 1; j <= jMax - 2; j++)
				for (i = 1; i <= iMax - 2; i++)
				{
					if (Mmask[ijkl(i, j, k, l)] == 0)
						imagePtr[ijkl(i, j, k, l)] = 0;
				}
	
	//Exchange volumes between processors
	_communication_mpi_(imagePtr);

	return true;
}

bool _generate4DsegmFunct_mpi_(double *segmFuntionPtr)
{
	if (segmFuntionPtr == NULL)
		return false;

	int i, j, k, l, c;
	double v = 0.5, R = 12.;
	double dx, dy, dz, norm_of_distance, new_value;
	int k_init, k_final, j_init, j_final, i_init, i_final;
	 
	for (l = 1; l <= procs_lMax; l++)
		for (c = 0; c < no_of_centers[l]; c++)
		{
			k_init 	= (int)(center_z[l][c] - ballRadius - 0.5);
			k_final = (int)(center_z[l][c] + ballRadius + 0.5);
			j_init 	= (int)(center_y[l][c] - ballRadius - 0.5);
			j_final = (int)(center_y[l][c] + ballRadius + 0.5);
			i_init 	= (int)(center_x[l][c] - ballRadius - 0.5);
			i_final = (int)(center_x[l][c] + ballRadius + 0.5);

			for (k = k_init; k <= k_final; k++)
				for (j = j_init; j <= j_final; j++)
					for (i = i_init; i <= i_final; i++)
					{
						dx = i - center_x[l][c];
						dy = j - center_y[l][c];
						dz = k - center_z[l][c];

						// Set Value
						norm_of_distance = sqrt((dx * dx) + (dy * dy) + (dz * dz));

						if (norm_of_distance <= R)
						{
							new_value = fabs((1. / (sqrt((dx * dx) + (dy * dy) + (dz * dz)) + v)) - (1. / (R + v)));
							if (segmFuntionPtr[ijkl(i, j, k, l)] < new_value)
								segmFuntionPtr[ijkl(i, j, k, l)] = new_value;

							Mmask[ijkl(i, j, k, l)] = 1.;
						}
					}
		}

	//Exchange volumes between processors
	_communication_mpi_(segmFuntionPtr);

	return true;
}
void _readData_mpi_(double *inputImage, unsigned char *fileDirectory)
{
	int i, l, prod_max_id;
	char temp[200], dim[200];//arrays for reading of header information
	int Points;
	double x, y, z;

	//location for volume just to extract information about spacing
	unsigned char inputFile[350], inputCenter[350];
	unsigned char inputFile_ending[350], inputCenter_ending[350];

	if (process_ID == no_of_processes - 1)
		prod_max_id = (lMax - 2) - procs_lMax;
	else
		prod_max_id = procs_lMax * process_ID;

	for (l = 1; l <= procs_lMax; l++)
	{
		/*Loading/reading 3D image data*/
		//location for 3D image data
		strcpy(inputFile, fileDirectory);
		sprintf(inputFile_ending, "/180420hZ_t%03d_ch00.vtk", prod_max_id + l);//180420hZ_t%03d_ch00 or p011_frame%d or p011_frame_at_time_%03d
		strcat(inputFile, inputFile_ending);

		input_file = fopen(inputFile, "r");
		//reading data information automatically
		if (input_file == NULL)
			exit(1);
		else
			_load3dDataArrayVTK_mpi_(inputImage, l, inputFile);
		// Close file after reading
		fclose(input_file);

		/*Reading 3D image centers*/
		//location for centers
		strcpy(inputCenter, fileDirectory);
		sprintf(inputCenter_ending, "/center_file_%03d.vtk", prod_max_id + l);//(for Hanh: center_file_%03d) (for Antonia: center_file%d)
		strcat(inputCenter, inputCenter_ending);

		input_center_file = fopen(inputCenter, "r");
		//reading center coordinates information automatically
		if (input_center_file == NULL)
		{
			exit(1);
		}
		else
		{
			//Read information from center file
			_readMetaData_mpi_(input_center_file);
			fgets(temp, 8, input_center_file); // "Points "
			_readNumberFromFile_mpi_(dim, input_center_file);

			Points = atoi(dim);  // Sets the number of points

			while (getc(input_center_file) != '\n');  /* skip to end of POINTS line */

			for (i = 0; i < Points; i++)
			{
				fscanf(input_center_file, "%lf", &x);
				center_x[l][i] = (double)(int)(x / spacing_x + 0.5);  // Sets the center_x

				fscanf(input_center_file, "%lf", &y);
				center_y[l][i] = (double)(int)(y / spacing_y + 0.5);  // Sets the center_y

				fscanf(input_center_file, "%lf", &z);
				center_z[l][i] = (double)(int)(z / spacing_z + 0.5);  // Sets the center_z
			}
		}
		// Close file after reading
		fclose(input_center_file);
	}
	//Exchange volumes between processors
	_communication_mpi_(inputImage);
}

int _getNumberOfCenters_mpi_(unsigned char *fileDirectory, int l)
{
	int prod_max_id;
	char temp[200], dim[200];//arrays for reading of header information
	int Points;
	
	//location for volume just to extract information about spacing
	unsigned char inputCenter[350], inputCenter_ending[350];
	
	if (process_ID == no_of_processes - 1)
		prod_max_id = (lMax - 2) - procs_lMax;
	else
		prod_max_id = procs_lMax * process_ID;

	/*Reading 3D image centers*/
	//location for centers
	strcpy(inputCenter, fileDirectory);
	sprintf(inputCenter_ending, "/center_file_%03d.vtk", prod_max_id + l);// (for Antonia: center_file%d)
	strcat(inputCenter, inputCenter_ending);

	input_center_file = fopen(inputCenter, "r");
	//reading center coordinates information automatically
	if (input_center_file == NULL)
	{
		exit(1);
	}
	else
	{
		//Read information from center file
		_readMetaData_mpi_(input_center_file);
		fgets(temp, 8, input_center_file); // "Points "
		_readNumberFromFile_mpi_(dim, input_center_file);

		Points = atoi(dim);  // Sets the number of points
	}
	// Close file after reading
	fclose(input_center_file);

	return Points;
}

bool _writeData_mpi_(double *array4DPtr, unsigned char * pathPtr)
{
	int i, j, k, l, prod_max_id;
	FILE * outputfile; //file stream
	//Array for name construction
	unsigned char name[350];
	unsigned char name_ending[350];

	if (process_ID == no_of_processes - 1)
		prod_max_id = (lMax - 2) - procs_lMax;
	else
		prod_max_id = procs_lMax * process_ID;

	for (l = 1; l <= procs_lMax; l++)
	{
		//store 3D object
		strcpy(name, pathPtr);
		sprintf(name_ending, "_at_time_%03d.vtk", prod_max_id + l);
		strcat(name, name_ending);

		//checks if the file was sucessfully opened
		outputfile = fopen(name, "w");
		if (outputfile == NULL) {
			puts("File could not be opened.");
		}
		else
		{
			fprintf(outputfile, "# vtk DataFile Version 3.0\n");
			fprintf(outputfile, "file in binary format\n");
			fprintf(outputfile, "BINARY\n");
			fprintf(outputfile, "DATASET STRUCTURED_POINTS\n");
			fprintf(outputfile, "DIMENSIONS %d %d %d\n", iMax - 2, jMax - 2, kMax - 2);
			fprintf(outputfile, "ORIGIN %f %f %f\n", origin_x, origin_y, origin_z);
			fprintf(outputfile, "SPACING %f %f %f\n", spacing_x, spacing_y, spacing_z);
			fprintf(outputfile, "POINT_DATA %d\n", (iMax - 2) * (jMax - 2) * (kMax - 2));
			fprintf(outputfile, "SCALARS scalars unsigned_char\n");
			fprintf(outputfile, "LOOKUP_TABLE default\n");
		}

		/*//writing binary data to file
		for (k = 1; k <= kMax - 2; k++)
			for (j = 1; j <= jMax - 2; j++)
				for (i = 1; i <= iMax - 2; i++)
					fputc((unsigned char)(255 * array4DPtr[ijkl(i, j, k, l)] + 0.5), outputfile);*/

		//writing binary data to file
		for (k = 1; k <= kMax - 2; k++)
			for (j = 1; j <= jMax - 2; j++)
				for (i = 1; i <= iMax - 2; i++)
					{
						if (array4DPtr[ijkl(i, j, k, l)] >= 0.5)
							fputc(1, outputfile);
						else
							fputc(0, outputfile);
					}
		fclose(outputfile);
	}
	return true;
}

void _readNumberFromFile_mpi_(char *dim, FILE *input_file)
{
	int count = 0, var;
	do
	{
		//chs[count]
		var = fgetc(input_file);
		//var = chs[count];
		dim[count] = (char)var;
		count++;
	} while (!isspace(var));
}

void _readMetaData_mpi_(FILE *input_file)
{
	while (getc(input_file) == '#')              /* skip comment lines */
	{
		while (getc(input_file) != '\n');          /* skip to end of comment line */
	}

	while (getc(input_file) != '\n');          /* skip to end of vtk output line */

	while (getc(input_file) != '\n');          /* skip to end of ASCII line */

	while (getc(input_file) != '\n');          /* skip to end of DATASET line */
}

bool _load3dDataArrayVTK_mpi_(double *imageDataPtr, int l, unsigned char * pathPtr)
{
	//checks if the memory was allocated
	if (imageDataPtr == NULL || pathPtr == NULL)
		return false;

	int i, j, k, value;
	FILE *file;
	unsigned char temp[256];
	//Reading data from file
	file = fopen(pathPtr, "rb");
	if (file == NULL) {
		return false;
	}
	else {
		//Read header 
		fgets(temp, 256, file);
		fgets(temp, 256, file);
		fgets(temp, 256, file);
		fgets(temp, 256, file);
		fgets(temp, 256, file);
		fgets(temp, 256, file);
		fgets(temp, 256, file);
		fgets(temp, 256, file);
		fgets(temp, 256, file);
		fgets(temp, 256, file);

		//Read other data
		for (k = 1; k <= kMax - 2; k++)
			for (j = 1; j <= jMax - 2; j++)
				for (i = 1; i <= iMax - 2; i++)
					imageDataPtr[ijkl(i, j, k, l)] = getc(file) / 255.;
		}
	fclose(file);
	return true;
}

int main(int argc, char **argv)
{
	double CPUT1, CPUT2, localCPUT, globalCPUT;
	double difference_btw_current_and_previous_sol;
	int l, prod_max_id;
	/* MPI Initialisation. Its important to put this call at the
	   begining of the program, after variable declarations.*/
	MPI_Init(&argc, &argv);

	/* Get the number of MPI tasks (or processes) and the taskid (or process ID) of each of these tasks (or processes).      */
	MPI_Comm_size(MPI_COMM_WORLD, &no_of_processes);
	MPI_Comm_rank(MPI_COMM_WORLD, &process_ID);
	
	//for Hanh's data
	iMax = 569;
	jMax = 579;
	kMax = 149;
	lMax = 72;

	/*//for Antonia's data
	iMax = 514;
	jMax = 514;
	kMax = 110;
	lMax = 21;*/
	
	/*//for artificial data
	iMax = 57;
	jMax = 57;
	kMax = 57;
	lMax = 22;*/

	ballRadius = 12;
	alpha = 0.87, beta = 0.13; //See also alpha = 0.85, beta = 0.15
	theta_img = 0.5, rho_thr = 0.5;

	maxNoGSIteration = 5;
	maxNoOfTimeSteps = 1;
	segTol = 1.0e-12;
	omega_c = 1.8;
	omega_sigma = 1.3;
	gsTol = 1.0e-12;
	h = 0.01;
	tau = 0.01;
	eps2 = 1.;
	K = 1.;
	sigma = (h * h) / 8;
	maxNoGSIteration_sigma = 10;
	
		
	/*//for artificial data
	spacing_x = h;
	spacing_y = h;
	spacing_z = h;*/
	
	//for Hanh's data
	spacing_x = 0.442808;
	spacing_y = 0.442808;
	spacing_z = 1.000000;
	
	/*//for Antonia's data
	spacing_x = 0.287362;
	spacing_y = 0.287362;
	spacing_z = 1.000000;*/

	origin_x = 0.;
	origin_y = 0.;
	origin_z = 0.;

	//file directiory
	unsigned char fileDirectory[] = "/data/users/xuba/frame70";

	procs_lMax = (int)ceil((lMax - 2) / no_of_processes);

	if (process_ID == no_of_processes - 1)
	{
		procs_lMax = (lMax - 2) - (no_of_processes - 1) * procs_lMax;
	}

	if (process_ID == no_of_processes - 1)
		prod_max_id = (lMax - 2) - procs_lMax;
	else
		prod_max_id = procs_lMax * process_ID;

	no_of_centers = (int *)malloc(sizeof(int) * (procs_lMax + 2));
	for (l = 0; l <= procs_lMax + 1; l++)
	{
		if (l == 0 || l == (procs_lMax + 1))
			no_of_centers[l] = 0;
		else
			no_of_centers[l] = _getNumberOfCenters_mpi_(fileDirectory, l);
	}

	_allocateMem_mpi_();

	//Read data from files
	_readData_mpi_(i0, fileDirectory);
		
	//Note the initial time
	localCPUT = MPI_Wtime();
	MPI_Reduce(&localCPUT, &globalCPUT, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
	CPUT1 = globalCPUT;

	// Perform local thresholding of the input image
	_locallyThreshold4Ddata_mpi_(i0, u);
		
	//Compute the PM coefficients
	_gf_inputImage_plus_thresholdedImage_mpi_(i0, u);

	//Generate the initial segmentation function (or the initial condition)
	_generate4DsegmFunct_mpi_(u);
	
	//rescale values of segmentation function globally to interval (0, 1)
	_rescale4DdataToZeroOne_mpi_(u);

	//loop for segmentation time steps
	int tCount = 1;
	do
	{
		numberOfCurrentTimeStep = tCount;

		// Call to function that will evolve segmentation function in each discrete time step
		_segTimeStep_mpi_();

		//Compute the L2 norm of the difference between the current and previous solutions
		difference_btw_current_and_previous_sol = _l2norm_mpi_(prev_u, u);

		if (process_ID == 0)
			printf("l2 norm of difference at step %d is %e\n", tCount, difference_btw_current_and_previous_sol);

		tCount++;
	} while ((tCount <= maxNoOfTimeSteps) && (difference_btw_current_and_previous_sol > segTol));
	
	//Note the final time
	localCPUT = MPI_Wtime();
	MPI_Reduce(&localCPUT, &globalCPUT, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	CPUT2 = globalCPUT;
	
	if (process_ID == 0)
		printf("CPU time is: %e secs\n", CPUT2 - CPUT1);
	
 	_writeData_mpi_(u, "/data/users/xuba/frame70/mpi_70_frame_data_result");
	_deallocateMem_mpi_();
	free(no_of_centers);

	MPI_Finalize();
	return 0;
}