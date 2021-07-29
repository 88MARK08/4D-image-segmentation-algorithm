/* Author: Markjoe Olunna UBA
 * Purpose: ImageInLife project - 4D Image Segmentation Methods
 * Language:  C */
#pragma once
#include <stdbool.h>
#include <stddef.h>

bool _allocateMem_mpi_();

bool _deallocateMem_mpi_();

double _l2norm_mpi_(double *dataArray3DPtr1, double *dataArray3DPtr2);

bool _gs_coefficients_mpi_();

bool _segTimeStep_mpi_();

bool _presmoothStep_mpi_(double *u);

bool _generate4DsegmFunct_mpi_(double *segmFuntionPtr);

bool _rescale4DdataToZeroOne_mpi_(double *imagePtr);

bool _locallyThreshold4Ddata_mpi_(double *inputImagePtr, double *thresholdedImagePtr);

bool _gf_inputImage_plus_thresholdedImage_mpi_(double *i0, double *i0_thr);

void _copy4DdataFromSrcToDest_mpi_(double *srcDataPtr, double *destDataPtr);

int ijkl(int i, int j, int k, int l);

double _gFunction_mpi_(double value, double coef);

double _get4DMin_mpi_(double *imagePtr);

double _get4DMax_mpi_(double *imagePtr);

double _get3DMaxWithinR_mpi_(double *imagePtr, int l, double center_x, double center_y, double center_z, double R);

double _get3DMinWithinR_mpi_(double *imagePtr, int l, double center_x, double center_y, double center_z, double R);

void _readNumberFromFile_mpi_(char *dim, FILE *input_file);

void _readMetaData_mpi_(FILE *input_file);

bool _load3dDataArrayVTK_mpi_(double *imageDataPtr, int l, unsigned char * pathPtr);

bool _writeData_mpi_(double *array4DPtr, unsigned char * pathPtr);

void _readData_mpi_(double *inputImage, unsigned char *fileDirectory);

bool _initialize4dArray_mpi_(double *array4DPtr, double value);

bool _reflection4Ddata_mpi_(double *u);

void _presmoothStepEXP_mpi_(double *u);

void _communication_mpi_(double *u);

double _mass4D_mpi_(double *dataArray3DPtr1);

int _getNumberOfCenters_mpi_(unsigned char *fileDirectory, int l);
