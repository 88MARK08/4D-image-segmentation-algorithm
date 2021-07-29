/* Author: Markjoe Olunna UBA
 * Purpose: ImageInLife project - 4D Image Segmentation Methods
 * Language:  C */
#pragma once
#include <stdbool.h>
#include <stddef.h>

bool _allocateMem_();

bool _deallocateMem_();

double _l2norm_(double *dataArray3DPtr1, double *dataArray3DPtr2);

bool _gs_coefficients_();

bool _segTimeStep_();

bool _presmoothStep_(double *u);

bool _set4DBoundaryToZeroDirichletBC_(double *u);

bool _generate4DsegmFunct_(double *segmFuntionPtr);

bool _rescale4DdataToZeroOne_(double *imagePtr);

bool _locallyThreshold4Ddata_(double *inputImagePtr, double *thresholdedImagePtr);

bool _gf_inputImage_plus_thresholdedImage_(double *i0, double *i0_thr);

void _copy4DdataFromSrcToDest_(double *srcDataPtr, double *destDataPtr);

int ijkl(int i, int j, int k, int l);

double _gFunction_(double value, double coef);

double _get4DMin_(double *imagePtr);

double _get4DMax_(double *imagePtr);

double _get3DMaxWithinR_(double *imagePtr, int l, double center_x, double center_y, double center_z, double R);

double _get3DMinWithinR_(double *imagePtr, int l, double center_x, double center_y, double center_z, double R);

void _readNumberFromFile_(char *dim, FILE *input_file);

void _readMetaData_(FILE *input_file);

bool _load3dDataArrayVTK_(double *imageDataPtr, int l, unsigned char * pathPtr);

bool _writeData_(double *array4DPtr, unsigned char * pathPtr);

void _readData_(double *inputImage, unsigned char *fileDirectory);

bool _initialize4dArray_(double *array4DPtr, double value);

bool _reflection4Ddata_(double *u);

void _printDataToFile_(double *u, unsigned char *centerFilePath);

void _presmoothStepEXP_(double *u);

double _mass4D_(double *dataArray3DPtr1);
