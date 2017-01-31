#pragma once

#include <math.h>
#include <stdio.h>
#include <limits.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//My library
#include "structs.h"

using namespace std;

#define MAX_THREADS_PER_BLOCK 1024

/*
handling kmeans algorithm using Cuda
*/

void cudaInitPoints(Point* aPointsInDevice, int* aNumberOfPointsForCuda, int* aNumberOfPointsInDeivce,
	double* aCosAngle, double* aSinAngle);
void cudaCalcDistanceToCentroid(Point* aPointsInDevice, int* aNumberOfPointsForCuda, int* aNumberOfPointsInDeivce,
	Centroid* aCentroidsInDevice, int* aNumberOfCentroids, bool* aHasChanged);
int  calculateNumOfBlocks(int* aNumOfPoints);
