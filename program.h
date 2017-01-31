#pragma once

#include <mpi.h>
#include <string>
#include <stdio.h>
#include <stdlib.h>     /* srand, rand */

#include <time.h>       /* time */
#include <iostream>
#include <fstream>      
#include <vector>

//My library
#include "kmeans.h"
#include "KmeansFilesManager.h"


using namespace std;

void buildResultArray(double* aResultArr, double aMinDistance, double aDeltaTime, Centroid* centroids, int aNumberOfCentroids);
void buildResultObject(KmeansResult* aResult, double* aResultArr, int aResultArraySize, int aNumberOfCentroid);
void clearAll(Point* aPoints, Centroid* aCentroid, double* aInputData, double* aResult, int procId);
MPI_Datatype CreatePointMpiType(void);

MPI_Datatype CreatePointMpiType(void)
{
	int mId;
	double mA;
	double mB;
	double mR;
	double mDistnceToCenter;
	int mCentroidId;
	double mX;
	double mY;
	const int nitems = POINT_STRUCT_ELEMENTS;
	int blocklengths[nitems] = { sizeof(int), sizeof(double), sizeof(double), sizeof(double), sizeof(double), sizeof(int), sizeof(double), sizeof(double) };
	MPI_Datatype types[nitems] = { MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_INT, MPI_DOUBLE, MPI_DOUBLE };
	MPI_Datatype mpi_point_type;
	MPI_Aint     offsets[nitems];
	offsets[0] = offsetof(struct Point, mId);
	offsets[1] = offsetof(struct Point, mA);
	offsets[2] = offsetof(struct Point, mB);
	offsets[3] = offsetof(struct Point, mR);
	offsets[4] = offsetof(struct Point, mDistnceToCenter);
	offsets[5] = offsetof(struct Point, mCentroidId);
	offsets[6] = offsetof(struct Point, mX);
	offsets[7] = offsetof(struct Point, mY);
	MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_point_type);
	MPI_Type_commit(&mpi_point_type);
	return mpi_point_type;
}


