
#pragma once
#define _CRT_SECURE_NO_WARNINGS

#pragma warning(disable : 4996) //_CRT_SECURE_NO_WARNINGS - due to thrusted library

//open mp
#include <omp.h>
#include <mpi.h>
#include "kmeans_cuda.h"

using namespace std;

const static int MASTER_THREAD = 0;

/*
handling kmeans algorithm using openMP
*/

double StartKmeansCalc(Point* aPoints, int* aNumberOfPoints, int* aNumberOfPointsForCuda, Centroid* aCentroids,
	int* aNumberOfCentroids, double* aCurrentInterval, double* aTotalTime, int* aMaxIteration, int procId);
void initPoints(Point* aPoints, int* aNumberOfPoints, int* aNumberOfPointsForCuda, Centroid* aCentroids,
	int* aNumberOfCentroids, double* aCurrentInterval, double* aTotalTime);
bool classify_points_to_centroid(Point* aPoints, int* aNumberOfPoints, int* aNumberOfPointsForCuda,
	Centroid* aCentroids, int* aNumberOfCentroids);
bool omp_calc_distance_point_to_centroids(Point* aPoints, Centroid* aCentroids, int* aNumberOfCentroids);
void omp_merge_centroids_results(Centroid* aCentroids, Centroid* aCentroidsFromDevice, int* aNumberOfCentroids);
double calcMinDistanceBetweenCentroids(Centroid* aCentroids, int* aNumberOfCentroids);
double calcMinDistanceFromCentroids(Centroid aCentroid, Centroid* aCentroids, int* aNumberOfCentroids);
void omp_update_centroids(Centroid* aCentroids, Point* aPoints, int* aNumberOfCentroids, int* aNumberOfPoints);



//cuda
void prepareCuda(int numberOfElementsForCuda, int aNumberOfCentroids);
void CudaCopyOriginalPointsFromHost(Point* aPoints, int* aNumberOfElementsForCuda, int* aNumberOfCentroids);
void CudaCopyAnglesFromHost(double* aCosAnge, double* aSinAngle);
void CudaCopyCentroidsFromHost(Centroid* aCentroids,int* aNumberOfCentroids);
void CudaGetHasChangedFlag(bool* aAnswer);
void CudaSetHasChangedFlag(bool* aHasChanged);
void CudaGetDeviceCentroids(Centroid* tempHostCentroids, int* numberOfCentroids);
void CudaFreeAll(int procId);
void statusCuda(cudaError_t cudaStatus, char* statusMessage);
