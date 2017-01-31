#include "kmeans.h"

//Cuda
Point* mPointsInDevice = NULL;
Centroid* mCentroidsInDevice = NULL;
int* mNumberOfCentroidsInDevice;
int*  mNumberOfPointsInDevice;
double* mCosAngleInDevice;
double*  mSinAngleInDevice;
bool*  mHasUpdatedInDevice;
Centroid* mCentroidsFromDevice = new Centroid;


double StartKmeansCalc(Point* aPoints, int* aNumberOfPoints, int* aNumberOfPointsForCuda, Centroid* aCentroids,
	int* aNumberOfCentroids, double* aCurrentInterval, double* aTotalTime, int* aMaxIteration, int procId)
{
	int currentIteration = 0;
	bool hasChanged = true;

	//calc X ,Y and init other point values of each point using cuda and omp
	initPoints(aPoints, aNumberOfPoints, aNumberOfPointsForCuda, aCentroids, aNumberOfCentroids,
		aCurrentInterval, aTotalTime);	
	
	
	//find clusters
	while (hasChanged && currentIteration < (*aMaxIteration))
	{
		double startTime = MPI_Wtime();
		hasChanged = false;
		
		CudaCopyCentroidsFromHost(aCentroids, aNumberOfCentroids);
		CudaSetHasChangedFlag(&hasChanged);

		hasChanged = classify_points_to_centroid(aPoints, aNumberOfPoints, aNumberOfPointsForCuda,
			aCentroids, aNumberOfCentroids);
		
		CudaGetDeviceCentroids(mCentroidsFromDevice, aNumberOfCentroids);

		omp_merge_centroids_results(aCentroids, mCentroidsFromDevice, aNumberOfCentroids);
		omp_update_centroids(aCentroids, aPoints, aNumberOfCentroids, aNumberOfPoints);
		currentIteration++;
		
	}

	//calculate minimum distance between centroids
	return calcMinDistanceBetweenCentroids(aCentroids, aNumberOfCentroids);;
}


void omp_update_centroids(Centroid* aCentroids, Point* aPoints, int* aNumberOfCentroids, int* aNumberOfPoints)
{

#pragma omp parallel for
	for (int i = 0; i < *aNumberOfCentroids; i++)
	{
		int tid = omp_get_thread_num();
		for (int j = 0; j < *aNumberOfPoints; j++)
		{
			if(aCentroids[i].mId == aPoints[j].mCentroidId)
			{
				aCentroids[i].mNumberOfPoints++;
				aCentroids[i].mXcount += aPoints[j].mX;
				aCentroids[i].mYCount += aPoints[j].mY;

			}
		}
		//in case there is isolated centroid without points assigned
		if (aCentroids[i].mNumberOfPoints > 0)
		{
			aCentroids[i].mX = aCentroids[i].mXcount / aCentroids[i].mNumberOfPoints;
			aCentroids[i].mY = aCentroids[i].mYCount / aCentroids[i].mNumberOfPoints;
			aCentroids[i].mNumberOfPoints = 0;
			aCentroids[i].mXcount = 0;
			aCentroids[i].mYCount = 0;
		}
	}
}


/*
update all points with  X and Y - Run parallel both cpu and gpu
*/
void initPoints(Point* aPoints, int* aNumberOfPoints, int* aNumberOfPointsForCuda, Centroid* aCentroids,
	int* aNumberOfCentroids, double* aCurrentInterval, double* aTotalTime)
{
	int numberOfPointsForHost = *aNumberOfPoints - *aNumberOfPointsForCuda;
	double PI = atan(1.0) * 4;
	double RadianAngle = 2 * PI * (*aCurrentInterval) / (*aTotalTime);
	double cosAngle = cos(RadianAngle), singAngle = sin(RadianAngle);
	bool cudaActivate = true;
	//update cuda with relevant angles
	CudaCopyAnglesFromHost(&cosAngle, &singAngle);

#pragma omp parallel for
	for (int i = 0; i < numberOfPointsForHost; i++)
	{
		int tid = omp_get_thread_num();
		if (tid == MASTER_THREAD && cudaActivate)
		{	
			cudaInitPoints(mPointsInDevice, aNumberOfPointsForCuda, mNumberOfPointsInDevice, mCosAngleInDevice,
				mSinAngleInDevice);
			cudaActivate = false;
		}
		aPoints[i].mX = aPoints[i].mA + aPoints[i].mR * cosAngle;
		aPoints[i].mY = aPoints[i].mB + aPoints[i].mR * singAngle;
		aPoints[i].mDistnceToCenter = DEFAULT_MAX_POINT_VALUE;
		aPoints[i].mCentroidId = 0;
		//set first points as centroids
		if (i < *aNumberOfCentroids)
		{
			aCentroids[i].mId = i;
			aCentroids[i].mX = aPoints[i].mX;
			aCentroids[i].mY = aPoints[i].mY;
			aCentroids[i].mNumberOfPoints = 0;
			aCentroids[i].mXcount = 0;
			aCentroids[i].mYCount = 0;
		}

	}
	
}



/*
clusterize each point with its relevant Centroid, using Cuda and openmp in parallel
*/
bool classify_points_to_centroid(Point* aPoints, int* aNumberOfPoints, int* aNumberOfPointsForCuda,
	Centroid* aCentroids, int* aNumberOfCentroids)
{
	bool pointChangedCentroidInCuda = false, pointChangedCentroidInOmp= false;
	bool activateCuda = true;
	int numberOfPointsForHost = *aNumberOfPoints - *aNumberOfPointsForCuda;
#pragma omp parallel for
	for (int i = 0; i < numberOfPointsForHost; i++) 
	{
		int tid = omp_get_thread_num();
		
		if (tid == MASTER_THREAD && activateCuda) {
			cudaCalcDistanceToCentroid(mPointsInDevice, aNumberOfPointsForCuda, mNumberOfPointsInDevice,
				mCentroidsInDevice, mNumberOfCentroidsInDevice, mHasUpdatedInDevice);
			CudaGetHasChangedFlag(&pointChangedCentroidInCuda);
			activateCuda = false;
		}
		pointChangedCentroidInOmp = omp_calc_distance_point_to_centroids(&aPoints[i], aCentroids, aNumberOfCentroids);

	}

	return pointChangedCentroidInCuda || pointChangedCentroidInOmp;
}

/*
calculate distance from point to each centroid and update the relevant centroid
*/
bool omp_calc_distance_point_to_centroids(Point* aPoint, Centroid* aCentroids, int* aNumberOfCentroids)
{

	for (int i = 0; i < (*aNumberOfCentroids); i++)
	{
		double dist = sqrt(pow(((*aPoint).mX - aCentroids[i].mX), 2) + pow(((*aPoint).mY - aCentroids[i].mY), 2));
		if (dist < (*aPoint).mDistnceToCenter)
		{
			(*aPoint).mDistnceToCenter = dist;
			(*aPoint).mCentroidId = aCentroids[i].mId;
			return true;
		}
	}
	return false;
}


/*Merge centroids arrays from GPU and CPU*/
void omp_merge_centroids_results(Centroid* aCentroids, Centroid* aCentroidsFromDevice, int* aNumberOfCentroids)
{
#pragma omp parallel for
	for (int i = 0; i < (*aNumberOfCentroids); i++)
	{
		int tid = omp_get_thread_num();
		aCentroids[i].mNumberOfPoints += aCentroidsFromDevice[i].mNumberOfPoints;
		aCentroids[i].mXcount += aCentroidsFromDevice[i].mXcount;
		aCentroids[i].mYCount += aCentroidsFromDevice[i].mYCount;
	}
}

double calcMinDistanceBetweenCentroids(Centroid* aCentroids, int* aNumberOfCentroids)
{	
	double minDistance = DBL_MAX;

//#pragma omp parallel for
	for (int i = 0; i < *aNumberOfCentroids - 1; i++)
	{
		//int tid = omp_get_thread_num();
		double dist = calcMinDistanceFromCentroids(aCentroids[i], aCentroids, aNumberOfCentroids);
		if (dist < minDistance)	//probably problem here beacause minDistance is shared
		{
			minDistance = dist;
		}
	}
	return minDistance;
}

double calcMinDistanceFromCentroids(Centroid aCentroid, Centroid* aCentroids, int* aNumberOfCentroids)
{
	double dist = DBL_MAX;
	for (int i = 0; i < *aNumberOfCentroids; i++)
	{
		//avoid centroid to itself check
		if (aCentroid.mId != aCentroids[i].mId)
		{
			double tmpDist = sqrt(pow((aCentroid.mX - aCentroids[i].mX), 2) + pow((aCentroid.mY - aCentroids[i].mY), 2));
			if(tmpDist < dist){
				dist = tmpDist;
			}
		}
	}
	return dist;
}

void prepareCuda(int numberOfElementsForCuda, int aNumberOfCentroids)
{
	//Allocate relevant data in GPU(Device)
	statusCuda(cudaMalloc((void**)&mPointsInDevice, numberOfElementsForCuda * sizeof(Point)), "prepareCuda - allocated points!\n");//points
	statusCuda(cudaMalloc((void**)&mCentroidsInDevice, aNumberOfCentroids * sizeof(Centroid)), "prepareCuda - allocated Centroids !\n"); //centroids
	statusCuda(cudaMalloc((void**)&mNumberOfPointsInDevice, sizeof(int)), "prepareCuda - allocated number of points !\n");
	statusCuda(cudaMalloc((void**)&mNumberOfCentroidsInDevice, sizeof(int)), "prepareCuda - allocated number of centroids !\n");
	statusCuda(cudaMalloc((void**)&mCosAngleInDevice, sizeof(double)), "prepareCuda - allocated cos angle !\n");
	statusCuda(cudaMalloc((void**)&mSinAngleInDevice, sizeof(double)), "prepareCuda - allocated sin angle !\n");
	statusCuda(cudaMalloc((void**)&mHasUpdatedInDevice, sizeof(bool)), "prepareCuda - allocated update flag !\n");
}


void CudaCopyOriginalPointsFromHost(Point* aPoints, int* aNumberOfElementsForCuda, int* aNumberOfCentroids)
{
	statusCuda(cudaMemcpy(mPointsInDevice, aPoints, *aNumberOfElementsForCuda * sizeof(Point), cudaMemcpyHostToDevice), "CudaCopyOriginalPointsFromHost - copy points!\n");//points
	statusCuda(cudaMemcpy(mNumberOfPointsInDevice, aNumberOfElementsForCuda, sizeof(int), cudaMemcpyHostToDevice), "CudaCopyOriginalPointsFromHost - copy num of points \n");//num of points
	statusCuda(cudaMemcpy(mNumberOfCentroidsInDevice, aNumberOfCentroids, sizeof(int), cudaMemcpyHostToDevice), "CudaCopyOriginalPointsFromHost - copy num of centroids \n");//num of centroids
}

/*Copy all relevant Data */
void CudaCopyAnglesFromHost(double* aCosAngle, double* aSinAngle)
{
	statusCuda(cudaMemcpy(mCosAngleInDevice, aCosAngle, sizeof(double), cudaMemcpyHostToDevice), "CudaCopyAnglesFromHost - copy cos angle!\n");
	statusCuda(cudaMemcpy(mSinAngleInDevice, aSinAngle, sizeof(double), cudaMemcpyHostToDevice), "CudaCopyAnglesFromHost - copy sin angle!\n");
}

/*Copy all relevant Data */
void CudaCopyCentroidsFromHost(Centroid* aCentroids, int* aNumberOfCentroids)
{
	statusCuda(cudaMemcpy(mCentroidsInDevice, aCentroids, (*aNumberOfCentroids) * sizeof(Centroid), cudaMemcpyHostToDevice), "CudaCopyCentroidsFromHost - copy centroids from host to Device !\n");
}

/*return the result from cuda*/
void CudaGetHasChangedFlag(bool* aAnswer)
{
	statusCuda(cudaMemcpy(aAnswer, mHasUpdatedInDevice, sizeof(bool), cudaMemcpyDeviceToHost), "cuda Get hasChanged Flag From Device\n");
}

/*return the result from cuda*/
void CudaSetHasChangedFlag(bool* aHasChanged)
{
	statusCuda(cudaMemcpy(mHasUpdatedInDevice, aHasChanged, sizeof(bool), cudaMemcpyHostToDevice), "cuda Set hasChanged Flag In Device\n");
}

/*return the result from cuda*/
void CudaGetDeviceCentroids(Centroid* tempHostCentroids, int* numberOfCentroids)
{
	statusCuda(cudaMemcpy(tempHostCentroids, mCentroidsInDevice, *numberOfCentroids * sizeof(Centroid), cudaMemcpyDeviceToHost), "Cuda Get Centroids - copy centroids from device to host!\n");
}

/*Free all cuda allocations*/
void CudaFreeAll(int procId)
{
	cudaFree(mPointsInDevice);
	cudaFree(mCentroidsInDevice);
	cudaFree(mNumberOfCentroidsInDevice);
	cudaFree(mNumberOfPointsInDevice);
	cudaFree(mCosAngleInDevice);
	cudaFree(mSinAngleInDevice);
	free(mCentroidsFromDevice);
	printf("proc [%d] Free all cuda data...\n", procId);
	fflush(stdout);
}

void statusCuda(cudaError_t cudaStatus, char* statusMessage)
{
	/*printf("Step -> %s", statusMessage);
	fflush(stdout);

	if (cudaStatus == cudaSuccess)
	{
		printf("Succeed\n");
		fflush(stdout);
	}
	else{
		printf("fail[%s]\n", cudaGetErrorString(cudaStatus));
		fflush(stdout);
	}*/
}

