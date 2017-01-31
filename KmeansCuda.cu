#include "kmeans_cuda.h"


// ----------------------------- Cuda methods ----------------------------- //

/*Update points and centroids correspondingly*/
__global__ void CudaInitPoints(Point* aPointsInDevice, int* aNumberOfPointsInDeivce, double* aCosAngle, double* aSinAngle)
{

	int i = threadIdx.x + (blockIdx.x * blockDim.x);
	if (i < *aNumberOfPointsInDeivce)
	{
		aPointsInDevice[i].mX = aPointsInDevice[i].mA + aPointsInDevice[i].mR * (*aCosAngle);
		aPointsInDevice[i].mY = aPointsInDevice[i].mB + aPointsInDevice[i].mR * (*aSinAngle);
		aPointsInDevice[i].mDistnceToCenter = double(INT_MAX);
		aPointsInDevice[i].mCentroidId = 0;
	}
	
}


/*Calc distance from points to	 centroids*/
__global__ void CudaCalcDistanceToCentroid(Point* aPointsInDevice, int* aNumberOfPointsInDeivce, Centroid* aCentroidsInDevice, int* aNumberOfCentroids, bool* aHasChanged)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int i = tid + (bid * MAX_THREADS_PER_BLOCK);
	if (i < *aNumberOfPointsInDeivce)
	{
		for (int j = 0; j < *aNumberOfCentroids; j++)
		{
			double dist = sqrt(pow((aPointsInDevice[i].mX - aCentroidsInDevice[j].mX), 2) + pow((aPointsInDevice[i].mY - aCentroidsInDevice[j].mY), 2));
			if (dist < aPointsInDevice[i].mDistnceToCenter)
			{
				aPointsInDevice[i].mDistnceToCenter = dist;
				aPointsInDevice[i].mCentroidId = aCentroidsInDevice[j].mId;
				*aHasChanged = true;
			}
		}
		if (*aHasChanged)
		{
			int centroidToUpdate = aPointsInDevice[i].mCentroidId;
			aCentroidsInDevice[centroidToUpdate].mNumberOfPoints++;
			aCentroidsInDevice[centroidToUpdate].mXcount += aPointsInDevice[i].mX;
			aCentroidsInDevice[centroidToUpdate].mYCount += aPointsInDevice[i].mY;
		}
	}
}



void cudaInitPoints(Point* aPointsInDevice,int* aNumberOfPointsForCuda, int* aNumberOfPointsInDeivce ,double* aCosAngle, double* aSinAngle)
{
	CudaInitPoints << <calculateNumOfBlocks(aNumberOfPointsForCuda), MAX_THREADS_PER_BLOCK >> >(aPointsInDevice, aNumberOfPointsInDeivce, aCosAngle, aSinAngle);
}

void cudaCalcDistanceToCentroid(Point* aPointsInDevice, int* aNumberOfPointsForCuda, int* aNumberOfPointsInDeivce, Centroid* aCentroidsInDevice, int* aNumberOfCentroids, bool* aHasChanged)
{
	CudaCalcDistanceToCentroid << <calculateNumOfBlocks(aNumberOfPointsForCuda), MAX_THREADS_PER_BLOCK >> >(aPointsInDevice, aNumberOfPointsInDeivce, aCentroidsInDevice, aNumberOfCentroids, aHasChanged);
}


int calculateNumOfBlocks(int* aNumOfPoints) {
	return *aNumOfPoints/MAX_THREADS_PER_BLOCK + 1;
}

