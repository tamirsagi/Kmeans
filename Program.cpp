/*
Main Program class, prepare all data and data structure prior start working on kmeans algorithms
*/

#include "program.h"


int main(int argc, char *argv[])
{
	double DONE = -1.0;

	int mNumprocs;
	int mMyid;
	char processor_name[MPI_MAX_PROCESSOR_NAME];

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &mMyid);
	MPI_Comm_size(MPI_COMM_WORLD, &mNumprocs);
	MPI_Status status;

	MPI_Datatype point_Type = CreatePointMpiType();

	double mTotalTime; // T
	double mTimeInterval; //Delta t
	double mCurrentInterval; //current t
	int mNumberOfPoints;
	int mNumberOfCentroids;
	double mCurrentTime;
	int mMaxIterations;
	bool mIsWorking;
	int mIsReadyForNewTask = 1;

	double mMinDistanceBetweenCentroids = DBL_MAX;
	double mDeltaTOnMinDistance;
	
	KmeansResult mFinalResult;
	Point* mPoints = NULL;
	Centroid* mCentroids = NULL;
	double* values = NULL;
	double* result;
	int numberOfElementsInResultArray;


	double startTime = MPI_Wtime();	

	mPoints = createInitialData(DEFAULT_FILE_NAME, &mNumberOfPoints, &mNumberOfCentroids, &mTimeInterval,&mTotalTime, &mMaxIterations);

	//check if file was generated properly
	if (mPoints == NULL)
	{
		printf("File was corrupted, generating a new file\n");
		fflush(stdout);
		if (generateFile(DEFAULT_FILE_NAME, DEFAULT_MAX_POINTS, DEFAULT_MAX_NUMBER_OF_CENTROIDS,
				DEFAULT_DELTA_TIME, DEFAULT_MAX_TIME, DEFAULT_MAX_ITERATIONS, DEFAULT_MAX_RANGE))
		{
			mPoints = createInitialData(DEFAULT_FILE_NAME, &mNumberOfPoints, &mNumberOfCentroids, &mTimeInterval,
				&mTotalTime, &mMaxIterations);
		}
		else
		{
			printf("Could not create file, aborting...\n");
			fflush(stdout);
			MPI_Abort(mNumprocs, 1);
		}
	}

		
	
	//set result parameters
	mFinalResult.mMinDistance = DEFAULT_MAX_POINT_VALUE;
	numberOfElementsInResultArray = NUMBER_OF_RESULT_ELEMENTS + NUMBER_OF_CENTROID_ELEMENTS * mNumberOfCentroids;
	result = new double[numberOfElementsInResultArray];
	mCentroids = new Centroid[mNumberOfCentroids];
	
	
	if (mMyid == MASTER_ID) 
	{
		printf("mNumberOfPoints : %d\nmNumberOfCentroids : %d\nmTimeInterval : %f\nmTotalTime : %f\nmMaxIterations : %d\n",
		mNumberOfPoints, mNumberOfCentroids, mTimeInterval, mTotalTime, mMaxIterations);
		fflush(stdout);

		//send jobs
		for (double mCurrentInterval = 0; mCurrentInterval <= mTotalTime; mCurrentInterval += mTimeInterval)
		{
			MPI_Recv(&mIsReadyForNewTask, 1, MPI_INT, MPI_ANY_SOURCE, TAG_NEW_TASK, MPI_COMM_WORLD, &status);
			printf("Master Received New Task request from Proc[%d] , sending interval [%f] : to proc [%d] time: %f\n",
				status.MPI_SOURCE, mCurrentInterval, status.MPI_SOURCE, (MPI_Wtime() - startTime));
			fflush(stdout);
			MPI_Send(&mCurrentInterval, 1, MPI_DOUBLE, status.MPI_SOURCE, TAG_NEW_TASK, MPI_COMM_WORLD); //send request for new task
		}
		
		printf("Master finished sending tasks, time passesd: %f\n" , (MPI_Wtime() - startTime));
		fflush(stdout);

		//Wait for all proccess to send new task ane notify done!
		int numProcces = mNumprocs - 1;
		
		for (int procces = 0; procces < numProcces; procces++)
		{
			MPI_Recv(&mIsReadyForNewTask, 1, MPI_DOUBLE, MPI_ANY_SOURCE, TAG_NEW_TASK, MPI_COMM_WORLD, &status);
			printf("Master Received New Task request from Proc[%d] , sending DONE [%f] : to proc [%d]\n", status.MPI_SOURCE, DONE, status.MPI_SOURCE);
			fflush(stdout);
			MPI_Send(&DONE, 1, MPI_DOUBLE, status.MPI_SOURCE, TAG_NEW_TASK, MPI_COMM_WORLD); //send request for new task
		}

		

		//Get all Relevant Data(Min distance , time occured and clusters) from all proccessors
		for (int procces = 0; procces < numProcces; procces++)
		{
			printf("Master Received Result  from Proc[%d]\n", status.MPI_SOURCE);
			fflush(stdout);

			MPI_Recv(result, numberOfElementsInResultArray, MPI_DOUBLE, MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &status);
			
			if (result[RESULT_ARRAY_MIN_DISTANCE_INDEX] < mFinalResult.mMinDistance)
			{
				buildResultObject(&mFinalResult, result, numberOfElementsInResultArray, mNumberOfCentroids);
			}
		}

		printf("finished algorithm, time passesd: %f\n" , (MPI_Wtime() - startTime));
		fflush(stdout);

		double endTime = MPI_Wtime();
		mFinalResult.mTotalTime = endTime - startTime;
		mFinalResult.mNumberOfProceessors = mNumprocs;
		mFinalResult.mNumberOfPoints = mNumberOfPoints;
		mFinalResult.mNumberOfCentroids = mNumberOfCentroids;
		SaveResultToFile(mFinalResult, DEFAULT_OUTPUT_FILE_NAME);
		free(mFinalResult.mCentroids);
		clearAll(mPoints, mCentroids, values,result, mMyid);
	}
	else //////////////////////////////   slaves //////////////////////////////
	{

		/*printf("proc:[%d] start slaves work..., time passesd: %f\n" ,mMyid , (MPI_Wtime() - startTime));
		fflush(stdout);*/
		
		int numberOfElementsForCuda = mNumberOfPoints / 2; // (assume 0.5n > K) , Cuda gets half array --> maybe needs to pick the correct number by number of centroids
		
		/*printf("proc:[%d] start preapering data for cuda in slave..., time passesd: %f\n" ,mMyid , (MPI_Wtime() - startTime));
		fflush(stdout);*/
		
		prepareCuda(numberOfElementsForCuda, mNumberOfCentroids);

		/*printf("proc:[%d] Copy array to cuda..., time passesd: %f\n" , mMyid, (MPI_Wtime() - startTime));
		fflush(stdout);*/

		//init cuda params prior working
		CudaCopyOriginalPointsFromHost(mPoints + numberOfElementsForCuda, &numberOfElementsForCuda, 
			&mNumberOfCentroids);

		/*printf("proc:[%d] finished copy to cuda, time passesd: %f\n" , mMyid, (MPI_Wtime() - startTime));
		fflush(stdout);*/

		mIsWorking = true;

		/*printf("proc [%d] num threads [%d] \n",mMyid, omp_get_num_threads());
		fflush(stdout);*/
		while (mIsWorking)
		{
			
			MPI_Send(&mIsReadyForNewTask, 1, MPI_INT, MASTER_ID, TAG_NEW_TASK, MPI_COMM_WORLD); //send request for new task
			MPI_Recv(&mCurrentInterval, 1, MPI_DOUBLE, MASTER_ID, TAG_NEW_TASK, MPI_COMM_WORLD, &status);
			/*printf("Slave [%d] recived task [%f]\n", mMyid, mCurrentInterval);
			fflush(stdout);*/
			if (mCurrentInterval == DONE)
			{
				mIsWorking = false;
				printf("proc:[%d] Received DONE -->Sending Results!!! time passesd: %f\n",mMyid, (MPI_Wtime() - startTime));
				fflush(stdout);
				buildResultArray(result, mMinDistanceBetweenCentroids, mDeltaTOnMinDistance, mCentroids,mNumberOfCentroids);
				MPI_Send(result, numberOfElementsInResultArray, MPI_DOUBLE, MASTER_ID, TAG_RESULT, MPI_COMM_WORLD); //send request for new task
				//clearAll(mPoints, mCentroids, values,result, mMyid);
			}
			else
			{
				
				/*printf("proc:[%d] start kmeans calc...time passesd: %f\n", mMyid, (MPI_Wtime() - startTime));
				fflush(stdout);*/

				double minDistance = StartKmeansCalc(mPoints, &mNumberOfPoints, &numberOfElementsForCuda,
					mCentroids, &mNumberOfCentroids, &mCurrentInterval, &mTotalTime, &mMaxIterations, mMyid);
				
				/*printf("proc:[%d] finish kmeans calc...time passesd: %f\n", mMyid, (MPI_Wtime() - startTime));
				fflush(stdout);*/

				if (minDistance < mMinDistanceBetweenCentroids)
				{
					mMinDistanceBetweenCentroids = minDistance;
					mDeltaTOnMinDistance = mCurrentInterval;
				}
			}
		}
	}
	
	MPI_Finalize();
	return 0;
}


///////////////////////////////////////////// Methods \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


/*
 build result array when slave is done working
*/
void buildResultArray(double* aResultArr, double aMinDistance, double aDeltaTime, Centroid* centroids, int aNumberOfCentroids)
{
	aResultArr[RESULT_ARRAY_MIN_DISTANCE_INDEX] = aMinDistance;
	aResultArr[RESULT_ARRAY_DELTA_T_INDEX] = aDeltaTime;
	for (int i = NUMBER_OF_RESULT_ELEMENTS , j = 0; j < aNumberOfCentroids; j++)
	{
		aResultArr[i++] = centroids[j].mX;
		aResultArr[i++] = centroids[j].mY;
	}
}

/*
build result object from array of doubles(result from slaves)
*/
void buildResultObject(KmeansResult* aResult, double* aResultArr, int aResultArraySize, int aNumberOfCentroid)
{
	aResult->mMinDistance = aResultArr[RESULT_ARRAY_MIN_DISTANCE_INDEX];
	aResult->mDeltaTime = aResultArr[RESULT_ARRAY_DELTA_T_INDEX];
	aResult->mCentroids = new Centroid[aNumberOfCentroid];
	for (int j = NUMBER_OF_RESULT_ELEMENTS, i = 0; i < aNumberOfCentroid; i++)
	{
		aResult->mCentroids[i].mX = aResultArr[j++];
		aResult->mCentroids[i].mY = aResultArr[j++];
	}
}

void clearAll(Point* aPoints, Centroid* aCentroid, double* aInputData, double* aResult, int procId)
{
	delete[](aPoints);
	delete[](aCentroid);
	delete[](aInputData);
	delete[](aResult);
	printf("proc [%d] Free all data...\n",procId);
	fflush(stdout);
	CudaFreeAll(procId);
}