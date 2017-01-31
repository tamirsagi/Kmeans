
#include "KmeansFilesManager.h"
#include <omp.h>


/*
build all necessary data prior kmeans algorithm begins
*/
Point* createInitialData(string aFileName, int* aNumberOfPoints,int* aNumberOfCentroids, double* aTimeInterval,
	double* aTotalTime, int* aMaxIterations)
{
	fstream file(aFileName, ios::in);
	
	file >> *aNumberOfPoints;
	file >> *aNumberOfCentroids;
	file >> *aTimeInterval;
	file >> *aTotalTime;
	file >> *aMaxIterations;
	Point* aPoints = new Point[*aNumberOfPoints];

	int id;
	double a, b, radius;
	while (true) {
		if (file.eof()) {
			break;
		}
		file >> id;
		file >> a;
		file >> b;
		file >> radius;
		aPoints[id].mId = id;
		aPoints[id].mA = a;
		aPoints[id].mB = b;
		aPoints[id].mR = radius;
		aPoints[id].mDistnceToCenter= double(INT_MAX);
		aPoints[id].mCentroidId = 0;
		aPoints[id].mX = 0;
		aPoints[id].mY = 0;
	}
	return aPoints;
}

vector<string> split(string aContext, string aDelimeter)
{
	vector<string> splited;
	size_t pos = aContext.find(aDelimeter);
	int current = 0, size = 0, len = aContext.length();
	while (pos != std::string::npos){
		splited.push_back(aContext.substr(0, pos));
		size++;
		aContext = aContext.substr(pos + 1);
		pos = aContext.find(aDelimeter);
	}
	if (!aContext.empty()){
		splited.push_back(aContext);
	}
	return splited;
}


bool generateFile(string aFileName, int pointsNum, int k, double dt, double T, int limit, int maxRange) {

	ofstream myfile;
	myfile.open(aFileName);
	string str = "";
	str += to_string(static_cast<long long>(pointsNum)) + " ";
	str += to_string(static_cast<long long>(k)) + " ";
	str += to_string(static_cast<long double>(dt)) + " ";
	str += to_string(static_cast<long double>(T)) + " ";
	str += to_string(static_cast<long long>(limit)) + "\n";
	myfile << str;

	str = "";
	for (int i = 0; i < pointsNum; i++){
		str.append(to_string(static_cast<long long>(i)));
		for (int j = 0; j < 2; j++){
			str.append(" ");
			str.append(to_string(static_cast<long double>(fRand(0, maxRange)))); // A,B
		}
		str.append(" " + to_string(static_cast<long double>(fRand(0, maxRange)))); //radius
		str.append("\n");
	}
	myfile << str;
	myfile.close();
	return true;
}

double fRand(double fMin, double fMax)
{
	double f = (double)rand() / RAND_MAX;
	return fMin + f * (fMax - fMin);
}

void SaveResultToFile(KmeansResult aResult, string aFileName)
{
	printf("Saving result into file\n");
	fflush(stdout);
	ofstream outputFile;
	outputFile.open(aFileName);
	string str = "Kmeans Result:\n";
	str += "Number Of Processors:" + to_string(static_cast<long long>(aResult.mNumberOfProceessors)) + "\n";
	str += "Total Time:" + to_string(static_cast<long double>(aResult.mTotalTime)) + " Seconds\n";
	str += "Min Distance was:" + to_string(static_cast<long double>(aResult.mMinDistance)) + "\n";
	str += "Delta t: " + to_string(static_cast<long double>(aResult.mDeltaTime)) + "\n";
	str += "Points:" + to_string(static_cast<long long>(aResult.mNumberOfPoints)) + "\n";
	str += "Centroids:" + to_string(static_cast<long long>(aResult.mNumberOfCentroids)) + "\n";
	for (int i = 0; i < aResult.mNumberOfCentroids; i++)
	{
		str += "X: " + to_string(static_cast<long double>(aResult.mCentroids[i].mX)) + " ";
		str += "Y: " + to_string(static_cast<long double>(aResult.mCentroids[i].mY)) + "\n";
	}
	outputFile << str;
	outputFile.close();
}
