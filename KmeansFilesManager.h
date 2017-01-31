#pragma once

#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>     /* srand, rand */

#include <time.h>       /* time */
#include <iostream>
#include <fstream>      
#include <vector>

//My library
#include "structs.h"
#include "kmeans_constants.h"

using namespace std;

string readFile(string aPath);
Point* createInitialData(string aFileName,int* aNumberOfPoints,int* aNumberOfCentroids, double* aTimeInterval,
	double* aTotalTime, int* aMaxIterations);
vector<string> split(string aContext, string aDelimeter);
bool generateFile(void);
bool generateFile(string aFileName, int pointsNum, int k, double dt, double T, int limit, int maxRange);
void SaveResultToFile(KmeansResult aResult, string aFileName);
double fRand(double fMin, double fMax);