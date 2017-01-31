#pragma once

#include <iostream>
#include <string>
using namespace std; 

const int DEFAULT_MIN_POINT_VALUE = INT_MIN;
const int DEFAULT_MAX_POINT_VALUE = INT_MAX;

const int MIN_ID = 0;
const int DEFAULT_VALUE = -1;

struct Centroid {
	int mId;
	double mX;
	double mY;
	int mNumberOfPoints;
	double mXcount;
	double mYCount;

};

struct Point {
	int mId;
	double mA;
	double mB;
	double mR;
	double mDistnceToCenter;
	int mCentroidId;
	double mX;
	double mY;

}typedef Point;

struct KmeansResult {
	
	double mTotalTime;
	double mMinDistance;
	double mDeltaTime;
	int mNumberOfCentroids;
	int mNumberOfPoints;
	int mNumberOfProceessors;
	Centroid* mCentroids;

}typedef KmeansResult;