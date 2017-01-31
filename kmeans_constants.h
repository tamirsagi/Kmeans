#include <string>

using namespace std;

const int POINT_STRUCT_ELEMENTS = 8;
const int CENTROID_STRUCT_ELEMENTS = 6;
//MPI DEFAULT
const static int MASTER_ID = 0;

const int DEFAULT_FILE_PATH_INDEX = 0;
const int NUMBER_OF_INITIAL_ELEMENTS_IN_POINTS = 4; // A,B,R
const int NUMBER_OF_SHARED_ELEMENTS = 4;			// Number of points, number of centroids, total time t and max iteration
const int NUMBER_OF_CELLS_SIZE = 1;
const int NUMBER_OF_RESULT_ELEMENTS = 2; //min distance and delta T
const int NUMBER_OF_CENTROID_ELEMENTS = 2; //x,y

//data indices
const int DATA_ARRAY_SIZE_INDEX = 0;
const int DATA_ARRAY_NUMBER_OF_POINTS_INDEX = 1;
const int DATA_ARRAY_NUMBER_OF_CENTROIDS_INDEX = 2;
const int DATA_ARRAY_TOTAL_TIME_INDEX = 3;
const int DATA_ARRAY_MAX_ITERATIONS_INDEX = 4;

//result indices
const int RESULT_ARRAY_MIN_DISTANCE_INDEX = 0;
const int RESULT_ARRAY_DELTA_T_INDEX = 1;

//Params From File

// DEFAULT
//const string DEFAULT_FILE_NAME = "d:\\Kmeans\\Kmeans\\kmeans_data_set.txt";
const string DEFAULT_FILE_NAME = "C:\\Program Files\\MPICH2\\bin\\kmeans_data_set.txt";
const string DEFAULT_OUTPUT_FILE_NAME = "C:\\Program Files\\MPICH2\\bin\\KmeansResult.txt";
//const string DEFAULT_OUTPUT_FILE_NAME = "d:\\Kmeans\\Kmeans\\KmeansResult.txt";
const int DEFAULT_MIN_POINTS = 1;
const int DEFAULT_MAX_POINTS = 300000;
const double DEFAULT_DELTA_TIME = 0.1;
const int DEFAULT_MAX_TIME = 100;
const int DEFAULT_MAX_ITERATIONS = 100;
const int DEFAULT_MAX_RANGE = 100;
const int DEFAULT_MAX_NUMBER_OF_CENTROIDS = 3;

//Kmeans data
const int NUMBER_OF_POINTS_INDEX = 0;
const int NUMBER_OF_CENTROIDS_INDEX = 1;
const int DELTA_TIME_INDEX = 2;
const int TOTAL_TIME_INDEX = 3;
const int MAX_ITERATIONS_INDEX = 4;

//Points Data
const int POINT_ID_INDEX = 0;
const int POINT_A_COORDINATE_INDEX = 1;
const int POINT_B_COORDINATE_INDEX = 2;
const int POINT_RADIUS_INDEX = 3;

//Tags
const int TAG_NEW_TASK = 0;
const int TAG_RESULT = 1;
const int TAG_POINTS = 2;