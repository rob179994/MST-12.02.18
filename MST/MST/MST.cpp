#pragma once
#include "stdafx.h"
#include <windows.h>


// my own
//#include "TreeEdge.h" 
//#include "TreeNode.h"
#include "MST.h"

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
	// find directory
	/*if (argc < 2)
	{
		cout << _T("Enter the dir from which to find the stereo files.") << endl;
		return 1;
	}
	
	// directories from user input
	string dataFolder = argv[1];

	// max length of directory
	char path[MAX_PATH];
	if (dataFolder.length() > MAX_PATH - 11)
	{
		cerr << _T("Your path is too long.") << endl;
		return -1;
	}

	WIN32_FIND_DATAA FindFileData;

	string firstLeftImage = dataFolder + "\\left\\" +"*";
	HANDLE hFind = FindFirstFileA(firstLeftImage.c_str(), &FindFileData);

	vector<string> leftImageNames;
	if (hFind== INVALID_HANDLE_VALUE) {
		cout<<"No png or jpg files found in left folder"<<endl;
		return 0;
	}
	else {
		while (FindNextFileA(hFind, &FindFileData)) {
			string ws(FindFileData.cFileName);

			if (ws.find(".jpg") != std::string::npos || ws.find(".png") != std::string::npos) {
				leftImageNames.push_back(ws);
			}
		}
	}
	FindClose(hFind);

	string firstRightImage = dataFolder + "\\right\\" + "*";
	hFind = FindFirstFileA(firstRightImage.c_str(), &FindFileData);
	vector<string> rightImageNames;
	if (hFind == INVALID_HANDLE_VALUE) {
		cout << "No png or jpg files found in right folder" << endl;
		return 0;
	}
	else {
		while (FindNextFileA(hFind, &FindFileData)) {
			string ws(FindFileData.cFileName);

			if (ws.find(".jpg") != std::string::npos|| ws.find(".png") != std::string::npos) {
				rightImageNames.push_back(ws);
			}
		}
	}
	FindClose(hFind);

	cout << endl << leftImageNames.size() << " Left Images"<< endl;
	cout << rightImageNames.size() << " Right Images" << endl << endl;*/

	/*Mat image1;
	Mat image2;
	//cout << dataFolder + "\\right\\" + rightImageNames[0] << endl;
	image1 = imread(dataFolder + "\\right\\" + rightImageNames[0], CV_LOAD_IMAGE_GRAYSCALE);
	image2 = imread(dataFolder + "\\left\\" + leftImageNames[0], CV_LOAD_IMAGE_GRAYSCALE);
	namedWindow("Right", WINDOW_AUTOSIZE);
	namedWindow("Left", WINDOW_AUTOSIZE);
	imshow("Right", image1);
	imshow("Left", image2);
	waitKey(0);*/

	

	








	
	// TEMP CREATING OWN MAT///
	/*
	 Mat imgLeft(4, 4, CV_8UC1);;
	 imgLeft.at<uchar>(0, 0) = 1;
	 imgLeft.at<uchar>(0, 1) = 7;
	 imgLeft.at<uchar>(0, 2) = 3;
	 imgLeft.at<uchar>(0, 3) = 4;
	 imgLeft.at<uchar>(1, 0) = 10;
	 imgLeft.at<uchar>(1, 1) = 0;
	 imgLeft.at<uchar>(1, 2) = 1;
	 imgLeft.at<uchar>(1, 3) = 10;
	 imgLeft.at<uchar>(2, 0) = 9;

	 imgLeft.at<uchar>(2, 1) = 5;
	 imgLeft.at<uchar>(2, 2) = 2;

	 imgLeft.at<uchar>(2, 3) = 8;
	 imgLeft.at<uchar>(3, 0) = 1;
	 imgLeft.at<uchar>(3, 1) = 3;
	 imgLeft.at<uchar>(3, 2) = 9;
	 imgLeft.at<uchar>(3, 3) = 0;


	 Mat imgRight(4, 4, CV_8UC1);;
	 imgRight.at<uchar>(0, 0) = 2;
	 imgRight.at<uchar>(0, 1) = 7;
	 imgRight.at<uchar>(0, 2) = 9;
	 imgRight.at<uchar>(0, 3) = 1;
	 imgRight.at<uchar>(1, 0) = 0;
	 imgRight.at<uchar>(1, 1) = 2;
	 imgRight.at<uchar>(1, 2) = 7;
	 imgRight.at<uchar>(1, 3) = 10;
	 imgRight.at<uchar>(2, 0) = 0;
	 imgRight.at<uchar>(2, 1) = 5;
	 imgRight.at<uchar>(2, 2) = 3;
	 imgRight.at<uchar>(2, 3) = 4;
	 imgRight.at<uchar>(3, 0) = 1;
	 imgRight.at<uchar>(3, 1) = 7;
	 imgRight.at<uchar>(3, 2) = 8;
	 imgRight.at<uchar>(3, 3) = 3;
	 */
	 // TEMP

	Mat imgLeft, imgEdgeLeftx;
	Mat imgRight, imgEdgeRight;
	imgLeft = imread("C:/Users/Rob McCormack/Documents/Visual Studio 2017/Projects/MST/x64/Release/image_L.png", IMREAD_GRAYSCALE);
	imgRight = imread("C:/Users/Rob McCormack/Documents/Visual Studio 2017/Projects/MST/x64/Release/image_R.png", IMREAD_GRAYSCALE);

	if (!imgLeft.data || !imgRight.data)	// error check for the image files
	{
		printf(" Not enough image data \n ");
		return -1;
	}

	//imgLeft = np.power(grayL, 0.75).astype('uint8')
	//imgRight = np.power(grayR, 0.75).astype('uint8')


	//resize(imgLeft, imgLeft, Size(), 0.25, 0.25, INTER_AREA);
	//resize(imgRight, imgRight, Size(), 0.25, 0.25, INTER_AREA);

	// Make minspantree object to access relevent functions
	MST mst;
	int squareDimension = 3;
	//int disparityRange = 5;
	int disparityRange = imgLeft.cols / 4;
	if (disparityRange%2==0) {
		disparityRange++;
	}

	int a = 1;
	int b = 0;
	int minD = 0;
	float sigma = 0.01f;
	//double focalLength = 399.9745178222656;
	//double baseLineDistance = 0.2090607502;
	double focalLength = 3740;
	double baseLineDistance = 0.160;


	clock_t t;
	t = clock();
	vector<vector<double>> disparityMatrix = mst.findDisparities(imgLeft,imgRight,a,b,disparityRange, squareDimension,sigma);

	//mst.print2DVector(disparityMatrix);
	// min = 270
	Mat disparityImage = mst.disparityImage(disparityMatrix, 0);
	
	//vector<vector<double>> depthMatrix = mst.calculateDepth(disparityMatrix, baseLineDistance , focalLength);
	//Mat depthImage = mst.depthImage(depthMatrix);
	Mat depthImage = mst.calculateDepthMat(disparityMatrix, baseLineDistance, focalLength);



	t = clock() - t;
	cout << "Total time: " << ((float)t) / CLOCKS_PER_SEC << endl;
	
	imwrite("Disparity.png",disparityImage);
	imwrite("Depth.png", depthImage);
	
	namedWindow("Left", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("Left", imgLeft);

	namedWindow("Right", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("Right", imgRight);

	namedWindow("Disparity", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("Disparity", disparityImage);

	namedWindow("Depth", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("Depth", depthImage);

	waitKey(0);

	//Mat difference = mst.differenceBetweenTwoImages(imgLeft,imgRight);
	//imwrite("Difference.png", difference);
	//imshow("Disparity", difference);

	return 0;
}

