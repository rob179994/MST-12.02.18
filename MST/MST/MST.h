using namespace std;

#include <opencv2/opencv.hpp>
#include <stdio.h>
//#include <cmath>
//#include <math.h> 
#include <time.h> 

//#include "TreeEdge.h" 
//#include "TreeNode.h"

using namespace cv;

class MST {
public:

	// A structure to represent a subset for union-find
	struct nodeStructure
	{
		int parent;
		int root;
		int rank;
		int index;
		vector<int> children;
		float weightToParent;
		int xCoord;
		int yCoord;

		bool operator< (const nodeStructure &other) const {
			return rank < other.rank;
		}

	};

	struct TreeEdge {
		int sourceNodeIndex;
		int destinationNodeIndex;
		double weight;

		bool operator< (const TreeEdge &other) const {
			return weight < other.weight;
		}
	};

	void Union(vector<nodeStructure> &allNodes, int rootOne, int rootTwo, int nodeOne, int nodeTwo,double edgeWeight) {

		// Attach smaller rank tree under root of high 
		// rank tree (Union by Rank)
		if (allNodes[rootOne].rank < allNodes[rootTwo].rank) {
			if (allNodes[nodeOne].root !=nodeOne) { // check subtree is in correct order
				reverseSubTree(allNodes,nodeOne, allNodes[allNodes[nodeOne].root].rank, nodeOne, nodeOne);
				rootOne = nodeOne;
			}
			allNodes[rootOne].root = rootTwo;
			setRootForChildren(allNodes,rootTwo,nodeOne);
			allNodes[nodeOne].parent = nodeTwo;
			allNodes[nodeOne].weightToParent = edgeWeight;
			allNodes[nodeTwo].children.push_back(nodeOne);
			setRankAbove(allNodes,nodeOne,allNodes[nodeOne].rank);
		}
		else if (allNodes[rootOne].rank > allNodes[rootTwo].rank) {

			if (allNodes[nodeTwo].root != nodeTwo) { // check subtree is in correct order
				reverseSubTree(allNodes, nodeTwo, allNodes[allNodes[nodeTwo].root].rank, nodeTwo, nodeTwo);
				rootTwo = nodeTwo;
			}

			allNodes[rootTwo].root = rootOne;
			setRootForChildren(allNodes, rootOne, nodeTwo);
			allNodes[nodeTwo].parent = nodeOne;
			allNodes[nodeTwo].weightToParent = edgeWeight;
			allNodes[nodeOne].children.push_back(nodeTwo);
			setRankAbove(allNodes, nodeTwo, allNodes[nodeTwo].rank);
		}
		// If ranks are same, then make one as root and 
		// increment its rank by one
		else
		{

			if (allNodes[nodeTwo].root != nodeTwo) { // check subtree is in correct order
				reverseSubTree(allNodes, nodeTwo, allNodes[allNodes[nodeTwo].root].rank, nodeTwo, nodeTwo);
				rootTwo = nodeTwo;
			}

			allNodes[rootTwo].root = rootOne;
			setRootForChildren(allNodes, rootOne, nodeTwo);
			allNodes[nodeTwo].parent = nodeOne;
			allNodes[nodeTwo].weightToParent = edgeWeight;
			allNodes[nodeOne].children.push_back(nodeTwo);
			allNodes[rootOne].rank++;
			setRankAbove(allNodes, nodeTwo, allNodes[nodeTwo].rank);
		}
	}

	void setRootForChildren(vector<nodeStructure> &allNodes,int rootToSet, int nodeToChangeRootBelow) {
		for (int i = 0; i < allNodes[nodeToChangeRootBelow].children.size();i++) {
			allNodes[allNodes[nodeToChangeRootBelow].children[i]].root = rootToSet;
			setRootForChildren(allNodes,rootToSet, allNodes[nodeToChangeRootBelow].children[i]);
		}
	}

	void setRankAbove(vector<nodeStructure> &allNodes, int index,int rankToSet) {
		if (rankToSet>=allNodes[index].rank) {
			allNodes[index].rank = rankToSet;
			if (allNodes[index].root != index) {
				setRankAbove(allNodes, allNodes[index].parent, rankToSet + 1);
			}			
		}
	}

	void reverseSubTree(vector<nodeStructure> &allNodes, int index, int rankToSet, int parentToBe, int root) {
		if (allNodes[index].root != index) {
			allNodes[index].children.push_back(allNodes[index].parent);

			findAndRemoveChild(allNodes,index, allNodes[index].parent);

			reverseSubTree(allNodes, allNodes[index].parent, rankToSet - 1, index, root);
		}
		else {
			findAndRemoveChild(allNodes, index, allNodes[index].parent);
		}

		allNodes[index].parent = parentToBe;
		allNodes[index].weightToParent = allNodes[parentToBe].weightToParent;

		if (index == root) {	// set all ranks and roots
			setRankDueToChildren(allNodes,index);
			setRootOfAllChildrenBelow(allNodes,index,root);
		}
	}

	void findAndRemoveChild(vector<nodeStructure> &allNodes, int indexToRemove, int parent) {
		for (int i = 0; i < allNodes[parent].children.size();i++) {
			if (allNodes[parent].children[i]==indexToRemove) {
				allNodes[parent].children.erase(allNodes[parent].children.begin()+i);
				return;
			}
		}
	}

	int setRankDueToChildren(vector<nodeStructure> &allNodes,int index) {
		int topRankOfChildren=0;
		for (int i = 0; i < allNodes[index].children.size();i++) {
			int topRankOfThisChild = setRankDueToChildren(allNodes, allNodes[index].children[i]);
			if (topRankOfThisChild>topRankOfChildren) {
				topRankOfChildren = topRankOfThisChild;
			}
		}

		if (allNodes[index].children.size()>0) {
			allNodes[index].rank = topRankOfChildren + 1;
		}
		else {
			allNodes[index].rank = 0;
		}
		
		return allNodes[index].rank;
	}

	void setRootOfAllChildrenBelow(vector<nodeStructure> &allNodes, int index,int root) {
		for (int i = 0; i < allNodes[index].children.size();i++) {
			setRootOfAllChildrenBelow(allNodes, allNodes[index].children[i],root);
		}
		allNodes[index].root = root;
	}


	vector<nodeStructure> makeMST(int yPixels, int xPixels, vector<TreeEdge> &sortedEdges) {
		vector<MST::nodeStructure> nodeMST(yPixels*xPixels);
		for (int v = 0; v < yPixels*xPixels; v++)
		{
			nodeMST[v].parent = v;
			nodeMST[v].root = v;
			nodeMST[v].rank = 0;
			nodeMST[v].index = v;
			nodeMST[v].xCoord = v % xPixels;
			nodeMST[v].yCoord = v / xPixels;
		}

		int i = 0;
		int sidesPicked = 0;
		while (sidesPicked < yPixels*xPixels - 1)
		{
			// Pick the smallest edge. And increment 
			// the index for next iteration
			TreeEdge smallestEdge = sortedEdges[i];

			int rootOne = nodeMST[smallestEdge.sourceNodeIndex].root;
			int rootTwo = nodeMST[smallestEdge.destinationNodeIndex].root;

			if (rootOne != rootTwo) {
				Union(nodeMST, rootOne, rootTwo, smallestEdge.sourceNodeIndex, smallestEdge.destinationNodeIndex, smallestEdge.weight);
				sidesPicked++;
			}
			i++;
		}
		return nodeMST;
	}


	vector<TreeEdge> shortestSidesOfImageFromVector2d(vector<vector<double>> &inputImage) {
		int yPixels = inputImage.size();
		int xPixels = inputImage[0].size();

		// edge array, vertical and horizontal
		// Note: need 1 less row/column in that dimension
		vector<TreeEdge> imageTreeHorizontalEdgesRows(xPixels - 1);
		vector<TreeEdge> imageTreeVerticalEdgesRows(xPixels);

		vector<vector<TreeEdge>> imageTreeHorizontalEdges(yPixels, imageTreeHorizontalEdgesRows);
		vector<vector<TreeEdge>> imageTreeVerticalEdges(yPixels - 1, imageTreeVerticalEdgesRows);

		// list of lowest weight edges
		vector<TreeEdge> sortedEdges;


		int nodeNumber = 0;
		for (int j = 0; j < yPixels; j++) {	// iterate rows
			for (int i = 0; i < xPixels; i++) {	//iterate columns

												// get pixel intensity at pixel
				double pixel1Intensity = inputImage[j][i];

				// set weight of westEdge
				if (i>0) {
					// get pixel intensity of one to left
					double pixel2Intensity = inputImage[j][i - 1];
					// as defined in MST Yang 2015 paper
					double intensityDifference = (pixel1Intensity - pixel2Intensity)*(pixel1Intensity - pixel2Intensity);
					// set weight and add to edge list
					imageTreeHorizontalEdges[j][i - 1].weight = intensityDifference;
					// set the nodedestination
					imageTreeHorizontalEdges[j][i - 1].destinationNodeIndex = nodeNumber;
					imageTreeHorizontalEdges[j][i - 1].sourceNodeIndex = nodeNumber - 1;

					// add to edge list
					sortedEdges.push_back(imageTreeHorizontalEdges[j][i - 1]);
				}


				// same for northEdge
				if (j>0) {
					double pixel2Intensity = inputImage[j - 1][i];
					double intensityDifference = (pixel1Intensity - pixel2Intensity)*(pixel1Intensity - pixel2Intensity);
					imageTreeVerticalEdges[j - 1][i].weight = intensityDifference;
					imageTreeVerticalEdges[j - 1][i].destinationNodeIndex = nodeNumber;
					imageTreeVerticalEdges[j - 1][i].sourceNodeIndex = nodeNumber - xPixels;
					sortedEdges.push_back(imageTreeVerticalEdges[j - 1][i]);
				}

				nodeNumber++;
			}
		}

		return sortedEdges;

	}

	vector<vector<double>> imageLocallyFiltered(int inputRows, int inputCols, vector<vector<double>> &inputImage, vector<nodeStructure> &nodeMST, float sigma) {
		vector<vector<double>> stepOneLocal = forwardsAggregation(inputRows, inputCols, inputImage, nodeMST, sigma);
		return backwardsAggregation(inputRows, inputCols, stepOneLocal, nodeMST, sigma);
	}

	vector<vector<double>> forwardsAggregation(int inputRows, int inputCols, vector<vector<double>> &inputImage, vector<nodeStructure> &nodeMST,float sigma) {
		
		// create 2D array of local costs
		vector<double> localOneDim(inputCols);
		vector<vector<double>> localCostsLeafToRoot(inputRows, localOneDim);

		float factor = exp(-(0.5f / (sigma*sigma)));
		aggregateStepOne(nodeMST, nodeMST[0].root, inputImage, localCostsLeafToRoot, (double)factor);

		return localCostsLeafToRoot;
	}

	void aggregateStepOne(vector<nodeStructure> &nodeMST,int parentIndex, vector<vector<double>> &inputImage, vector<vector<double>> &localCostsLeafToRoot, double factor) {

		for (int i = 0; i < nodeMST[parentIndex].children.size(); i++) {
			aggregateStepOne(nodeMST, nodeMST[parentIndex].children[i],inputImage,localCostsLeafToRoot,factor);
			double childSimilarity = localCostsLeafToRoot[nodeMST[nodeMST[parentIndex].children[i]].yCoord][nodeMST[nodeMST[parentIndex].children[i]].xCoord] * (double)factor;
			localCostsLeafToRoot[nodeMST[parentIndex].yCoord][nodeMST[parentIndex].xCoord] += childSimilarity;
		}
		// add own intensity 
		double pixelintensity = inputImage[nodeMST[parentIndex].yCoord][nodeMST[parentIndex].xCoord];
		localCostsLeafToRoot[nodeMST[parentIndex].yCoord][nodeMST[parentIndex].xCoord] += pixelintensity;
	}

	vector<vector<double>> backwardsAggregation(int inputRows, int inputCols, vector<vector<double>> &localCostsStepOne, vector<nodeStructure> &nodeMST,float sigma) {
		// create 2D array of local costs step 2
		vector<double> localOneDim(inputCols);
		vector<vector<double>> localCostsRootToLeaf(inputRows, localOneDim);

		// factors defined in paper
		double nodeFactor = (1 - exp(-1 / (sigma*sigma)));
		double parentFactor = exp((double)(-0.5) / (sigma*sigma));

		aggregateStepTwo(nodeMST, nodeMST[0].root, localCostsStepOne, localCostsRootToLeaf,parentFactor,nodeFactor);

		return localCostsRootToLeaf;
	}

	void aggregateStepTwo(vector<nodeStructure> &nodeMST, int parentIndex, vector<vector<double>> &localCostsStepOne, vector<vector<double>> &localCostsLeafToRoot, double parentFactor, double nodeFactor) {
		// add value from step 1
		localCostsLeafToRoot[nodeMST[parentIndex].yCoord][nodeMST[parentIndex].xCoord] += (nodeFactor*localCostsStepOne[nodeMST[parentIndex].yCoord][nodeMST[parentIndex].xCoord]);
		
		// add value from parent
		if (parentIndex != nodeMST[parentIndex].parent) {
			localCostsLeafToRoot[nodeMST[parentIndex].yCoord][nodeMST[parentIndex].xCoord] += localCostsLeafToRoot[nodeMST[nodeMST[parentIndex].parent].yCoord][nodeMST[nodeMST[parentIndex].parent].xCoord] *parentFactor;
		}
		// call same function for all children
		for (int i = 0; i < nodeMST[parentIndex].children.size(); i++) {
			aggregateStepTwo(nodeMST, nodeMST[parentIndex].children[i], localCostsStepOne, localCostsLeafToRoot, parentFactor, nodeFactor);
		}
	}

	// non local
	vector<vector<double>> imageNonLocalFilter(int inputRows, int inputCols, vector<nodeStructure> &nodeMST, int squareDimension, vector<vector<double>> &inputImage) {
		
		vector<double> temp(inputCols,0);
		vector<vector<double>> nonLocalFilter(inputRows,temp);
		int xmax = inputCols - 1;
		int ymax = inputRows - 1;

		// create an entry for every pixel
		for (int j = 0; j < ymax+1; j++) {
			for (int i = 0; i < xmax+1; i++) {
				nonLocalFilter[j][i] = nonLocalAtPixel(ymax, xmax,j,i,nodeMST, squareDimension,inputImage);
			}
		}
		return nonLocalFilter;
	}

	// non local value at each pixel
	double nonLocalAtPixel(int ymax, int xmax, int y, int x , vector<nodeStructure> &nodeMST, int squareDimension, vector<vector<double>> &inputImage) {

		vector<double> nodeWeights(9,0);
		vector<double> nodeIntensities(9,0);

		bool allZeroWeights = true;
		int numberEitherside = (squareDimension - 1) / 2;
		int index = 0;
		for (int j = y - numberEitherside; j < y + numberEitherside + 1; j++) {
			for (int i = x - numberEitherside; i < x + numberEitherside + 1; i++) {

				// out of range or the centre pixel
				if (j<0 || i<0 || j>ymax || i>xmax || (j == y && i == x)) {
					index++;
					continue;
				}
				else {
					int centreNodeIndex = y*(xmax+1) + x;
					int thisNodeIndex = j*(xmax+1) + i;

					// add to intensity list
					nodeIntensities[index] = inputImage[j][i];
					// find weight from p to q
					float weight = findWeight(nodeMST, thisNodeIndex, centreNodeIndex);
					if (weight!=0 && allZeroWeights) {
						allZeroWeights = false;
					}
					nodeWeights[index] = (weight);
					index++;
				}
			}
		}
		

		// find min b
		int minb = -1;
		int bCost = -1;

		if (allZeroWeights) {
			return 0;
		}
		else {
			// iteratate all b values 
			for (int i = 0; i < nodeWeights.size(); i++) {
				if (nodeWeights[i]==0) {
					continue;
				}
				double thisbCost = nonLocalWithb(nodeIntensities[i], nodeIntensities, nodeWeights);

				if (bCost<0 || thisbCost<bCost) {
					bCost = thisbCost;
					minb = nodeIntensities[i];
				}
			}
		}
		return minb;
	}


	double findWeight(vector<nodeStructure> &nodeMST, int node1, int node2) {

		float weight = 0;

		while (nodeMST[node1].rank != nodeMST[node2].rank) {
			if (nodeMST[node1].rank < nodeMST[node2].rank) {
				weight += nodeMST[node1].weightToParent;
				node1 = nodeMST[node1].parent;
			}
			else {
				weight += nodeMST[node2].weightToParent;
				node2 = nodeMST[node2].parent;
			}
		}

		weight += findWeightToConnectedParent(nodeMST, node1, node2, 0);

		// ranks are the same
		return weight;
	}

	double findWeightToConnectedParent(vector<nodeStructure> &nodeMST, int node1, int node2, int iteration) {
		if (node1 == node2) {
			return 0;
		}

		if (nodeMST[node1].rank > nodeMST[node2].rank) {
			return nodeMST[node2].weightToParent + findWeightToConnectedParent(nodeMST, node1, nodeMST[node2].parent, iteration + 1);
		}
		else if (nodeMST[node1].rank < nodeMST[node2].rank) {
			return nodeMST[node1].weightToParent + findWeightToConnectedParent(nodeMST, nodeMST[node1].parent, node2, iteration + 1);
		}
		else {
			return nodeMST[node1].weightToParent + nodeMST[node2].weightToParent + findWeightToConnectedParent(nodeMST, nodeMST[node1].parent, nodeMST[node2].parent, iteration + 1);
		}
	}

	double nonLocalWithb(int b, vector<double> &nodeIntensities, vector<double> &nodeWeights) {

		double cost=0;

		for (int i = 0; i < nodeIntensities.size();i++){//nodeIntensities.size();i++) {
			cost += nodeWeights[i]*abs(b-nodeIntensities[i]);
		}

		return cost;
	}


	vector<vector<double>> productOfImages(Mat &inputImage1, Mat &inputImage2, int disparity, bool diparityForBoth,bool squareRoot) {
		// Note: reference frame is image1

		vector<double> product1d(inputImage1.cols,0);
		vector<vector<double>> product(inputImage1.rows,product1d);
		
		for (int j = 0; j < inputImage1.rows;j++) {
			for (int i = 0; i < inputImage1.cols;i++) {
				// value doesn't exhist (zero)
				if ((i-disparity)<0 || (i - disparity)>inputImage1.cols-1) {
					continue;
				}

				Scalar pixel1Intensity;
				
				if (diparityForBoth) {
					pixel1Intensity = (double)inputImage1.at<uchar>(j, i- disparity);
				}
				else {
					pixel1Intensity = (double)inputImage1.at<uchar>(j, i);
				}
				
				Scalar pixel2Intensity = (double)inputImage2.at<uchar>(j, i - disparity);

				if (squareRoot) {
					product[j][i] = (double)*pixel1Intensity.val;
					//outputImage.at<uchar>(j, i) = (double)*pixel1Intensity.val;
				}
				else {
					product[j][i] = (double)*pixel1Intensity.val * (double)*pixel2Intensity.val;
					//outputImage.at<uchar>(j, i) = (double)*pixel1Intensity.val * (double)*pixel2Intensity.val;
				}
				
			}
		}

		return product;
	}


	vector<vector<double>> matToVector2d(Mat &inputImage) {
		vector<double> output1d(inputImage.cols, 0);
		vector<vector<double>> output(inputImage.rows, output1d);

		for (int j = 0; j < output.size(); j++) {
			for (int i = 0; i < output[0].size(); i++) {
				Scalar intensity = inputImage.at<uchar>(j, i);
				output[j][i] = (double)*intensity.val;
			}
		}

		return output;
	}

	
	vector<vector<double>> findDisparities(Mat &imgLeft, Mat &imgRight, int a_p, int b_p, int disparityRange, int squareDimension,float sigma) {

		// local Cost matrix
		vector<double> localCosts1D(disparityRange, -1);
		vector<vector<double>> localCosts2D(imgLeft.cols, localCosts1D);
		vector<vector<vector<double>>> localCosts(imgLeft.rows, localCosts2D);
		
		// non local Cost matrix
		vector<double> nonLocalCosts1D(disparityRange, -1);
		vector<vector<double>> nonLocalCosts2D(imgLeft.cols, nonLocalCosts1D);
		vector<vector<vector<double>>> nonLocalCosts(imgLeft.rows, nonLocalCosts2D);

		// left squared image doesn't change with d
		vector<vector<double>> leftSquared = productOfImages(imgLeft,imgLeft,0,false,false);
		vector<nodeStructure> nodeMSTLeftSquared = vector2DToMST(leftSquared);
		vector<vector<double>> localFilterLeftSquared = imageLocallyFiltered(leftSquared.size(), leftSquared[0].size(),leftSquared,nodeMSTLeftSquared,sigma);
		vector<vector<double>> nonLocalFilterLeftSquared = imageNonLocalFilter(leftSquared.size(), leftSquared[0].size(), nodeMSTLeftSquared, squareDimension, leftSquared);
		cout << "Left Squared Finished" << endl;


		// left currently b=0 so not needed
		vector<double> localFilterLeft1d(imgLeft.cols, 0);
		vector<vector<double>> localFilterLeft(imgLeft.rows, localFilterLeft1d);
		vector<double> nonLocalFilterLeft1d(imgLeft.cols, 0);
		vector<vector<double>> nonLocalFilterLeft(imgLeft.rows, nonLocalFilterLeft1d);
		if (b_p != 0) {
			vector<vector<double>> leftImage = matToVector2d(imgLeft);
			vector<nodeStructure> nodeMSTLeft = vector2DToMST(leftImage);
			localFilterLeft = imageLocallyFiltered(leftImage.size(), leftImage[0].size(), leftImage, nodeMSTLeft, sigma);
			nonLocalFilterLeft = imageNonLocalFilter(leftImage.size(), leftImage[0].size(), nodeMSTLeft, squareDimension, leftImage);
		}
		cout << "Left Finished" << endl;
		

		// iterate all possible disparities
		for (int d = 0; d < disparityRange;d++) {
			//cout << "q = "<<q<< " and d = "<< d<<endl;
			cout << "d = " << d << endl;
			clock_t t;
			t = clock();

			// rightDelta squared
			vector<vector<double>> rightDeltaSquared = productOfImages(imgRight, imgRight,d,true,false);
			vector<nodeStructure> nodeMSTRightDeltaSquared = vector2DToMST(rightDeltaSquared);
			vector<vector<double>> localFilterRightDeltaSquared = imageLocallyFiltered(rightDeltaSquared.size(), rightDeltaSquared[0].size(), rightDeltaSquared, nodeMSTRightDeltaSquared, sigma);
			vector<vector<double>> nonLocalFilterRightDeltaSquared = imageNonLocalFilter(rightDeltaSquared.size(), rightDeltaSquared[0].size(), nodeMSTRightDeltaSquared, squareDimension, rightDeltaSquared);
			cout << "Right Delta Squared Finished" << endl;


			// leftRightDelta
			vector<vector<double>> leftRightDelta = productOfImages(imgLeft, imgRight,d,false,false);
			vector<nodeStructure> nodeMSTLeftRightDelta = vector2DToMST(leftRightDelta);
			vector<vector<double>> localFilterLeftRightDelta = imageLocallyFiltered(leftRightDelta.size(), leftRightDelta[0].size(), leftRightDelta, nodeMSTLeftRightDelta, sigma);
			vector<vector<double>> nonLocalFilterLeftRightDelta = imageNonLocalFilter(leftRightDelta.size(), leftRightDelta[0].size(), nodeMSTLeftRightDelta, squareDimension, leftRightDelta);
			cout << "Left Right Delta Finished" << endl;

			// rightDelta 
			vector<vector<double>> rightDelta = productOfImages(imgRight, imgRight, d, true, true);
			vector<double> localFilterRightDelta1d(rightDelta[0].size(), 0);
			vector<vector<double>> localFilterRightDelta(rightDelta.size(), localFilterRightDelta1d);
			vector<double> nonLocalFilterRightDelta1d(rightDelta[0].size(), 0);
			vector<vector<double>> nonLocalFilterRightDelta(rightDelta.size(), nonLocalFilterRightDelta1d);
			// currently b=0 so not needed
			if (b_p != 0) {
				vector<nodeStructure> nodeMSTRightDelta = vector2DToMST(rightDelta);
				localFilterRightDelta = imageLocallyFiltered(rightDelta.size(), rightDelta[0].size(), rightDelta, nodeMSTRightDelta, sigma);
				nonLocalFilterRightDelta = imageNonLocalFilter(rightDelta.size(), rightDelta[0].size(), nodeMSTRightDelta, squareDimension, rightDelta);
			}
			cout << "Right Delta Finished" << endl;
			
			// local costs
			costFunction(localFilterLeftSquared, localFilterLeft, localFilterRightDeltaSquared, localFilterLeftRightDelta, localFilterRightDelta,localCosts,a_p,b_p,d);
			// non local costs
			costFunction(nonLocalFilterLeftSquared, nonLocalFilterLeft, nonLocalFilterRightDeltaSquared, nonLocalFilterLeftRightDelta, nonLocalFilterRightDelta, nonLocalCosts, a_p, b_p,d);
			//q++;

			t = clock() - t;
			cout << "Time: " << ((float)t) / CLOCKS_PER_SEC << endl;
			cout << "Disparity "<< d << " out of " << disparityRange -1 << endl;	
			cout << endl;
		}
		// add two costs together
		vector<vector<vector<double>>> totalCosts = addCostMatrices(localCosts, nonLocalCosts, true);

		// takes absolute with the true value;
	return disparityMatrix(totalCosts,disparityRange,false);

	}

	vector<nodeStructure> vector2DToMST(vector<vector<double>> inputImage) {
		vector<TreeEdge> sortedEdges(shortestSidesOfImageFromVector2d(inputImage));

		sort(sortedEdges.begin(), sortedEdges.end());
		return makeMST(inputImage.size(), inputImage[0].size(), sortedEdges);
	}


	void costFunction(vector<vector<double>> &filterLeftSquared, vector<vector<double>> &filterLeft, vector<vector<double>> &filterRightDeltaSquared, vector<vector<double>> &filterLeftRightDelta, vector<vector<double>> &filterRightDelta, vector<vector<vector<double>>> &costMatrix,int a_p,int b_p, int d) {

		for (int j = 0; j < filterLeftSquared.size();j++) {
			for (int i = 0; i < filterLeftSquared[0].size(); i++) {
				costMatrix[j][i][d] = filterLeftSquared[j][i] + (a_p * filterRightDeltaSquared[j][i]) + (b_p * b_p) - (2 * b_p * filterLeft[j][i]) - (2 * a_p * filterLeftRightDelta[j][i]) + (2 * a_p * b_p * filterRightDelta[j][i]);
			}
		}
	}


	vector<vector<vector<double>>> addCostMatrices(vector<vector<vector<double>>> &cost1, vector<vector<vector<double>>> &cost2, bool takeAbsolute) {

		vector<double> result1D(cost1[0][0].size(), -1);
		vector<vector<double>> result2D(cost1[0].size(), result1D);
		vector<vector<vector<double>>> result3D(cost1.size(), result2D);

		for (int j = 0; j < cost1.size(); j++) {
			for (int i = 0; i < cost1[0].size(); i++) {
				for (int q = 0; q < cost1[0][0].size(); q++) {
					if (takeAbsolute) {
						result3D[j][i][q] = abs(cost1[j][i][q] + cost2[j][i][q]);
					}
					else {
						result3D[j][i][q] = cost1[j][i][q] + cost2[j][i][q];
					}
				}
			}
		}

		return result3D;
	}


	vector<vector<double>> disparityMatrix(vector<vector<vector<double>>> &costsMatrix, int disparityRange,bool absoluteOfDisparity) {
		//cout<< "Entered Here "<< endl;
		vector<double> disparity1D(costsMatrix[0].size(), -1);
		vector<vector<double>> disparity2D(costsMatrix.size(), disparity1D);

		for (int j = 0; j < costsMatrix.size(); j++) {
			for (int i = 0; i < costsMatrix[0].size();i++) {
				float minCost = -9999;
				int minD = -1;
				for (int q = 0; q < costsMatrix[0][0].size(); q++) {
					double thisCost = costsMatrix[j][i][q];
					if (minCost == -9999) {
						minCost = thisCost;
						minD = q;
					}
					else if (thisCost<minCost) {
						minCost = thisCost;
						minD = q;
					}
					else if (thisCost==minCost) {
						if (abs(q)<abs(minD)) {
							minD = q;
						}
					}
					
				}

				if (absoluteOfDisparity) {
					disparity2D[j][i] = abs(minD);
				}
				else {
					disparity2D[j][i] = minD;
				}
			}
		}

		return disparity2D;
	}

	Mat disparityImage(vector<vector<double>> &disparityMatrix, int minD) {
		Mat outputImage(disparityMatrix.size(), disparityMatrix[0].size(), CV_8UC1);

		double min=9999;
		double max=-9999;

		for (int j = 0; j < disparityMatrix.size(); j++) {
			for (int i = 0; i < disparityMatrix[0].size(); i++) {
				if (disparityMatrix[j][i] + minD>max) {
					max = disparityMatrix[j][i] + minD;
				}
				else if (disparityMatrix[j][i] + minD<min) {
					min = disparityMatrix[j][i] + minD;
				}
			}
		}

		cout<< "Min: "<< min<< endl;
		cout << "Max: " << max << endl;

		scaleMatrix(disparityMatrix, min,max, false);

		for (int j = 0; j < disparityMatrix.size(); j++) {
			for (int i = 0; i < disparityMatrix[0].size(); i++) {
				outputImage.at<uchar>(j, i) = disparityMatrix[j][i] + minD;
			}
		}		

		return outputImage;
	}

	void scaleMatrix(vector<vector<double>> &matrix, int min, int max, bool depth) {

		if (max-min==0) {
			return;
		}

		for (int j = 0; j < matrix.size(); j++) {
			for (int i = 0; i < matrix[0].size(); i++) {
				matrix[j][i] = (matrix[j][i] - (double)min)*((double)255 / ((double)max - (double)min));
			}
		}
	}






	// temp /////////////////////////////////////////////

	Mat calculateDepthMat(vector<vector<double>> disparityMatrix, double baseLineDistance, double focalLength) {
		vector<vector<double>> depthMatrix = calculateDepth(disparityMatrix, baseLineDistance, focalLength);
		Mat depth = depthImage(depthMatrix);
		//reverseImage(depth);

		return depth;
	}

	vector<vector<double>> calculateDepth(vector<vector<double>> &disparityMatrix, double baselineDistance, double focalLength) {
		vector<double> depth1D(disparityMatrix[0].size(), -1);
		vector<vector<double>> depthMatrix(disparityMatrix.size(), depth1D);

		double min = 9999;
		double max = -9999;

		for (int j = 0; j < depthMatrix.size();j++) {
			for (int i = 0; i < depthMatrix[0].size();i++) {
				if (disparityMatrix[j][i]==0) {
					depthMatrix[j][i] = -1;
				}
				else {
					depthMatrix[j][i] = ((double)focalLength *((double)baselineDistance / abs((double)disparityMatrix[j][i])));
				}

				// find max and min
				if (depthMatrix[j][i] > max) {
					max = depthMatrix[j][i];
				}
				else if (depthMatrix[j][i] <min) {
					min = depthMatrix[j][i];
					if (min == -1) {
						cout << min << " at " << j << "," << i << endl;
					}
				}
			}
		}

		// set 0's to max depth
		for (int j = 0; j < depthMatrix.size(); j++) {
			for (int i = 0; i < depthMatrix[0].size(); i++) {
				if (disparityMatrix[j][i] == -1) {
					depthMatrix[j][i] = max;
				}
			}
		}

		cout << "Min Depth: " << min << endl;
		cout << "Max Depth: " << max << endl;

		scaleMatrix(depthMatrix, min, max,true);

		return depthMatrix;
	}

	Mat depthImage(vector<vector<double>> &depthMatrix) {
		Mat depth(depthMatrix.size(), depthMatrix[0].size(), CV_8UC1);

		for (int j = 0; j < depthMatrix.size(); j++) {
			for (int i = 0; i < depthMatrix[0].size(); i++) {
				depth.at<uchar>(j, i) = depthMatrix[j][i];
			}
		}
		return depth;
	}

	void reverseImage(Mat &image) {
		for (int j = 0; j < image.rows;j++) {
			for (int i = 0; i < image.cols; i++) {
				Scalar intensity = image.at<uchar>(j, i);
				image.at<uchar>(j, i) = 255 - (double)*intensity.val;
			}
		}
	}


	
	Mat differenceBetweenTwoImages(Mat &img1, Mat &img2) {
		Mat output(img1.rows,img1.cols, CV_8UC1);

		for (int j = 0; j < output.rows; j++) {
			for (int i = 0; i < output.cols; i++) {
				Scalar intensity1 = img1.at<uchar>(j, i);
				Scalar intensity2 = img2.at<uchar>(j, i);
				output.at<uchar>(j, i) = abs((double)*intensity1.val- (double)*intensity2.val);
			}
		}

		return output;
	}



	void print2DVector(vector<vector<double>> input) {
		cout << endl;
		for (int j = 0; j < input.size(); j++) {
			for (int i = 0; i < input[0].size(); i++) {
				cout << input[j][i];
				if (i!= input[0].size()-1) {
					cout<< ", ";
				}
			}
			cout << endl;
		}
		cout << endl << endl;
	}

	void printAllCostMatrices(vector<vector<vector<double>>> &costs) {


		vector<double> costsTemp(costs[0].size(),-1);
		vector<vector<double>> costs2d(costs.size(),costsTemp);

		for (int q = 0; q < costs[0][0].size(); q++) {
			for (int j = 0; j < costs.size(); j++) {
				for (int i = 0; i < costs[0].size(); i++) {
					costs2d[j][i] = costs[j][i][q];
				}
			}

			print2DVector(costs2d);
		}

	}

};