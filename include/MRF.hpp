#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

enum Direction { Left, Right, Up, Down, Data };

typedef double msg_t;
typedef unsigned int energy_t;
typedef unsigned int smoothness_cost_t;
typedef unsigned int data_cost_t;

struct MarkovRandomFieldNode {
	msg_t* leftMessage;
	msg_t* rightMessage;
	msg_t* upMessage;
	msg_t* downMessage;
	msg_t* dataMessage;
	int bestAssignmentIndex;
};

struct MarkovRandomFieldParam {
	int maxDisparity, lambda, iteration, smoothnessParam;
};

struct MarkovRandomField {
	std::vector<MarkovRandomFieldNode> grid;
	MarkovRandomFieldParam param;
	int height, width;
};

void destroyMRF(MarkovRandomField& mrf);
void initializeMarkovRandomField(MarkovRandomField& mrf, std::string leftImgPath, std::string rightImgPath, MarkovRandomFieldParam param);
void sendMsg(MarkovRandomField& mrf, int x, int y, Direction dir);
void beliefPropagation(MarkovRandomField& mrf, Direction dir);
energy_t calculateMaxPosteriorProbability(MarkovRandomField& mrf, int iter);

data_cost_t calculateDataCost(const cv::Mat& leftImg, const cv::Mat& rightImg, int x, int y, int disparity);
smoothness_cost_t calculateSmoothnessCost(int i, int j, int lambda, int smoothnessParam);
