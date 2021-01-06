#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "include/MRF.hpp"

const int ITERATION = 20;
const int MAX_DISPARITY = 40;

double countTime() {
	return static_cast<double>(clock());
}

int main(int argc, char* argv[]) {
	int LAMBDA = 5;
	int SMOOTHNESS_PARAM = 3;
	if (argc > 1){
		LAMBDA = atoi(argv[1]);
		SMOOTHNESS_PARAM = atoi(argv[2]);
	}
	const double beginTime = countTime();
	int ascending_cnt = 0;

	MarkovRandomField mrf;
	const std::string leftImgPath = "../test-data/venus/im0.png";
	const std::string rightImgPath = "../test-data/venus/im8.png";
	MarkovRandomFieldParam param;
	energy_t old_energy = UINT32_MAX;

	param.iteration = ITERATION;
	param.lambda = LAMBDA;
	param.maxDisparity = MAX_DISPARITY;
	param.smoothnessParam = SMOOTHNESS_PARAM;

	initializeMarkovRandomField(mrf, leftImgPath, rightImgPath, param);

	for (int i = 0; i < mrf.param.iteration; i++) {
		beliefPropagation(mrf, Left);
		beliefPropagation(mrf, Right);
		beliefPropagation(mrf, Up);
		beliefPropagation(mrf, Down);

		const energy_t energy = calculateMaxPosteriorProbability(mrf, i);

		std::cout << "Iteration: " << i << ";  Energy: " << energy << "." << std::endl;
		if (old_energy < energy){
			ascending_cnt ++;
			if (ascending_cnt > 3){
				std::cout << "Energy starts to ascend again. Exiting...\n";
				break;
			}
		}
		else{
			ascending_cnt = 0;
		}
		old_energy = energy;
	}

	cv::Mat output = cv::Mat::zeros(mrf.height, mrf.width, CV_8U);

	for (int i = 0; i < mrf.height; i++) {
		for (int j = 0; j < mrf.width; j++) {
			output.at<uchar>(i, j) = mrf.grid[i * mrf.width + j].bestAssignmentIndex * (256 / mrf.param.maxDisparity);
		}
	}

	cv::resize(output, output, cv::Size(output.cols * 2, output.rows * 2));

	const double endTime = countTime();

	std::cout << (endTime - beginTime) / CLOCKS_PER_SEC << std::endl;

	cv::imshow("Output", output);
	cv::waitKey();
	std::string opath = "../outs/pic_L" + std::to_string(LAMBDA) + "S" + std::to_string(SMOOTHNESS_PARAM) + ".png";
	cv::imwrite(opath, output);
	destroyMRF(mrf);
	return 0;
}
