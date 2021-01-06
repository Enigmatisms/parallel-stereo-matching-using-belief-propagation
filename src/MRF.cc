#include "../include/MRF.hpp"

// float weights[25] = {1, 2, 3, 2, 1,
// 					2, 5, 6, 5, 2, 
// 					3, 6, 8, 6, 3,
// 					2, 5, 6, 5, 2, 
// 					1, 2, 3, 2, 1};

float weights[25] = {1, 1, 1, 1, 1,
					1, 1, 1, 1, 1, 
					1, 1, 1, 1, 1,
					1, 1, 1, 1, 1, 
					1, 1, 1, 1, 1};

const data_cost_t knorm = 25;

void initializeMarkovRandomField(MarkovRandomField& mrf, const std::string leftImgPath, const std::string rightImgPath,
                                 const MarkovRandomFieldParam param) {
	cv::Mat leftImg = cv::imread(leftImgPath, 0);
	cv::Mat rightImg = cv::imread(rightImgPath, 0);
	
	printf("Left image is empty? %d\n", leftImg.empty());
	printf("Right image is empty? %d\n", rightImg.empty());

	mrf.height = leftImg.rows;
	mrf.width = leftImg.cols;
	mrf.grid.resize(mrf.height * mrf.width);
	mrf.param = param;

	for (int pos = 0; pos < mrf.height * mrf.width; pos++) {
		mrf.grid[pos].leftMessage = new msg_t[mrf.param.maxDisparity];
		mrf.grid[pos].rightMessage = new msg_t[mrf.param.maxDisparity];
		mrf.grid[pos].upMessage = new msg_t[mrf.param.maxDisparity];
		mrf.grid[pos].downMessage = new msg_t[mrf.param.maxDisparity];
		mrf.grid[pos].dataMessage = new msg_t[mrf.param.maxDisparity];

		for (int idx = 0; idx < mrf.param.maxDisparity; idx++) {
			mrf.grid[pos].leftMessage[idx] = 0;
			mrf.grid[pos].rightMessage[idx] = 0;
			mrf.grid[pos].upMessage[idx] = 0;
			mrf.grid[pos].downMessage[idx] = 0;
			mrf.grid[pos].dataMessage[idx] = 0;
		}
	}

	const int border = mrf.param.maxDisparity;

	// 初始化光度误差计算 可以加个权的
	for (int y = border; y < mrf.height - border; y++) {
		for (int x = border; x < mrf.width - border; x++) {
			for (int i = 0; i < mrf.param.maxDisparity; i++) {
				mrf.grid[y * mrf.width + x].dataMessage[i] = calculateDataCost(leftImg, rightImg, x, y, i);
			}
		}
	}
}

// 原来没有delete
void destroyMRF(MarkovRandomField& mrf){
	for (int pos = 0; pos < mrf.height * mrf.width; pos++) {
		delete [] mrf.grid[pos].leftMessage; 
		delete [] mrf.grid[pos].rightMessage;
		delete [] mrf.grid[pos].upMessage;
		delete [] mrf.grid[pos].downMessage;
		delete [] mrf.grid[pos].dataMessage;
	}
}

void sendMsg(MarkovRandomField& mrf, const int x, const int y, const Direction dir) {
	const int disp = mrf.param.maxDisparity;
	const int w = mrf.width;
	const int now_pos = y * w + x;
	int index = 0;
	msg_t norm = 0.0;
	msg_t* msgs = new msg_t[disp];
	switch (dir){
		case Left:
			index = y * w + x - 1;
			break;
		case Right:
			index = y * w + x + 1;
			break;
		case Up:
			index = (y - 1) * w + x;
			break;
		default:
			index = (y + 1) * w + x;
			break;
	}

	for (int i = 0; i < disp; i++) {
		msg_t minVal = UINT_MAX;
		for (int j = 0; j < disp; j++) {
			// message 计算，由论文中的max F(x) 变为 min log(F(x))，省去了负指数操作
			// 公式为 ：κ max ψ st (xs , xt)ms (x s ) * \prod{m_{ks}}

			// 1. 此处是 ψ st (xs , xt) 直接使用了 truncated |x|函数 
			msg_t p = calculateSmoothnessCost(i, j, mrf.param.lambda, mrf.param.smoothnessParam);

			// 2. ms (x s )的计算
			p += mrf.grid[now_pos].dataMessage[j];		// 当前位置 视差为j的光度误差

			// 3. 每个节点的N(s)节点消息计算
			if (dir != Left) p += mrf.grid[now_pos].leftMessage[j];
			if (dir != Right) p += mrf.grid[now_pos].rightMessage[j];
			if (dir != Up) p += mrf.grid[now_pos].upMessage[j];
			if (dir != Down) p += mrf.grid[now_pos].downMessage[j];

			minVal = std::min(minVal, p);
		}
		norm += minVal * minVal;
		msgs[i] = minVal;
		// switch (dir) {
		// 	case Left:
		// 		mrf.grid[index].leftMessage[i] = minVal;
		// 		break;
		// 	case Right:
		// 		mrf.grid[index].rightMessage[i] = minVal;		// 原来这里写的是 y * w + x - 1
		// 		break;
		// 	case Up:
		// 		mrf.grid[index].upMessage[i] = minVal;
		// 		break;
		// 	default:
		// 		mrf.grid[index].downMessage[i] = minVal;
		// 		break;
		// }
	}
	norm = sqrt(norm) / 5;											// 矩阵范数，进行消息大小归一化
	for (int i = 0; i < disp; i++){
		switch (dir) {
			case Left:
				mrf.grid[index].leftMessage[i] = msgs[i] / norm;
				break;
			case Right:
				mrf.grid[index].rightMessage[i] = msgs[i] / norm;		// 原来这里写的是 y * w + x - 1
				break;
			case Up:
				mrf.grid[index].upMessage[i] = msgs[i] / norm;
				break;
			default:
				mrf.grid[index].downMessage[i] = msgs[i] / norm;
				break;
		}
	}
	delete [] msgs;
}

void beliefPropagation(MarkovRandomField& mrf, const Direction dir) {
	const int w = mrf.width;
	const int h = mrf.height;

	// 压缩映射
	switch (dir) {
		case Left:
			for (int y = 0; y < h; y++) {					// 不管哪一行，都不应该与上一行行末有关
				for (int x = w - 1; x > 0; x--) {			/// when y = 0, x can't be 0 (index x - 1)
					sendMsg(mrf, x, y, Left);
				}
			}
			break;
		case Right:
			for (int y = 0; y < h; y++) {				// 原来的写法会直接越界		
				for (int x = 0 ; x < w - 1; x++) {
					sendMsg(mrf, x, y, Right);
				}
			}
			break;
		case Up:
			for (int x = 0; x < w; x++) {
				for (int y = h - 1; y > 0; y--) {
					sendMsg(mrf, x, y, Up);
				}
			}
			break;
		case Down:
			for (int x = 0; x < w; x++) {
				for (int y = 0; y < h - 1; y++) {
					sendMsg(mrf, x, y, Down);
				}
			}
			break;
		default:
			break;
	}
}

energy_t calculateMaxPosteriorProbability(MarkovRandomField& mrf, int iter) {
	for (int i = 0; i < mrf.grid.size(); i++) {
		double best = 1e40;
		for (int j = 0; j < mrf.param.maxDisparity; j++) {
			double cost = 0;

			cost += mrf.grid[i].leftMessage[j];
			cost += mrf.grid[i].rightMessage[j];
			cost += mrf.grid[i].upMessage[j];
			cost += mrf.grid[i].downMessage[j];
			cost += mrf.grid[i].dataMessage[j];
			if (cost < best) {
				best = cost;
				mrf.grid[i].bestAssignmentIndex = j;
			}
		}
	}

	const int w = mrf.width;
	const int h = mrf.height;

	energy_t energy = 0;
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			const int pos = y * mrf.width + x;
			const int bestAssignmentIndex = mrf.grid[y * mrf.width + x].bestAssignmentIndex;

			energy += mrf.grid[pos].dataMessage[bestAssignmentIndex];

			if (x >= 1) {
				energy += calculateSmoothnessCost(bestAssignmentIndex, mrf.grid[y * mrf.width + x - 1].bestAssignmentIndex,
				                                  mrf.param.lambda, mrf.param.smoothnessParam);
			}

			if (x < w - 1) {
				energy += calculateSmoothnessCost(bestAssignmentIndex, mrf.grid[y * mrf.width + x + 1].bestAssignmentIndex,
				                                  mrf.param.lambda, mrf.param.smoothnessParam);
			}

			if (y >= 1) {
				energy += calculateSmoothnessCost(bestAssignmentIndex, mrf.grid[(y - 1) * mrf.width + x].bestAssignmentIndex,
				                                  mrf.param.lambda, mrf.param.smoothnessParam);
			}

			if (y < h - 1) {
				energy += calculateSmoothnessCost(bestAssignmentIndex, mrf.grid[(y + 1) * mrf.width + x].bestAssignmentIndex,
				                                  mrf.param.lambda, mrf.param.smoothnessParam);
			}
		}
	}

	return energy;
}

// 光度误差计算
data_cost_t calculateDataCost(const cv::Mat& leftImg, const cv::Mat& rightImg, const int x, const int y, const int disparity) {
	const int radius = 2;
	int cnt = 0;
	int sum = 0;

	uchar* left = leftImg.data;
	uchar* right = rightImg.data;
	for (int dy = -radius; dy <= radius; dy++) {
		for (int dx = -radius; dx <= radius; dx++) {
			const int l = left[(y + dy) * leftImg.step + x + dx];
			const int r = right[(y + dy) * rightImg.step + x + dx - disparity];
			sum += abs(l - r) * weights[cnt];
			cnt += 1;
		}
	}

	const data_cost_t avg = sum / knorm;
	if (avg <= 1){
		return 0;
	}
	return log(avg);
}

inline smoothness_cost_t calculateSmoothnessCost(const int i, const int j, const int lambda, const int smoothnessParam) {
	const int d = i - j;
	return lambda * std::min(abs(d), smoothnessParam);
}
