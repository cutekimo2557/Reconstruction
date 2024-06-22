#pragma once
#ifndef SINOGRAM_H
#define SINOGRAM_H

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;

Mat radonTransform(const Mat& img, const std::vector<double>& theta);

#endif
