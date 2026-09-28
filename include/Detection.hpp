#pragma once

#include <string>
#include <opencv2/opencv.hpp>

struct Detection {
    int classId;
    std::string className;
    float confidence;
    cv::Rect box;
};