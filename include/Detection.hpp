#pragma once

#include <string>
#include <opencv2/core/types.hpp>

struct Detection {
    int classId = -1;
    std::string className;
    float confidence = 0.0f;
    cv::Rect box;
};