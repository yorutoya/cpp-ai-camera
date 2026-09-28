#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <Detection.hpp>

#include<string>
#include<vector>

class YoloDetector {
public:
    explicit YoloDetector(const std::string& modelPath);
    std::vector<Detection> detect(const cv::Mat& frame);
    double getInferenceTime() const;

private:
    cv::dnn::Net net;
    float confidenceThreshold = 0.5f;
    float nmsThreshold = 0.4f;
    std::vector<std::string> classNames;
    double inferenceTimeMs = 0.0;

    cv::Mat letterbox(
        const cv::Mat& image,
        float& scale,
        int& padX,
        int& padY
    );
};