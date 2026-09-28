#pragma once

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include "Detection.hpp"

#include <string>
#include <vector>

class YoloDetector {
public:
    explicit YoloDetector(const std::string& modelPath);
    std::vector<Detection> detect(const cv::Mat& frame);
    double getInferenceTime() const;

private:
    cv::dnn::Net net;
    static constexpr int inputSize = 640;
    static constexpr float confidenceThreshold = 0.5f;
    static constexpr float nmsThreshold = 0.4f;
    std::vector<std::string> classNames;
    double inferenceTimeMs = 0.0;

    static cv::Mat letterbox(
        const cv::Mat& image,
        float& scale,
        int& padX,
        int& padY
    );
};