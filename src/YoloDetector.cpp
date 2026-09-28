#include "YoloDetector.hpp"
#include <iostream>
#include <stdexcept>
#include <chrono>

YoloDetector::YoloDetector(const std::string &modelPath) {
    std::cout << "Loading YOLO model..." << std::endl;

    net = cv::dnn::readNetFromONNX(modelPath);
    if (net.empty()) {
        throw std::runtime_error("Failed to load YOLO model.");
    }

    classNames = {
        "person", "bicycle", "car", "motorcycle", "airplane",
        "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird",
        "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut",
        "cake", "chair", "couch", "potted plant", "bed",
        "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven",
        "toaster", "sink", "refrigerator", "book", "clock",
        "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    std::cout << "Yolo model loaded successfully!" << std::endl;
}

cv::Mat YoloDetector::letterbox(
    const cv::Mat& image,
    float& scale,
    int& padX,
    int& padY
) {
    const int inputWidth = 640;
    const int inputHeight = 640;

    float scaleX =
        static_cast<float>(inputWidth) / image.cols;
    float scaleY =
        static_cast<float>(inputHeight) / image.rows;
    scale = std::min(scaleX, scaleY);

    int newWidth = static_cast<int>(image.cols * scale);
    int newHeight = static_cast<int>(image.rows * scale);
    cv::Mat resized;

    cv::resize(
        image,
        resized,
        cv::Size(newWidth, newHeight));

    int totalPadX = inputWidth - newWidth;
    int totalPadY = inputHeight - newHeight;

    padX = totalPadX / 2;
    padY = totalPadY / 2;

    int rightPad = totalPadX - padX;
    int bottomPad = totalPadY - padY;

    cv::Mat padded;

    cv::copyMakeBorder(
        resized,
        padded,
        padY,
        bottomPad,
        padX,
        rightPad,
        cv::BORDER_CONSTANT,
        cv::Scalar(114,  114, 114));

    return padded;
}

std::vector<Detection> YoloDetector::detect(const cv::Mat& frame) {

    float scale;
    int padX, padY;
    cv::Mat inputImage = letterbox(frame, scale, padX, padY);
    cv::Mat blob = cv::dnn::blobFromImage(
        inputImage,
        1.0 / 255.0,
        cv::Size(640, 640),
        cv::Scalar(),
        true,
        false
    );

    net.setInput(blob);
    std::vector<cv::Mat> outputs;

    auto inferenceStart = std::chrono::steady_clock::now();

    net.forward(
      outputs,
      net.getUnconnectedOutLayersNames()
    );
    auto inferenceEnd = std::chrono::steady_clock::now();
    inferenceTimeMs =
        std::chrono::duration<double, std::milli>(inferenceEnd - inferenceStart).count();

    cv::Mat output = outputs[0];
    cv::Mat detections(
        output.size[1],
        output.size[2],
        CV_32F,
        output.ptr<float>());

    detections = detections.t();

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<float> classIds;

    const float confidenceThreshold = 0.5f;
    const float nmsThreshold = 0.4f;
    for (int i = 0; i < detections.rows; i++) {
        cv::Mat classScores = detections.row(i).colRange(4, 84);

        cv::Point classIdPoint;
        double confidence;

        cv::minMaxLoc(
            classScores,
            nullptr,
            &confidence,
            nullptr,
            &classIdPoint
        );

        if (confidence >= confidenceThreshold) {
            float x = detections.at<float>(i, 0);
            float y = detections.at<float>(i, 1);
            float w = detections.at<float>(i, 2);
            float h = detections.at<float>(i, 3);

            float xScale = static_cast<float>(frame.cols) / 640.0f;
            float yScale = static_cast<float>(frame.rows) / 640.0f;
            int left = static_cast<int>((x - w / 2.0f) * xScale);
            int top = static_cast<int>((y - h / 2.0f) * yScale);
            int width = static_cast<int>(w * xScale);
            int height = static_cast<int>(h * yScale);

            cv::Rect box(
              left,
              top,
              width,
              height
            );

            cv::Rect imageBounds(
                0,
                0,
                frame.cols,
                frame.rows
            );
            box = box & imageBounds;
            if (box.empty()) {
                continue;
            }
            boxes.push_back(box);

            confidences.push_back(
                static_cast<float>(confidence)
            );

            classIds.push_back(classIdPoint.x);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(
        boxes,
        confidences,
        confidenceThreshold,
        nmsThreshold,
        indices);

    std::vector<Detection> results;
    for (int index : indices) {
        int classId = classIds[index];
        Detection detection;
        detection.classId = classId;
        detection.className = classNames[classId];
        detection.confidence = confidences[index];
        detection.box = boxes[index];
        results.push_back(detection);
    }

    return results;
}

double YoloDetector::getInferenceTime() const {
    return inferenceTimeMs;
}
