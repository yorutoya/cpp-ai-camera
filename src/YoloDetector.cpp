#include "YoloDetector.hpp"
#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <chrono>

YoloDetector::YoloDetector(const std::string &modelPath) {

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

}

cv::Mat YoloDetector::letterbox(
    const cv::Mat& image,
    float& scale,
    int& padX,
    int& padY
) {
    const int inputWidth = inputSize;
    const int inputHeight = inputSize;

    float scaleX =
        static_cast<float>(inputWidth) / image.cols;
    float scaleY =
        static_cast<float>(inputHeight) / image.rows;
    scale = std::min(scaleX, scaleY);

    int newWidth = std::clamp(static_cast<int>(std::round(image.cols * scale)), 1, inputWidth);
    int newHeight = std::clamp(static_cast<int>(std::round(image.rows * scale)), 1, inputHeight);
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

    inferenceTimeMs = 0.0;
    if (frame.empty() || frame.type() != CV_8UC3) {
        throw std::invalid_argument("detect() requires a nonempty 8-bit BGR image.");
    }

    float scale;
    int padX, padY;
    cv::Mat inputImage = letterbox(frame, scale, padX, padY);
    cv::Mat blob = cv::dnn::blobFromImage(
        inputImage,
        1.0 / 255.0,
        cv::Size(inputSize, inputSize),
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

    // Contract: raw YOLO11 COCO detection export, batch 1, no embedded NMS.
    const int attributes = 4 + static_cast<int>(classNames.size());
    if (outputs.size() != 1 || outputs.front().empty()) {
        throw std::runtime_error("Expected one nonempty YOLO detection output.");
    }
    const cv::Mat& output = outputs.front();
    if (output.type() != CV_32F || output.dims != 3 ||
        output.size[0] != 1 || output.size[1] != attributes ||
        output.size[2] <= 0 || !output.isContinuous()) {
        throw std::runtime_error(
            "Expected contiguous float32 YOLO11 COCO output [1,84,N]. "
            "Use a 640x640 detection export without embedded NMS.");
    }
    const cv::Mat detections = output.reshape(1, attributes).t();

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;

    for (int i = 0; i < detections.rows; i++) {
        cv::Mat classScores = detections.row(i).colRange(4, attributes);

        cv::Point classIdPoint;
        double confidence;

        cv::minMaxLoc(
            classScores,
            nullptr,
            &confidence,
            nullptr,
            &classIdPoint
        );

        if (std::isfinite(confidence) && confidence > confidenceThreshold) {
            float x = detections.at<float>(i, 0);
            float y = detections.at<float>(i, 1);
            float w = detections.at<float>(i, 2);
            float h = detections.at<float>(i, 3);

            if (!std::isfinite(x) || !std::isfinite(y) ||
                !std::isfinite(w) || !std::isfinite(h) || w <= 0 || h <= 0) {
                continue;
            }

            // Undo centered padding first, then the uniform letterbox gain.
            // Clip floating-point corners before converting to integer pixels.
            const double left = std::clamp(
                (static_cast<double>(x) - w / 2.0 - padX) / scale,
                0.0, static_cast<double>(frame.cols));
            const double top = std::clamp(
                (static_cast<double>(y) - h / 2.0 - padY) / scale,
                0.0, static_cast<double>(frame.rows));
            const double right = std::clamp(
                (static_cast<double>(x) + w / 2.0 - padX) / scale,
                0.0, static_cast<double>(frame.cols));
            const double bottom = std::clamp(
                (static_cast<double>(y) + h / 2.0 - padY) / scale,
                0.0, static_cast<double>(frame.rows));
            if (right <= left || bottom <= top) {
                continue;
            }
            const int x1 = static_cast<int>(std::floor(left));
            const int y1 = static_cast<int>(std::floor(top));
            const cv::Rect box(x1, y1,
                static_cast<int>(std::ceil(right)) - x1,
                static_cast<int>(std::ceil(bottom)) - y1);
            boxes.push_back(box);

            confidences.push_back(
                static_cast<float>(confidence)
            );

            classIds.push_back(classIdPoint.x);
        }
    }

    // Suppress duplicates within each class. Overlapping different objects survive.
    // Per-class NMSBoxes also works with OpenCV versions predating NMSBoxesBatched.
    std::vector<int> indices;
    for (int classId = 0; classId < static_cast<int>(classNames.size()); ++classId) {
        std::vector<cv::Rect> classBoxes;
        std::vector<float> classConfidences;
        std::vector<int> originalIndices;
        for (int i = 0; i < static_cast<int>(boxes.size()); ++i) {
            if (classIds[i] == classId) {
                classBoxes.push_back(boxes[i]);
                classConfidences.push_back(confidences[i]);
                originalIndices.push_back(i);
            }
        }
        if (classBoxes.empty()) {
            continue;
        }
        std::vector<int> kept;
        cv::dnn::NMSBoxes(classBoxes, classConfidences,
            confidenceThreshold, nmsThreshold, kept);
        for (int i : kept) {
            indices.push_back(originalIndices[i]);
        }
    }
    std::stable_sort(indices.begin(), indices.end(), [&](int a, int b) {
        return confidences[a] > confidences[b];
    });

    std::vector<Detection> results;
    results.reserve(indices.size());
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
