#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <chrono>
#include "YoloDetector.hpp"
#include <algorithm>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>
#include "CameraAnalyzer.hpp"

int main(int argc, char* argv[])
{
    try
    {
        if (argc > 2) {
            std::cerr << "Usage: ai-camera [path/to/yolo11n.onnx]\n";
            return 1;
        }
        const std::string modelPath = argc == 2 ? argv[1] : "models/yolo11n.onnx";
        YoloDetector detector(modelPath);
        const CameraAnalyzer analyzer;

        cv::VideoCapture camera(0);

        if (!camera.isOpened())
        {
            std::cerr << "Cannot open camera." << std::endl;
            return 1;
        }

        const std::string windowName = "AI Camera";

        cv::namedWindow(
            windowName,
            cv::WINDOW_AUTOSIZE
        );

        cv::Mat frame;

        auto previousTime = std::chrono::steady_clock::now();
        double fps = 0.0;

        while (true) {
            if (!camera.read(frame) || frame.empty()) {
                throw std::runtime_error("Camera failed to return a frame.");
            }

            // AI detection
            const std::vector<Detection> detections = detector.detect(frame);
            const AnalysisResult analysis = analyzer.analyze(detections);

            for (const Detection& detection : detections) {
                cv::rectangle(
                    frame,
                    detection.box,
                    cv::Scalar(0, 255, 0),
                    2
                );

                std::string label = detection.className + " " + std::to_string(static_cast<int>(detection.confidence * 100)) +"%";
                cv::putText(
                    frame,
                    label,
                    cv::Point(
                        detection.box.x,
                        std::max(20, detection.box.y - 10)
                    ),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.7,
                    cv::Scalar(0, 255, 0),
                    2
                );
            }

            std::string objectText = "Objects: " + std::to_string(analysis.totalObjects);
            cv::putText(
                frame,
                objectText,
                cv::Point(20, 130),
                cv::FONT_HERSHEY_SIMPLEX,
                0.7,
                cv::Scalar(0, 255, 0),
                2
            );

            int y = 230;
            for (const auto& [className, count] : analysis.objectCounts) {
                std::string text = className + ": " + std::to_string(count);
                cv::putText(
                    frame,
                    text,
                    cv::Point(20, y),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.7,
                    cv::Scalar(0, 255, 0),
                    2
                );
                y += 30;
            }

            auto currentTime = std::chrono::steady_clock::now();
            double frameTime = std::chrono::duration<double>(
                currentTime-previousTime).count();
            previousTime = currentTime;
            if (frameTime > 0) {
                fps = 1.0 / frameTime;
            }

            cv::putText(
                frame,
                "AI CAMERA ANALYZER",
                cv::Point(20, 35),
                cv::FONT_HERSHEY_SIMPLEX,
                0.8,
                cv::Scalar(0, 255, 0),
                2
            );

            std::string fpsText = "FPS: " + std::to_string(static_cast<int>(fps));
            std::string inferenceText = "Inference: " + std::to_string(static_cast<int>(detector.getInferenceTime())) + " ms";
            cv::putText(
                frame,
                fpsText,
                cv::Point(20, 70),
                cv::FONT_HERSHEY_SIMPLEX,
                0.8,
                cv::Scalar(0, 255, 0),
                2
            );
            cv::putText(
                frame,
                inferenceText,
                cv::Point(20, 100),
                cv::FONT_HERSHEY_SIMPLEX,
                0.8,
                cv::Scalar(0, 255, 0),
                2
            );

            std::string personText = "People: " + std::to_string(analysis.personCount);
            std::string statusText = analysis.occupied ? "Status: OCCUPIED" : "Status: EMPTY";
            cv::putText(
                frame,
                personText,
                cv::Point(20, 160),
                cv::FONT_HERSHEY_SIMPLEX,
                0.7,
                cv::Scalar(0, 255, 0),
                2
            );
            cv::putText(
                frame,
                statusText,
                cv::Point(20, 190),
                cv::FONT_HERSHEY_SIMPLEX,
                0.7,
                cv::Scalar(0, 255, 0),
                2
            );

            cv::imshow(windowName, frame);

            int key = cv::waitKey(1);

            if (key == 'q' ||
                key == 'Q' ||
                key == 27)
            {
                break;
            }

            if (cv::getWindowProperty(
                    windowName,
                    cv::WND_PROP_VISIBLE
                ) < 1)
            {
                break;
            }
        }

        camera.release();
        cv::destroyAllWindows();
    }
    catch (const cv::Exception& error)
    {
        std::cerr
            << "OpenCV error:\n"
            << error.what()
            << std::endl;

        return 1;
    }
    catch (const std::exception& error)
    {
        std::cerr
            << "Error: "
            << error.what()
            << std::endl;

        return 1;
    }

    return 0;
}