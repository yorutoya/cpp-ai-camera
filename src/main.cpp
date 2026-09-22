#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

int main()
{
    std::cout << "OpenCV version: " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << std::endl;

    cv::VideoCapture camera(0);

    if (!camera.isOpened()) {
        std::cerr << "Error: Cannot open camera." << std::endl;
        return 1;
    }

    cv::Mat frame;

    auto previousTime = std::chrono::steady_clock::now();

    while (true) {
        camera >> frame;

        if (frame.empty()) {
            std::cerr << "Error: Empty frame." << std::endl;
            break;
        }

        //Calculate FPS
        auto currentTime = std::chrono::steady_clock::now();
        double elapsedTime = std::chrono::duration<double>(currentTime - previousTime).count();
        double fps = 1.0 / elapsedTime;
        previousTime = currentTime;

        // Draw FPS
        std::string fpsText = "FPS: " + std::to_string(static_cast<int>(fps));
        cv::putText(
            frame,
            fpsText,
            cv::Point(20,40),
            cv::FONT_HERSHEY_SIMPLEX,
            1.0,
            cv::Scalar(0, 255, 0),
            2
        );

        cv::imshow("AI Camera", frame);

        int key = cv::waitKey(1);

        //Use keyboard to quit
        if (key == 'q' || key == 'Q' || key == 27) {
            break;
        }

        //Use right corner to quit
        if (cv::getWindowProperty(
                "AI Camera",
                cv::WND_PROP_VISIBLE
            ) < 1) {
            break;
        }
    }

    camera.release();
    cv::destroyAllWindows();

    return 0;
}