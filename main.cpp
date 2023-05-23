#include <iostream>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>

int main() {
    cv::VideoCapture video_capture;
    if (!video_capture.open(0)) {
        return 0;
    }

    cv::CascadeClassifier faceDetect;
    cv::Mat frame;

    std::filesystem::path executablePath = std::filesystem::current_path().parent_path() += "\\haarcascade_frontalface_default.xml";

    faceDetect.load(executablePath.string());

    while (true) {
        video_capture >> frame;
        std::vector<cv::Rect> faces;

        faceDetect.detectMultiScale(frame, faces, 1.3, 5);
        for (auto face : faces) {
            cv::rectangle(frame, face.tl(), face.br(), cv::Scalar(50, 50, 255), 2);

        }
        imshow("Image", frame);

        if (cv::waitKey(10) == 27) {
            break;
        }
    }

    cv::destroyAllWindows();
    video_capture.release();


    return 0;
}
