#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>

#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>

int main() {
    cv::VideoCapture video_capture;
    if (!video_capture.open(0)) {
        return 0;
    }

    std::filesystem::path executablePath = std::filesystem::current_path().parent_path();
    std::filesystem::path executablePathAssets = std::filesystem::current_path().parent_path() += "\\Assets\\";

    std::string model = executablePathAssets.string() + "squeezenet_v1.1.caffemodel";
    std::string config = executablePathAssets.string() + "deploy.prototxt";

    cv::CascadeClassifier faceDetect;
    cv::Mat frame;

    std::float_t scale = 0.017;
    cv::Scalar mean = cv::Scalar(104.0, 117.0, 123.0);
    cv::Size size = cv::Size(227, 227);


    faceDetect.load(executablePath.string());
    cv::dnn::Net net = cv::dnn::readNet(model, config);

    while (true) {
        video_capture >> frame;
        std::vector<cv::Rect> faces;

        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, scale, size, mean, false, false);

        net.setInput(blob);

        cv::Mat detection = net.forward();
        cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F,
                             detection.ptr<float>());

        cv::Mat output;
        frame.copyTo(output);

        float confidenceThreshold = 0.5;
        for (int i = 0; i < detectionMat.rows; i++) {
            float confidence = detectionMat.at<float>(i, 2);

            if (confidence > confidenceThreshold) {

                int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3)
                                                   * frame.cols);
                int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4)
                                                   * frame.rows);
                int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5)
                                                 * frame.cols);
                int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6)
                                                 * frame.rows);

                cv::Rect boundBox(xLeftBottom, yLeftBottom, (xRightTop - xLeftBottom), (yRightTop - yLeftBottom));
                cv::Rect headerBox(xLeftBottom, cv::max(0, yLeftBottom - 15), (xRightTop - xLeftBottom), 15);

                rectangle(output, headerBox, cv::Scalar(0, 255, 0), -1);

                rectangle(output, boundBox, cv::Scalar(0, 255, 0), 2);
            }
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