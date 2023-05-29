#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <filesystem>

static
void visualize(cv::Mat &input, int frame, cv::Mat &faces, double fps) {
    std::string fpsString = cv::format("FPS : %.2f", (float) fps);
    if (frame >= 0)
        std::cout << "Frame " << frame << ", ";
    std::cout << "FPS: " << fpsString << std::endl;
    for (int i = 0; i < faces.rows; i++) {
        // Print results
        std::cout << "Face " << i
                  << ", top-left coordinates: (" << faces.at<float>(i, 0) << ", " << faces.at<float>(i, 1) << "), "
                  << "box width: " << faces.at<float>(i, 2) << ", box height: " << faces.at<float>(i, 3) << ", "
                  << "score: " << cv::format("%.2f", faces.at<float>(i, 14))
                  << std::endl;
        // Draw bounding box
        rectangle(input, cv::Rect2i(int(faces.at<float>(i, 0)), int(faces.at<float>(i, 1)), int(faces.at<float>(i, 2)),
                                    int(faces.at<float>(i, 3))), cv::Scalar(0, 255, 0),
                  2);
        // Draw landmarks
        circle(input, cv::Point2i(int(faces.at<float>(i, 4)), int(faces.at<float>(i, 5))), 2, cv::Scalar(255, 0, 0), 2);
        circle(input, cv::Point2i(int(faces.at<float>(i, 6)), int(faces.at<float>(i, 7))), 2, cv::Scalar(0, 0, 255), 2);
        circle(input, cv::Point2i(int(faces.at<float>(i, 8)), int(faces.at<float>(i, 9))), 2, cv::Scalar(0, 255, 0), 2);
        circle(input, cv::Point2i(int(faces.at<float>(i, 10)), int(faces.at<float>(i, 11))), 2, cv::Scalar(255, 0, 255),
               2);
        circle(input, cv::Point2i(int(faces.at<float>(i, 12)), int(faces.at<float>(i, 13))), 2, cv::Scalar(0, 255, 255),
               2);
    }
    putText(input, fpsString, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
}

int main(int argc, char **argv) {
    std::filesystem::path executablePath = std::filesystem::current_path().parent_path();
    std::filesystem::path executablePathAssets = std::filesystem::current_path().parent_path() += "\\Assets\\";

    cv::String fd_modelPath = executablePathAssets.string() + "face_detection_yunet_2022mar.onnx";
    cv::String fr_modelPath = executablePathAssets.string() + "deploy.prototxt";

    float scoreThreshold = 0.9f;
    float nmsThreshold = 0.3f;
    int topK = 5000;
    bool save = false;
    float scale = 1.0f;

    // Initialize FaceDetectorYN
    cv::Ptr<cv::FaceDetectorYN> detector = cv::FaceDetectorYN::create(fd_modelPath, "", cv::Size(320, 320),
                                                                      scoreThreshold, nmsThreshold, topK);
    cv::TickMeter tm;
    // If input is an image

    int frameWidth, frameHeight;
    cv::VideoCapture capture;

    capture.open(0);

    if (capture.isOpened()) {
        frameWidth = int(capture.get(cv::CAP_PROP_FRAME_WIDTH) * scale);
        frameHeight = int(capture.get(cv::CAP_PROP_FRAME_HEIGHT) * scale);
        std::cout << "Video "
                  << ": width=" << frameWidth
                  << ", height=" << frameHeight
                  << std::endl;
    } else {
        std::cout << "Could not initialize video capturing" << "\n";
        return 1;
    }
    detector->setInputSize(cv::Size(frameWidth, frameHeight));
    std::cout << "Press 'SPACE' to save frame, any other key to exit..." << std::endl;
    int nFrame = 0;
    while (true) {
        // Get frame
        cv::Mat frame;
        if (!capture.read(frame)) {
            std::cerr << "Can't grab frame! Stop\n";
            break;
        }
        resize(frame, frame, cv::Size(frameWidth, frameHeight));
        // Inference
        cv::Mat faces;
        tm.start();
        detector->detect(frame, faces);
        tm.stop();
        cv::Mat result = frame.clone();
        // Draw results on the input image
        visualize(result, nFrame, faces, tm.getFPS());
        // Visualize results
        imshow("Live", result);
        int key = cv::waitKey(1);
        bool saveFrame = save;
        if (key == ' ') {
            saveFrame = true;
            key = 0;  // handled
        }
        if (saveFrame) {
            std::string frame_name = cv::format("frame_%05d.png", nFrame);
            std::string result_name = cv::format("result_%05d.jpg", nFrame);
            std::cout << "Saving '" << frame_name << "' and '" << result_name << "' ...\n";
            imwrite(frame_name, frame);
            imwrite(result_name, result);
        }
        ++nFrame;
        if (key > 0)
            break;
    }
    std::cout << "Processed " << nFrame << " frames" << std::endl;
    std::cout << "Done." << std::endl;
    return 0;
}