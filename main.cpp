#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

#include "visualize.cpp"

int main() {
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

    int frameWidth, frameHeight;
    cv::VideoCapture capture;

    capture.open(0);

    if (capture.isOpened()) {
        frameWidth = int(capture.get(cv::CAP_PROP_FRAME_WIDTH) * scale);
        frameHeight = int(capture.get(cv::CAP_PROP_FRAME_HEIGHT) * scale);
    } else {
        std::cout << "Could not initialize video capturing" << "\n";
        return 1;
    }

    detector->setInputSize(cv::Size(frameWidth, frameHeight));

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
        detector->detect(frame, faces);
        cv::Mat result = frame.clone();
        // Draw results on the input image
        visualize(result, faces);
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