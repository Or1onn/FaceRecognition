#include "visualize.h"

static void visualize(cv::Mat &input, cv::Mat &faces) {
    for (int i = 0; i < faces.rows; i++) {
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
}