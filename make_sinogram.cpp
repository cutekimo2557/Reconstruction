#include "make_sinogram.hpp"

Mat radonTransform(const Mat& img, const std::vector<double>& theta) {
    int rows = img.rows;
    int cols = img.cols;
    int diagonal = std::ceil(std::sqrt(rows * rows + cols * cols));
    Mat radon_image = Mat::zeros(diagonal, theta.size(), CV_64F);

    Point2f center(cols / 2.0, rows / 2.0);
    Mat rotation_matrix, rotated_img;

    for (size_t t = 0; t < theta.size(); ++t) {
        rotation_matrix = getRotationMatrix2D(center, theta[t], 1.0);
        warpAffine(img, rotated_img, rotation_matrix, Size(cols, rows), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));

        for (int i = 0; i < diagonal; ++i) {
            Point pt1(0, center.y - i + diagonal / 2);
            Point pt2(cols, center.y - i + diagonal / 2);
            LineIterator it(rotated_img, pt1, pt2, 8);
            for (int j = 0; j < it.count; ++j, ++it) {
                radon_image.at<double>(i, t) += (*it)[0];
            }
        }
    }
    return radon_image;
}
