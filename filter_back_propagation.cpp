#include <Windows.h>
#include "filter_back_propagation.hpp"
#include "make_sinogram.h"
#include "contact_error.h"

Mat Perform_radon_Transform(string file_name = "CTex.png", uint arbit_angle = 180) {
    Mat img = imread(file_name, IMREAD_COLOR);
    if (img.empty()) {
        std::string error_load_image = "이미지를 불러오는 데 실패하였습니다.";
        ShowErrorMessageBox(error_load_image);
        exit(1);
    }

    img.convertTo(img, CV_64F, 1.0 / 255.0);

    vector<double> theta(arbit_angle);
    for (int i = 0; i < arbit_angle; ++i) {
        theta[i] = i;
    }

    Mat R = radonTransform(img, theta);

    return R;

}

Mat sinogram_make(Mat R) {
    Mat R_normalized;
    normalize(R, R_normalized, 0, 255, NORM_MINMAX);
    R_normalized.convertTo(R_normalized, CV_8U);
    imwrite("sinogram.png", R_normalized);

    return R_normalized;
}

Mat Perform_iradon_Transform() {
    Mat sinogram = imread("sinogram.png", IMREAD_COLOR);
    if (sinogram.empty()) {
        std::string error_load_sinogram = "Sinogram을 불러오는 데 실패하였습니다.";
        ShowErrorMessageBox(error_load_sinogram);
        exit(1);
    }

    Mat filtered_sinogram = filter_sinogram(sinogram); //filter the sinogram
    //imwrite("filtered_sinogram.png", filtered_sinogram);


    Mat reconstruction = iradon(filtered_sinogram, false); //perform back projection. Change false to true if sinogram is a full turn
    renormalize255_frame(reconstruction); //normalize to 255

    Mat F_normalized;
    normalize(reconstruction, F_normalized, 0, 255, NORM_MINMAX);
    F_normalized.convertTo(F_normalized, CV_8U);
    imwrite("Reconstruction.png", reconstruction); //save reconstruction

    return F_normalized;
}

int main() {
    while (true) {
        string File_name;
        uint Projection_Angle;

        cout << "이미지 이름 및 확장자를 입력하세요" << endl;
        cout << "이미지는 프로젝트 경로 내에 존재해야 합니다." << endl;
        cout << "Ex) CT.png  혹은 Ex) C:\\Users\\[Username]\\desktop\\CT.png" << endl;
        cin >> File_name;

        cout << "투사 각도를 입력하세요(0~360)." << endl;
        cin >> Projection_Angle;

        Mat radon_transformed = Perform_radon_Transform(File_name, Projection_Angle);
        Mat R_normalized = sinogram_make(radon_transformed);
        Mat F_normalized = Perform_iradon_Transform();

        namedWindow("Sinogram", WINDOW_NORMAL);
        namedWindow("Reconstructed Image", WINDOW_NORMAL);
        imshow("Sinogram", R_normalized);
        imshow("Reconstructed Image", F_normalized);
        waitKey(0);

        return 0;
    }
}
