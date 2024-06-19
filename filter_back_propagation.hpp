#pragma once            //이 헤더 파일 중복 정의 방지    

#include <opencv2/opencv.hpp>   //opencv 내의 imwrite, imread 등의 함수들을 사용하기 위해 정의
#include <iostream>             //입출력 스트림 사용하기 위한 정의
#include <cstdlib>              //메모리 관리 및 변수 관리를 위한 정의
#include <cstring>              //문자열 처리 관련 정의
#include <vector>               //벡터 형태의 자료 구조를 사용하기 위한 정의
#include <fstream>              //파일 관련 입출력을 사용하기 위한 정의
#include <cmath>          //C++ 관련 수학 함수들을 사용하기 위한 정의

#define M_PI 3.141592   // 원주율 매크로 정의

using namespace cv;     //이름공간 cv 정의
using namespace std;    //이름공간 std 정의

Mat iradon(Mat& sinogram, bool full_turn) //Sinogram must be a 32bit single channel grayscale image normalized 0-1
{
    Mat reconstruction(sinogram.size().height, sinogram.size().height, CV_32FC1);
    float delta_t;
    if (full_turn) delta_t = 2.0 * M_PI / sinogram.size().width;
    else delta_t = 1.0 * M_PI / sinogram.size().width;
    unsigned int t, f, c, rho;
    for (f = 0; f < reconstruction.size().height; f++)
    {
        for (c = 0; c < reconstruction.size().width; c++)
        {
            reconstruction.at<float>(f, c) = 0;
            for (t = 0; t < sinogram.size().width; t++)
            {
                rho = ((f - 0.5 * sinogram.size().height) * cos(delta_t * t) + (c - 0.5 * sinogram.size().height) * sin(delta_t * t) + 0.5 * sinogram.size().height);
                if ((rho > 0) && (rho < sinogram.size().height)) reconstruction.at<float>(f, c) += sinogram.at<float>(rho, t);
            }
            if (reconstruction.at<float>(f, c) < 0)reconstruction.at<float>(f, c) = 0;
        }
    }
    rotate(reconstruction, reconstruction, ROTATE_90_CLOCKWISE);
    return reconstruction;
}

void renormalize255_frame(Mat& frame)
{
    unsigned int f, c;
    float maxm = 0;
    for (f = 0; f < frame.size().height; f++)
    {
        for (c = 0; c < frame.size().width; c++)
        {
            if (frame.at<float>(f, c) > maxm)maxm = frame.at<float>(f, c);
        }
    }

    for (f = 0; f < frame.size().height; f++)
    {
        for (c = 0; c < frame.size().width; c++)
        {
            frame.at<float>(f, c) = frame.at<float>(f, c) * 255.0 / maxm;
        }
    }
    return;
}

void convert_frame2bw(Mat& frame)
{
    unsigned int f, c;
    Mat converted(frame.size().height, frame.size().width, CV_32FC1);
    for (f = 0; f < frame.size().height; f++)
    {
        for (c = 0; c < frame.size().width; c++)
        {
            converted.at<float>(f, c) = 1.0 / 3.0 * (frame.at<Vec3f>(f, c)[0] + frame.at<Vec3f>(f, c)[1] + frame.at<Vec3f>(f, c)[2]);
            //converted.at<float>(f,c)=frame.at<Vec3f>(f,c)[1];
        }
    }
    frame = converted.clone();
    return;
}

void convert_frame2f(Mat& frame)
{
    frame.convertTo(frame, CV_32FC3, 1.0 / 255.0);
    return;
}

Mat filter_sinogram(Mat& sinogram)
{
    Mat filtered_sinogram;
    transpose(sinogram, filtered_sinogram);
    if (filtered_sinogram.type() == CV_8UC3) //Convert to gray scale and 32bit if the input is a 8bit RGB image
    {
        cout << "Converting to 32bit grayscale..." << endl;
        convert_frame2f(filtered_sinogram);
        convert_frame2bw(filtered_sinogram);
    }
    Mat dft_sinogram[2] = { filtered_sinogram,Mat::zeros(filtered_sinogram.size(),CV_32F) };
    Mat dftReady;
    merge(dft_sinogram, 2, dftReady);
    dft(dftReady, dftReady, DFT_ROWS | DFT_COMPLEX_OUTPUT, 0);
    split(dftReady, dft_sinogram);
    unsigned int f, c;
    for (f = 0; f < dft_sinogram[0].size().height; f++)
    {
        for (c = 0; c < dft_sinogram[0].size().width; c++)
        {
            //Sine Filter
            dft_sinogram[0].at<float>(f, c) *= (1.0 / (2.0 * M_PI)) * 1.0 * abs(sin(1.0 * M_PI * (c) / dft_sinogram[0].size().width));
            dft_sinogram[1].at<float>(f, c) *= (1.0 / (2.0 * M_PI)) * 1.0 * abs(sin(1.0 * M_PI * (c) / dft_sinogram[0].size().width));
        }
    }
    merge(dft_sinogram, 2, dftReady);
    dft(dftReady, filtered_sinogram, DFT_INVERSE | DFT_ROWS | DFT_REAL_OUTPUT, 0);
    transpose(filtered_sinogram, filtered_sinogram);
    return filtered_sinogram;
}

