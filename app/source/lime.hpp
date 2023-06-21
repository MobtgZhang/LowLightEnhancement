# pragma once
# ifndef LIME_HPP_FILE
# define LIME_HPP_FILE
# include <opencv2/opencv.hpp>
constexpr unsigned int window_1{ 7 };

void get_illuminationmap(const cv::Mat&  , const cv::Mat*, cv::Mat & , cv::Mat & , cv::Mat &,cv::Mat & );
cv::Mat fastGuidedFilter(const cv::Mat &I_org, const cv::Mat &p_org, int r, double eps, int s);
void cal_weight(const cv::Mat &image, const cv::Mat *HSV_channel,cv::Mat &weight_bright, cv::Mat &weight_constract, const double &alpha, const double &fai);
void multi_fusion(const cv::Mat& image_double, cv::Mat &image_enhanced, const double & alpha, const double &fai);

# endif
