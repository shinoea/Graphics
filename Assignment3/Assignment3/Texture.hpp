//
// Created by LEI XU on 4/27/19.
//

#ifndef RASTERIZER_TEXTURE_H
#define RASTERIZER_TEXTURE_H
#include "global.hpp"
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
class Texture{
private:
    cv::Mat image_data;

public:
    Texture(const std::string& name)
    {
        image_data = cv::imread(name);
        cv::cvtColor(image_data, image_data, cv::COLOR_RGB2BGR);
        width = image_data.cols;
        height = image_data.rows;
    }

    int width, height;

    Eigen::Vector3f getColor(float u, float v)
    {
        // 限制(u, v)坐标范围
        u = std::clamp(u, 0.0f, 1.f);
        v = std::clamp(v, 0.0f, 1.f);

        auto u_img = u * width;
        auto v_img = (1 - v) * height;
        auto color = image_data.at<cv::Vec3b>(v_img, u_img);
        return Eigen::Vector3f(color[0], color[1], color[2]);
    }

    //双线性插值
    Eigen::Vector3f getColorBilinear(float u, float v)
    {
        // 限制(u, v)坐标范围
        u = std::clamp(u, 0.0f, 1.f);
        v = std::clamp(v, 0.0f, 1.f);

        auto u_img = u * width;
        auto v_img = (1 - v) * height;

        float u0_ = std::fmax(0.0f, floor(u_img - 0.5));//u01和u00的u坐标
        float u1_ = std::fmin(width, floor(u_img + 0.5));//u10和u11的u坐标
        float v0_ = std::fmax(0.0f, floor(v_img - 0.5));//u01和u00的v坐标
        float v1_ = std::fmin(height, floor(v_img + 0.5));//u10和u11的v坐标
        float s = (u_img - u0_) / (u1_ - u0_), t = (v_img - v0_) / (v1_ - v0_);

        auto u00 = image_data.at<cv::Vec3b>(v0_, u0_);
        auto u01= image_data.at<cv::Vec3b>(v1_, u0_);
        auto u10= image_data.at<cv::Vec3b>(v0_, u1_);
        auto u11= image_data.at<cv::Vec3b>(v1_, u1_);

        auto u0 = u00 + s * (u10 - u00);
        auto u1 = u01 + s * (u11 - u01);
        auto color = u0 + t * (u1 - u0);

        return Eigen::Vector3f(color[0], color[1], color[2]);
    }

};
#endif //RASTERIZER_TEXTURE_H
