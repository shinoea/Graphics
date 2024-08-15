#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

std::vector<cv::Point2f> control_points;

void mouse_handler(int event, int x, int y, int flags, void *userdata) 
{
    if (event == cv::EVENT_LBUTTONDOWN && control_points.size() < 4) 
    {
        std::cout << "Left button of the mouse is clicked - position (" << x << ", "
        << y << ")" << '\n';
        control_points.emplace_back(x, y);
    }     
}

void naive_bezier(const std::vector<cv::Point2f> &points, cv::Mat &window) 
{
    auto &p_0 = points[0];
    auto &p_1 = points[1];
    auto &p_2 = points[2];
    auto &p_3 = points[3];

    for (double t = 0.0; t <= 1.0; t += 0.001) 
    {
        auto point = std::pow(1 - t, 3) * p_0 + 3 * t * std::pow(1 - t, 2) * p_1 +
                 3 * std::pow(t, 2) * (1 - t) * p_2 + std::pow(t, 3) * p_3;

        window.at<cv::Vec3b>(point.y, point.x)[2] = 255;
    }
}

cv::Point2f recursive_bezier(const std::vector<cv::Point2f> &control_points, float t) 
{
    // TODO: Implement de Casteljau's algorithm
    // 递归出口条件：一阶贝塞尔曲线
    auto& p = control_points;
    if (p.size() == 2)
        return (1 - t) * p[0] + t * p[1];
    std::vector<cv::Point2f> b;
    for (int i = 0; i < p.size() - 1; i++)
        b.push_back((1 - t) * p[i] + t * p[i + 1]);
    return recursive_bezier(b, t);
}

void bezier(const std::vector<cv::Point2f> &control_points, cv::Mat &window) 
{
    // TODO: Iterate through all t = 0 to t = 1 with small steps, and call de Casteljau's 
    // recursive Bezier algorithm.
    for (float t = 0.0; t <= 1; t += 0.001)
    {
        auto point = recursive_bezier(control_points, t);
        //绿色通道值设为最大值
        window.at<cv::Vec3b>(point.y, point.x)[1] = 255;
    }
}

void bezier2(const std::vector<cv::Point2f>& control_points, cv::Mat& window)
{
    // TODO: Iterate through all t = 0 to t = 1 with small steps, and call de Casteljau's 
    // recursive Bezier algorithm.
    for (float t = 0.0; t <= 1; t += 0.001)
    {
        auto point = recursive_bezier(control_points, t);
        window.at<cv::Vec3b>(point.y, point.x)[2] = 255;
        int x = point.x, y = point.y;// 转成int类型，截断小数部分，+0.5代表像素中心
        for (int dx = -1; dx <= 1; dx++)
        {
            for (int dy = -1; dy <= 1; dy++)
            {
                if (x + dx > 700 || x + dx < 0 || y + dy > 700 || y + dy < 0) continue;
                double d = std::sqrt(std::pow(point.x - (x + dx + 0.5), 2) + std::pow(point.y - (y + dy + 0.5), 2));
                double ratio = 1 - sqrt(2) * d / 3;
                window.at<cv::Vec3b>(point.y + dy, point.x + dx)[1] = std::fmax(window.at<cv::Vec3b>(point.y + dy, point.x + dx)[1], 255 * ratio);
            }
        }
    }
}

int main() 
{
    cv::Mat window = cv::Mat(700, 700, CV_8UC3, cv::Scalar(0));
    cv::cvtColor(window, window, cv::COLOR_BGR2RGB);
    cv::namedWindow("Bezier Curve", cv::WINDOW_AUTOSIZE);

    cv::setMouseCallback("Bezier Curve", mouse_handler, nullptr);

    int key = -1;
    while (key != 27) 
    {
        for (auto &point : control_points) 
        {
            cv::circle(window, point, 3, {255, 255, 255}, 3);
        }

        if (control_points.size() == 4) 
        {
            //naive_bezier(control_points, window);
            //bezier(control_points, window);
            bezier2(control_points, window);

            cv::imshow("Bezier Curve", window);
            cv::imwrite("my_bezier_curve.png", window);
            key = cv::waitKey(0);

            return 0;
        }

        cv::imshow("Bezier Curve", window);
        key = cv::waitKey(20);
    }

return 0;
}
