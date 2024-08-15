//
// Created by goksu on 4/6/19.
//

#pragma once

#include "Triangle.hpp"
#include <algorithm>
#include <Eigen/Dense>
#include <vector>
#include <map>
using namespace Eigen;

namespace rst {
enum class Buffers
{
    Color = 1,
    Depth = 2
};

inline Buffers operator|(Buffers a, Buffers b)//重载Buffers类的 | 运算符
{
    return Buffers((int)a | (int)b);
}

inline Buffers operator&(Buffers a, Buffers b)//重载Buffers类的 & 运算符
{
    return Buffers((int)a & (int)b);
}

enum class Primitive
{
    Line,
    Triangle
};

/*
 * For the curious : The draw function takes two buffer id's as its arguments.
 * These two structs make sure that if you mix up with their orders, the
 * compiler won't compile it. Aka : Type safety
 * 对于好奇的人来说:draw 函数接受两个缓冲区 ID 作为它的参数，
 * 这两个结构体确保了如果你弄混了它们的顺序,编译器就不会编译通过。这意味着提供了类型安全。
 * 也就是说,使用这种结构体,可以防止在调用 draw 函数时,不小心将颜色缓冲区和深度缓冲区的顺序搞混。
 * 编译器会检查参数类型,如果传入的不是正确的值,就会报错。
 * 如果使用的是两个int类型，在传入数据顺序错误时，仍会进行编译
 * */
struct pos_buf_id
{
    int pos_id = 0;
};

struct ind_buf_id
{
    int ind_id = 0;
};

class rasterizer
{
  public:
    rasterizer(int w, int h);
    pos_buf_id load_positions(const std::vector<Eigen::Vector3f>& positions);//三角形三个点
    ind_buf_id load_indices(const std::vector<Eigen::Vector3i>& indices);

    void set_model(const Eigen::Matrix4f& m);
    void set_view(const Eigen::Matrix4f& v);
    void set_projection(const Eigen::Matrix4f& p);//设置MVP矩阵

    //将屏幕像素点 (x, y) 设为(r, g, b) 的颜色，并写入相应的帧缓冲区位置。
    void set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color);

    void clear(Buffers buff);

    void draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, Primitive type);

    std::vector<Eigen::Vector3f>& frame_buffer() { return frame_buf; }

  private:
    void draw_line(Eigen::Vector3f begin, Eigen::Vector3f end);
    void rasterize_wireframe(const Triangle& t);

  private:
    Eigen::Matrix4f model;
    Eigen::Matrix4f view;
    Eigen::Matrix4f projection;

    std::map<int, std::vector<Eigen::Vector3f>> pos_buf;
    std::map<int, std::vector<Eigen::Vector3i>> ind_buf;

    std::vector<Eigen::Vector3f> frame_buf;
    std::vector<float> depth_buf;
    int get_index(int x, int y);

    int width, height;

    int next_id = 0;
    int get_next_id() { return next_id++; }
};
} // namespace rst
