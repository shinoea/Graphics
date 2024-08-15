//
// Created by LEI XU on 5/13/19.
//
#pragma once
#ifndef RAYTRACING_OBJECT_H
#define RAYTRACING_OBJECT_H

#include "Vector.hpp"
#include "global.hpp"
#include "Bounds3.hpp"
#include "Ray.hpp"
#include "Intersection.hpp"

class Object
{
public:
    Object() {}
    virtual ~Object() {}
    virtual bool intersect(const Ray& ray) = 0;// �����ж�һ�������Ƿ���������ཻ
    // Ҳ�����ڼ�������������Ƿ��ཻ�����˺������᷵�ؽ���Ĳ�������ʾ���ཻ������
    virtual bool intersect(const Ray& ray, float &, uint32_t &) const = 0;
    virtual Intersection getIntersection(Ray _ray) = 0;// ���������������Ľ�����Ϣ
    // �ú������ڻ�ȡ�����������ԣ������ķ��ߡ����������
    virtual void getSurfaceProperties(const Vector3f &, const Vector3f &, const uint32_t &, const Vector2f &, Vector3f &, Vector2f &) const = 0;
    virtual Vector3f evalDiffuseColor(const Vector2f &) const =0;// �����������ض����������µ���������ɫ
    virtual Bounds3 getBounds()=0;// ��������İ�Χ��
    virtual float getArea()=0;// ��������ı������ÿһ����״�ļ��㷽�������Բ�һ��
    virtual void Sample(Intersection &pos, float &pdf)=0;// ������������һ���㣬���ڹ�Դ������`pos` �����ǲ��������Ϣ��`pdf` �Ǹõ�ĸ����ܶȺ���ֵ��
    virtual bool hasEmit()=0;// �жϸ������Ƿ񷢹⣬Ҳ�����Ƿ�Ϊ��Դ��
};



#endif //RAYTRACING_OBJECT_H
