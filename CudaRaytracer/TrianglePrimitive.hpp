#pragma once

#include <cuda_runtime.h>
#include "utils.hpp"
#include "Ray.hpp"

struct TrianglePrimitive {
    float3 vertices[3];
    float3 normal;
	float area;


    // Constructor
    __host__ __device__ TrianglePrimitive(float3 a, float3 b, float3 c) {
        vertices[0] = a;
        vertices[1] = b;
        vertices[2] = c;

        float3 v0 = vertices[1] - vertices[0];
        float3 v1 = vertices[2] - vertices[0];
        normal = normalize(cross(v0, v1));

		area = 0.5f * magnitude(cross(v0, v1));
    }


    __host__ __device__ TrianglePrimitive(float3 a, float3 b, float3 c, float3 normal)
        : normal(normal) {
        vertices[0] = a;
        vertices[1] = b;
        vertices[2] = c;

        float3 v0 = vertices[2] - vertices[0];
        float3 v1 = vertices[1] - vertices[0];

        area = 0.5f * magnitude(cross(v0, v1));
    }

    __host__ __device__ TrianglePrimitive() : normal(make_float3(0.0f, 0.0f, 0.0f)) {
        vertices[0] = make_float3(0.0f, 0.0f, 0.0f);
        vertices[1] = make_float3(0.0f, 0.0f, 0.0f);
        vertices[2] = make_float3(0.0f, 0.0f, 0.0f);

        float3 v0 = vertices[2] - vertices[0];
        float3 v1 = vertices[1] - vertices[0];

        area = 0.5f * magnitude(cross(v0, v1));
    }

    __host__ __device__ float3 ray_intersect(const Ray& ray) {

        float denom = dot(ray.direction, normal);

        if (abs(denom) < 1e-6) {
            return make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
        }

        float t = dot(vertices[0] - ray.origin, normal) / denom;

        if (t < 0.0f) {
            return make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
        }

        float3 point = ray.origin + t * ray.direction;

        return point;
    }

    __host__ __device__ float3 center() const {
        return (vertices[0] + vertices[1] + vertices[2]) / 3.0f;
    }


    __host__ __device__ bool point_inside(const float3& point) const {

        float3 v0 = vertices[2] - vertices[0];
        float3 v1 = vertices[1] - vertices[0];
        float3 v2 = point - vertices[0];

        float dot00 = dot(v0, v0);
        float dot01 = dot(v0, v1);
        float dot02 = dot(v0, v2);
        float dot11 = dot(v1, v1);
        float dot12 = dot(v1, v2);

        float invDenom = 1.0f / (dot00 * dot11 - dot01 * dot01);
        float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
        float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

        return (u >= 0.0f) && (v >= 0.0f) && (u + v <= 1.0f);
    }

};
