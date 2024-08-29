#pragma once

#include <cuda_runtime.h>
#include "utils.hpp"
#include "Ray.hpp"

struct TrianglePrimitive {
    float3 vertices[3];
    float3 normal;
    uchar3 color;
	float min_x, max_x, min_y, max_y, min_z, max_z;


    // Constructor
    __host__ __device__ TrianglePrimitive(float3 a, float3 b, float3 c, uchar3 color)
        : color(color) {
        vertices[0] = a;
        vertices[1] = b;
        vertices[2] = c;

        float3 v0 = vertices[2] - vertices[0];
        float3 v1 = vertices[1] - vertices[0];
        normal = normalize(cross(v0, v1));

		min_x = fminf(fminf(vertices[0].x, vertices[1].x), vertices[2].x);
		max_x = fmaxf(fmaxf(vertices[0].x, vertices[1].x), vertices[2].x);
		min_y = fminf(fminf(vertices[0].y, vertices[1].y), vertices[2].y);
		max_y = fmaxf(fmaxf(vertices[0].y, vertices[1].y), vertices[2].y);
		min_z = fminf(fminf(vertices[0].z, vertices[1].z), vertices[2].z);
		max_z = fmaxf(fmaxf(vertices[0].z, vertices[1].z), vertices[2].z);
    }


    __host__ __device__ TrianglePrimitive(float3 a, float3 b, float3 c, float3 normal, uchar3 color)
        : normal(normal), color(color) {
        vertices[0] = a;
        vertices[1] = b;
        vertices[2] = c;

        min_x = fminf(fminf(vertices[0].x, vertices[1].x), vertices[2].x);
        max_x = fmaxf(fmaxf(vertices[0].x, vertices[1].x), vertices[2].x);
        min_y = fminf(fminf(vertices[0].y, vertices[1].y), vertices[2].y);
        max_y = fmaxf(fmaxf(vertices[0].y, vertices[1].y), vertices[2].y);
        min_z = fminf(fminf(vertices[0].z, vertices[1].z), vertices[2].z);
        max_z = fmaxf(fmaxf(vertices[0].z, vertices[1].z), vertices[2].z);
    }

    __host__ __device__ TrianglePrimitive() : normal(make_float3(0.0f, 0.0f, 0.0f)), color(make_uchar3(255, 255, 255)) {
        vertices[0] = make_float3(0.0f, 0.0f, 0.0f);
        vertices[1] = make_float3(0.0f, 0.0f, 0.0f);
        vertices[2] = make_float3(0.0f, 0.0f, 0.0f);

        min_x = fminf(fminf(vertices[0].x, vertices[1].x), vertices[2].x);
        max_x = fmaxf(fmaxf(vertices[0].x, vertices[1].x), vertices[2].x);
        min_y = fminf(fminf(vertices[0].y, vertices[1].y), vertices[2].y);
        max_y = fmaxf(fmaxf(vertices[0].y, vertices[1].y), vertices[2].y);
        min_z = fminf(fminf(vertices[0].z, vertices[1].z), vertices[2].z);
        max_z = fmaxf(fmaxf(vertices[0].z, vertices[1].z), vertices[2].z);
    }

    __host__ __device__ float3 ray_intersect(const Ray& ray) {

        float denom = dot(ray.direction, normal);

        if (abs(denom) < 1e-6) {
            return make_float3(0.0f, 0.0f, 0.0f);
        }

        float t = dot(vertices[0] - ray.origin, normal) / denom;

        if (t < 0.0f) {
            return make_float3(0.0f, 0.0f, 0.0f);
        }

        float3 point = ray.origin + t * ray.direction;

        return point;
    }

    __host__ __device__ float3 center() const {
        return (vertices[0] + vertices[1] + vertices[2]) / 3.0f;
    }


    __host__ __device__ bool point_inside(const float3& point) const {

        // Check bounding box first
		if (point.x < min_x || point.x > max_x || point.y < min_y || point.y > max_y || point.z < min_z || point.z > max_z) {
			return false;
		}

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

        return (u >= 0.0f) && (v >= 0.0f) && (u + v < 1.0f);
    }

};
