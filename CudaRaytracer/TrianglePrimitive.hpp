#pragma once

#include <cuda_runtime.h>
#include "utils.hpp"
#include "Ray.hpp"

struct TrianglePrimitive {
    float3 vertices[3];
    float3 normal;
	float2 uv_coords[3];


    // Constructor
    __host__ __device__ TrianglePrimitive(float3 a, float3 b, float3 c) {
        vertices[0] = a;
        vertices[1] = b;
        vertices[2] = c;

        float3 v0 = vertices[1] - vertices[0];
        float3 v1 = vertices[2] - vertices[0];
        normal = normalize(cross(v0, v1));
    }


    __host__ __device__ TrianglePrimitive(float3 a, float3 b, float3 c, float3 normal)
        : normal(normal) {
        vertices[0] = a;
        vertices[1] = b;
        vertices[2] = c;

        float3 v0 = vertices[2] - vertices[0];
        float3 v1 = vertices[1] - vertices[0];

    }

    __host__ __device__ TrianglePrimitive(float3 a, float3 b, float3 c, float3 normal, float2 uv_a, float2 uv_b, float2 uv_c)
        : normal(normal) {
        vertices[0] = a;
        vertices[1] = b;
        vertices[2] = c;

        float3 v0 = vertices[2] - vertices[0];
        float3 v1 = vertices[1] - vertices[0];


        uv_coords[0] = uv_a;
		uv_coords[1] = uv_b;
		uv_coords[2] = uv_c;
    }

    __host__ __device__ TrianglePrimitive() : normal(make_float3(0.0f, 0.0f, 0.0f)) {
        vertices[0] = make_float3(0.0f, 0.0f, 0.0f);
        vertices[1] = make_float3(0.0f, 0.0f, 0.0f);
        vertices[2] = make_float3(0.0f, 0.0f, 0.0f);

        float3 v0 = vertices[2] - vertices[0];
        float3 v1 = vertices[1] - vertices[0];

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


    __host__ __device__ float2 ray_inside(const Ray& ray, float& t)
    {
        float3 v0v1 = vertices[1] - vertices[0];
        float3 v0v2 = vertices[2] - vertices[0];

		float3 N = cross(v0v1, v0v2);

		float area = magnitude(N);

		float N_dot_ray_dir = dot(N, ray.direction);

        if (abs(N_dot_ray_dir) < 1e-6) 
			return make_float2(FLT_MAX, FLT_MAX);
        
		float d = -dot(N, vertices[0]);

		t = -(dot(N, ray.origin) + d) / N_dot_ray_dir;

		if (t < 0.0f)
			return make_float2(FLT_MAX, FLT_MAX);

		float3 P = ray.origin + t * ray.direction;

		float3 C;

		float3 edge0 = vertices[1] - vertices[0];
		float3 vp0 = P - vertices[0];

		C = cross(edge0, vp0);

		if (dot(N, C) < 0.0f)
			return make_float2(FLT_MAX, FLT_MAX);

		float3 edge1 = vertices[2] - vertices[1];
		float3 vp1 = P - vertices[1];

		C = cross(edge1, vp1);
		float u = magnitude(C) / area;

		if (dot(N, C) < 0.0f)
			return make_float2(FLT_MAX, FLT_MAX);

		float3 edge2 = vertices[0] - vertices[2];
		float3 vp2 = P - vertices[2];

		C = cross(edge2, vp2);
		float v = magnitude(C) / area;

		if (dot(N, C) < 0.0f)
			return make_float2(FLT_MAX, FLT_MAX);

        // Calculate barycentric weight for the third vertex
        float w = 1.0f - u - v;

        // Interpolate the UV coordinates
        float2 uv0 = uv_coords[0];
        float2 uv1 = uv_coords[1];
        float2 uv2 = uv_coords[2];

        float2 interpolated_uv = (u * uv0) + (v * uv1) + (w * uv2);

		return interpolated_uv;

    }

    __host__ __device__ float2 point_inside(const float3& point) const {


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

        // If point is inside the triangle, interpolate UV coordinates
        if ((u >= 0.0f) && (v >= 0.0f) && (u + v <= 1.0f)) {

            // Calculate barycentric weight for the third vertex
            float w = 1.0f - u - v;

            // Interpolate the UV coordinates
            float2 uv0 = uv_coords[0];
            float2 uv1 = uv_coords[1];
            float2 uv2 = uv_coords[2];

            float2 interpolated_uv = (u * uv0) + (v * uv1) + (w * uv2);

            return interpolated_uv;
        }

        return make_float2(FLT_MAX, FLT_MAX);
    }

};
