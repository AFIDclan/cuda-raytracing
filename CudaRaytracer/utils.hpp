#pragma once

#include <cuda_runtime.h>
#include "transforms.hpp"

using namespace transforms;

static __host__ __device__ float magnitude(float3 v) {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

static __host__ __device__ float3 normalize(float3 v) {
	float mag = magnitude(v);
	return make_float3(v.x / mag, v.y / mag, v.z / mag);
}

static __host__ __device__ float3 cross(float3 a, float3 b) {
	return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

static __host__ __device__ float dot(float3 a, float3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

static __host__ __device__ float3 operator+(float3 a, float3 b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static __host__ __device__ float3 operator-(float3 a, float3 b) {
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static __host__ __device__ float3 operator*(float3 a, float b) {
	return make_float3(a.x * b, a.y * b, a.z * b);
}

static __host__ __device__ float3 operator*(float b, float3 a) {
	return make_float3(a.x * b, a.y * b, a.z * b);
}

static __host__ __device__ float3 operator/(float3 a, float b) {
	return make_float3(a.x / b, a.y / b, a.z / b);
}

template <typename T>
__host__ __device__ void cu_swap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

static __host__ __device__ float4 apply_matrix(const float4x4& matrix, const float4& vec) {
    float4 result;
    result.x = matrix.m[0][0] * vec.x + matrix.m[0][1] * vec.y + matrix.m[0][2] * vec.z + matrix.m[0][3] * vec.w;
    result.y = matrix.m[1][0] * vec.x + matrix.m[1][1] * vec.y + matrix.m[1][2] * vec.z + matrix.m[1][3] * vec.w;
    result.z = matrix.m[2][0] * vec.x + matrix.m[2][1] * vec.y + matrix.m[2][2] * vec.z + matrix.m[2][3] * vec.w;
    result.w = matrix.m[3][0] * vec.x + matrix.m[3][1] * vec.y + matrix.m[3][2] * vec.z + matrix.m[3][3] * vec.w;
    return result;
}

static __host__ __device__ float3 apply_matrix(const float3x3& matrix, const float3& vec) {
    float3 result;
    result.x = matrix.m[0][0] * vec.x + matrix.m[0][1] * vec.y + matrix.m[0][2] * vec.z;
    result.y = matrix.m[1][0] * vec.x + matrix.m[1][1] * vec.y + matrix.m[1][2] * vec.z;
    result.z = matrix.m[2][0] * vec.x + matrix.m[2][1] * vec.y + matrix.m[2][2] * vec.z;
    return result;
}

static __host__ __device__ float3x3 invert_intrinsic(const float3x3& K) {
    float fx_inv = 1.0f / K.m[0][0];
    float fy_inv = 1.0f / K.m[1][1];
    float cx = K.m[0][2];
    float cy = K.m[1][2];

    float3x3 K_inv;
    K_inv.m[0][0] = fx_inv;
    K_inv.m[0][1] = 0.0f;
    K_inv.m[0][2] = -cx * fx_inv;
    K_inv.m[1][0] = 0.0f;
    K_inv.m[1][1] = fy_inv;
    K_inv.m[1][2] = -cy * fy_inv;
    K_inv.m[2][0] = 0.0f;
    K_inv.m[2][1] = 0.0f;
    K_inv.m[2][2] = 1.0f;

    return K_inv;
}
