#pragma once

#include <cuda_runtime.h>

struct Ray {
    float3 origin;
    float3 direction;
    uint2 pixel;
    uchar3 color;
    float illumination;

    // Constructor
    __device__ Ray(float3 o, float3 d, uint2 p)
        : origin(o), direction(d), pixel(p), color(make_uchar3(255, 255, 255)), illumination(0.0f) {}
};


