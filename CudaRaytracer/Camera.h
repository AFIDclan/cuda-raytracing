#pragma once

#include <cuda_runtime.h>
#include "transforms.hpp"
#include "Scene.h"
#include "raycast.h"

using namespace transforms;

class Camera
{
	
public:

	Camera(int width, int height, float3x3 K, float4 D);

	lre pose;
	int width;
	int height;

	float3x3 K;
	float3x3 K_inv;
	float4 D;

	void render_scene(Scene& scene, uchar3* img_ptr, size_t pitch, bool synchronize = false);

private:
	dim3 block_size;
	dim3 grid_size;
};

