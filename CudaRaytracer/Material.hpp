
#pragma once
#include <cuda_runtime.h>

struct Material
{
	float roughness;
	float3 albedo;
	float metallic;
	float illumination;

	__host__ __device__ Material() : roughness(0.0f), albedo(make_float3(1.0f, 1.0f, 1.0f)), metallic(0.0f), illumination(0.0f) {}

	__host__ __device__ Material* to_device()
	{
		Material* device_material;
		cudaMalloc(&device_material, sizeof(Material));
		cudaMemcpy(device_material, this, sizeof(Material), cudaMemcpyHostToDevice);
		return device_material;
	}
};