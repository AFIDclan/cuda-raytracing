
#pragma once
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

struct Material
{
	float roughness;
	float3 albedo;
	float metallic;
	float illumination;

	uchar3* texture;
	int texture_width;
	int texture_height;
	size_t texture_pitch;


	__host__ __device__ Material() : roughness(0.0f), albedo(make_float3(1.0f, 1.0f, 1.0f)), metallic(0.0f), illumination(0.0f) {}

	__host__ __device__ Material* to_device()
	{
		Material* device_material;
		cudaMalloc(&device_material, sizeof(Material));
		cudaMemcpy(device_material, this, sizeof(Material), cudaMemcpyHostToDevice);
		return device_material;
	}

	void upload_texture(std::string fp)
	{

		cv::Mat img_cpu = cv::imread(fp, cv::IMREAD_COLOR);

		texture_width = img_cpu.cols;
		texture_height = img_cpu.rows;

		cudaMallocPitch(&texture, &texture_pitch, texture_width * sizeof(uchar3), texture_height);

		cv::cuda::GpuMat img_gpu(texture_height, texture_width, CV_8UC3, texture, texture_pitch);

		img_gpu.upload(img_cpu);

	}
};