#pragma once

#include <cuda_runtime.h>
#include "utils.hpp"

struct MeshInstance {

    int mesh_index;

    lre pose;
    lre inv_pose;

	float3 rotation;
	float3 inv_rotation;

    float3 scale;
    float3 inv_scale;

	__host__ __device__ MeshInstance() : mesh_index(-1) {}

	__host__ __device__ MeshInstance(int mesh_index)
		: mesh_index(mesh_index) {

		scale = make_float3(1.0f, 1.0f, 1.0f);

		this->build_inv();

	}

	__host__ __device__ MeshInstance(int mesh_index, lre pose, float3 scale)
		: mesh_index(mesh_index), pose(pose), scale(scale) {
		
		this->build_inv();

	}


	__host__ __device__ void build_inv() {

		inv_pose = invert_lre(this->pose);
		inv_scale = make_float3(1 / scale.x, 1 / scale.y, 1 / scale.z);

		rotation = make_float3(pose.yaw, pose.pitch, pose.roll);
		inv_rotation = make_float3(inv_pose.yaw, inv_pose.pitch, inv_pose.roll);
	}

};
