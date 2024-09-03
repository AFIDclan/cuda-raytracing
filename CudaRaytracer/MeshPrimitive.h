#pragma once

#include <vector>
#include "TrianglePrimitive.hpp"
#include "transforms.hpp"
#include <cuda_runtime.h>
#include "utils.hpp"
#include "BVHTree.hpp"

using namespace transforms;


struct d_MeshPrimitive {
	int num_triangles;
	TrianglePrimitive* triangles;
	d_BVHTree* bvh_top;
	lre pose;
	lre inv_pose;

	float3 scale;
	float3 inv_scale;

	float3 rotation;
	float3 inv_rotation;

	__host__ __device__ d_MeshPrimitive() {
		num_triangles = 0;
		triangles = nullptr;
	}

	__host__ __device__ d_MeshPrimitive(int num_triangles, TrianglePrimitive* triangles, d_BVHTree* bvh_top, lre pose) : num_triangles(num_triangles), triangles(triangles), bvh_top(bvh_top), pose(pose) {
		rotation = make_float3(pose.yaw, pose.pitch, pose.roll);

		inv_pose = invert_lre(this->pose);
		inv_rotation = make_float3(inv_pose.yaw, inv_pose.pitch, inv_pose.roll);

		scale = make_float3(1.0, 1.0, 1.0);
		//scale = make_float3(0.5, 0.5, 0.5);
		inv_scale = make_float3(1 / scale.x, 1 / scale.y, 1 / scale.z);


	}
};

class MeshPrimitive
{

public:
	MeshPrimitive(std::vector<TrianglePrimitive> triangles);
	

	
	int num_triangles;
	BVHTree bvh_top;

	void set_world_rotation(float3 rotation);
	void set_world_position(float3 position);

	d_MeshPrimitive* to_device();


private:
	void build_bvh();

	TrianglePrimitive* triangles;
	lre pose;

};

