#pragma once

#include <vector>
#include "TrianglePrimitive.hpp"
#include "transforms.hpp"
#include <cuda_runtime.h>
#include "utils.hpp"

using namespace transforms;

static struct d_BVHBoundingBox {

	float3 min;
	float3 max;
	int* triangle_indices;
	int count_triangles;

	__host__ __device__ d_BVHBoundingBox() : min(make_float3(0.0f, 0.0f, 0.0f)), max(make_float3(0.0f, 0.0f, 0.0f)) {}

	__host__ __device__ d_BVHBoundingBox(float3 min, float3 max, int* triangle_indices, int count_triangles) : min(min), max(max), triangle_indices(triangle_indices), count_triangles(count_triangles) {}

	__host__ __device__ bool ray_intersects(const Ray& ray) {
		float tmin = (min.x - ray.origin.x) / ray.direction.x;
		float tmax = (max.x - ray.origin.x) / ray.direction.x;

		if (tmin > tmax) cu_swap(tmin, tmax);

		float tymin = (min.y - ray.origin.y) / ray.direction.y;
		float tymax = (max.y - ray.origin.y) / ray.direction.y;

		if (tymin > tymax) cu_swap(tymin, tymax);

		if ((tmin > tymax) || (tymin > tmax))
			return false;

		if (tymin > tmin)
			tmin = tymin;

		if (tymax < tmax)
			tmax = tymax;

		float tzmin = (min.z - ray.origin.z) / ray.direction.z;
		float tzmax = (max.z - ray.origin.z) / ray.direction.z;

		if (tzmin > tzmax) cu_swap(tzmin, tzmax);

		if ((tmin > tzmax) || (tzmin > tmax))
			return false;

		if (tzmin > tmin)
			tmin = tzmin;

		if (tzmax < tmax)
			tmax = tzmax;

		return (tmax >= tmin);
	}

};

static struct BVHBoundingBox {
	float3 min;
	float3 max;
	std::vector<int> triangle_indices;


	BVHBoundingBox() : min(make_float3(0.0f, 0.0f, 0.0f)), max(make_float3(0.0f, 0.0f, 0.0f)) {}
	BVHBoundingBox(TrianglePrimitive* triangle_list, int triangle_count) {

		for (int i = 0; i < triangle_count; i++) {
			triangle_indices.push_back(i);
			grow_to_include(triangle_list[i]);
		}
	}

	d_BVHBoundingBox* to_device_compatible() {
		int* d_triangle_list;
		cudaMalloc(&d_triangle_list, triangle_indices.size() * sizeof(int));
		cudaMemcpy(d_triangle_list, triangle_indices.data(), triangle_indices.size() * sizeof(int), cudaMemcpyHostToDevice);

		// Allocate and return the device-compatible struct
		d_BVHBoundingBox* d_bvh_bbox;
		cudaMalloc(&d_bvh_bbox, sizeof(d_BVHBoundingBox));
		cudaMemcpy(d_bvh_bbox, new d_BVHBoundingBox(min, max, d_triangle_list, triangle_indices.size()), sizeof(d_BVHBoundingBox), cudaMemcpyHostToDevice);

		return d_bvh_bbox;
	}


	void grow_to_include(TrianglePrimitive& triangle) {

		for (int i = 0; i < 3; i++) {
			grow_to_include(triangle.vertices[i]);
		}
	}

	void grow_to_include(float3 vertex) {
		min.x = fminf(min.x, vertex.x);
		min.y = fminf(min.y, vertex.y);
		min.z = fminf(min.z, vertex.z);

		max.x = fmaxf(max.x, vertex.x);
		max.y = fmaxf(max.y, vertex.y);
		max.z = fmaxf(max.z, vertex.z);
	}

};

class MeshPrimitive
{

public:
	MeshPrimitive(std::vector<TrianglePrimitive> triangles);
	

	
	int num_triangles;
	TrianglePrimitive* world_triangles;
	BVHBoundingBox bvh_top;

	void set_world_rotation(float3 rotation);
	void set_world_position(float3 position);


private:
	void genarate_world_triangles();

	TrianglePrimitive* triangles;
	lre pose;

};

