#pragma once

#include <cuda_runtime.h>
#include "utils.hpp"
#include "Ray.hpp"
#include <vector>
#include "TrianglePrimitive.hpp"


static struct d_BVHTree {

	float3 min;
	float3 max;
	int* triangle_indices;
	int count_triangles;

	int child_index_a;
	int child_index_b;

	__host__ __device__ d_BVHTree() {
		min.x = FLT_MAX;
		min.y = FLT_MAX;
		min.z = FLT_MAX;

		max.x = -FLT_MAX;
		max.y = -FLT_MAX;
		max.z = -FLT_MAX;
	}

	__host__ __device__ d_BVHTree(float3 min, float3 max, int child_index_a, int child_index_b, int* triangle_indices, int count_triangles) : min(min), max(max), child_index_a(child_index_a), child_index_b(child_index_b), triangle_indices(triangle_indices), count_triangles(count_triangles) {}

	__host__ __device__ bool ray_intersects(const Ray& ray) {
		float tmin = (min.x - ray.origin.x) / ray.direction.x;
		float tmax = (max.x - ray.origin.x) / ray.direction.x;


		if (tmin > tmax)
		{
			float temp = tmin;
			tmin = tmax;
			tmax = temp;
		}

		float tymin = (min.y - ray.origin.y) / ray.direction.y;
		float tymax = (max.y - ray.origin.y) / ray.direction.y;

		if (tymin > tymax)
		{
			float temp = tymin;
			tymin = tymax;
			tymax = temp;
		}

		// Check for intersection failures
		if ((tmin > tymax) || (tymin > tmax))
			return false;

		// Update tmin and tmax to account for y-axis intersection
		if (tymin > tmin)
			tmin = tymin;
		if (tymax < tmax)
			tmax = tymax;

		float tzmin = (min.z - ray.origin.z) / ray.direction.z;
		float tzmax = (max.z - ray.origin.z) / ray.direction.z;

		if (tzmin > tzmax)
		{
			float temp = tzmin;
			tzmin = tzmax;
			tzmax = temp;
		}

		// Final intersection check
		if ((tmin > tzmax) || (tzmin > tmax))
			return false;

		// Update tmin and tmax to account for z-axis intersection
		if (tzmin > tmin)
			tmin = tzmin;
		if (tzmax < tmax)
			tmax = tzmax;

		return (tmax >= tmin);
	}


};

static struct BVHTree {
	float3 min;
	float3 max;

	std::vector<int> triangle_indices;


	int child_index_a = -1;
	int child_index_b = -1;

	std::vector<BVHTree*>* master_list_trees;
	TrianglePrimitive* master_list_triangles;


	BVHTree() {
		min.x = FLT_MAX;
		min.y = FLT_MAX;
		min.z = FLT_MAX;

		max.x = -FLT_MAX;
		max.y = -FLT_MAX;
		max.z = -FLT_MAX;
	}

	BVHTree(std::vector<BVHTree*>* master_list_trees, TrianglePrimitive* master_list_triangles, std::vector<int> &triangle_indices) :
		master_list_trees(master_list_trees), 
		master_list_triangles(master_list_triangles),
		triangle_indices(triangle_indices) {

		min.x = FLT_MAX;
		min.y = FLT_MAX;
		min.z = FLT_MAX;

		max.x = -FLT_MAX;
		max.y = -FLT_MAX;
		max.z = -FLT_MAX;
	}

	d_BVHTree to_device_compatible() {
		int* d_triangle_list;
		cudaMalloc(&d_triangle_list, triangle_indices.size() * sizeof(int));
		cudaMemcpy(d_triangle_list, triangle_indices.data(), triangle_indices.size() * sizeof(int), cudaMemcpyHostToDevice);

		return d_BVHTree(min, max, child_index_a, child_index_b, d_triangle_list, triangle_indices.size());
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

	void fill(int depth, int max_depth)
	{

		for (int i = 0; i < triangle_indices.size(); i++) {
			int idx = triangle_indices[i];
			grow_to_include(master_list_triangles[idx]);
		}

		if (depth >= max_depth) 
			return;
		


		float3 mid = (min + max) / 2.0f;

		std::vector<int> left_indices;
		std::vector<int> right_indices;

		for (int i = 0; i < triangle_indices.size(); i++) {
			int idx = triangle_indices[i];
			TrianglePrimitive triangle = master_list_triangles[idx];

			float3 triangle_center = triangle.center();

			if (triangle_center.x < mid.x) {
				left_indices.push_back(idx);
			}
			else {
				right_indices.push_back(idx);
			}
		}

		if (left_indices.size() == 0 || right_indices.size() == 0) 
			return;
		

		child_index_a = master_list_trees->size();
		master_list_trees->push_back(new BVHTree(master_list_trees, master_list_triangles, left_indices));
		master_list_trees->at(child_index_a)->fill(depth + 1, max_depth);

		child_index_b = master_list_trees->size();
		master_list_trees->push_back(new BVHTree(master_list_trees, master_list_triangles, right_indices));
		master_list_trees->at(child_index_b)->fill(depth + 1, max_depth);


	}

	static d_BVHTree* compile_tree(BVHTree& top) {
		// Allocate host memory for an array of device-compatible d_BVHTree structures
		d_BVHTree* host_device_tree = new d_BVHTree[top.master_list_trees->size()];

		for (int i = 0; i < top.master_list_trees->size(); i++) {
			host_device_tree[i] = top.master_list_trees->at(i)->to_device_compatible();
		}


		d_BVHTree* d_device_tree;
		cudaMalloc(&d_device_tree, top.master_list_trees->size() * sizeof(d_BVHTree));

		cudaMemcpy(d_device_tree, host_device_tree, top.master_list_trees->size() * sizeof(d_BVHTree), cudaMemcpyHostToDevice);

		// Clean up host memory
		delete[] host_device_tree;

		// Return the device-compatible array of d_BVHTree structures
		return d_device_tree;
	}

};

