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

		if (tmin > tmax) cu_swap(tmin, tmax);

		float tymin = (min.y - ray.origin.y) / ray.direction.y;
		float tymax = (max.y - ray.origin.y) / ray.direction.y;

		if (tymin > tymax) cu_swap(tymin, tymax);

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

		if (tzmin > tzmax) cu_swap(tzmin, tzmax);

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

		std::string check = "x";

		if (max.y - min.y > max.x - min.x && max.y - min.y > max.z - min.z) {
			check = "y";
		}

		else if (max.z - min.z > max.x - min.x && max.z - min.z > max.y - min.y) {
			check = "z";
		}

		

		for (int i = 0; i < triangle_indices.size(); i++) {
			int idx = triangle_indices[i];
			TrianglePrimitive triangle = master_list_triangles[idx];

			float3 triangle_center = triangle.center();

			//default to x
			float mid_check = mid.x;

			float vert_0_check = triangle.vertices[0].x;
			float vert_1_check = triangle.vertices[1].x;
			float vert_2_check = triangle.vertices[2].x;

			if (check == "y")
			{
				mid_check = mid.y;

				vert_0_check = triangle.vertices[0].y;
				vert_1_check = triangle.vertices[1].y;
				vert_2_check = triangle.vertices[2].y;
			}
			else if (check == "z") {
				mid_check = mid.z;

				vert_0_check = triangle.vertices[0].z;
				vert_1_check = triangle.vertices[1].z;
				vert_2_check = triangle.vertices[2].z;
			}



			// We want triangles to be in both children if they are on the boundary
			if (vert_0_check <= mid_check || vert_1_check <= mid_check || vert_2_check <= mid_check) {
				left_indices.push_back(idx);
			}


			if (vert_0_check >= mid_check || vert_1_check >= mid_check || vert_2_check >= mid_check) {
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

