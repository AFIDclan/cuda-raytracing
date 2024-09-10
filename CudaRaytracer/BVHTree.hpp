#pragma once

#include <cuda_runtime.h>
#include "utils.hpp"
#include "Ray.hpp"
#include <vector>
#include "TrianglePrimitive.hpp"
#include <cfloat>
#include <math.h>

struct SplitEvaluation
{
	float best_cost = FLT_MAX;
	float best_split = 0.0f;

};

struct d_BVHTree {

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

	__host__ __device__ float ray_intersects(const Ray& ray) {

		float3 tmin = (min - ray.origin) * ray.direction_inv;
		float3 tmax = (max - ray.origin) * ray.direction_inv;

		float3 t1 = f3_min(tmin, tmax);
		float3 t2 = f3_max(tmin, tmax);

		float dst_far = fminf(fminf(t2.x, t2.y), t2.z);
		float dst_near = fmaxf(fmaxf(t1.x, t1.y), t1.z);

		bool hit = dst_far >= dst_near && dst_far > 0.0f;

		return hit ? dst_near : FLT_MAX;
	}


};

struct BVHTree {
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
		
		// Only copy the triangle indices if this is a leaf node
		if (child_index_a != -1 || child_index_b != -1) {

			return d_BVHTree(min, max, child_index_a, child_index_b, nullptr, 0);
		}
		else {

			int* d_triangle_list;
			cudaMalloc(&d_triangle_list, triangle_indices.size() * sizeof(int));
			cudaMemcpy(d_triangle_list, triangle_indices.data(), triangle_indices.size() * sizeof(int), cudaMemcpyHostToDevice);

			return d_BVHTree(min, max, child_index_a, child_index_b, d_triangle_list, triangle_indices.size());
		}

	
	}


	void print_stats() {
		int count_nodes = 0;
		int max_triangles_per_node = 0;
		int min_triangles_per_node = 1000000;
		int max_depth = 0;
		int count_leaves = 0;

		

		std::vector<BVHTree*> stack;

		stack.push_back(this);


		while (stack.size() > 0) {
			BVHTree* node = stack.back();
			stack.pop_back();

			max_depth = fmaxf(max_depth, stack.size());

			count_nodes++;

			if (node->child_index_a == -1)
			{
				if (node->triangle_indices.size() > max_triangles_per_node) {
					max_triangles_per_node = node->triangle_indices.size();
				}

				if (node->triangle_indices.size() < min_triangles_per_node) {
					min_triangles_per_node = node->triangle_indices.size();
				}

				count_leaves++;

			}
			

			if (node->child_index_a != -1) {
				stack.push_back(master_list_trees->at(node->child_index_a));
				stack.push_back(master_list_trees->at(node->child_index_b));
			}

		}

		float avg_tris_per_leaf = (float)triangle_indices.size() / (float)count_leaves;

		std::cout << "BVH Stats: " << std::endl;
		std::cout << "Number of nodes: " << count_nodes << std::endl;
		std::cout << "Max triangles per node: " << max_triangles_per_node << std::endl;
		std::cout << "Min triangles per node: " << min_triangles_per_node << std::endl;
		std::cout << "Max depth: " << max_depth << std::endl;
		std::cout << "Number of leaves: " << count_leaves << std::endl;
		std::cout << "Average triangles per leaf: " << avg_tris_per_leaf << std::endl;


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

	float cost() {
		if (triangle_indices.size() == 0)
			return FLT_MAX;


		float3 size = max - min;
		
		float half_area = size.x * (size.y + size.z) + size.y * size.z;
		return half_area * triangle_indices.size();
	}

	void fill(int depth, int max_depth)
	{

		for (int i = 0; i < triangle_indices.size(); i++) {
			int idx = triangle_indices[i];
			grow_to_include(master_list_triangles[idx]);
		}

		if (depth >= max_depth) 
			return;

		if (triangle_indices.size() <= 1)
			return;


		std::pair<float, float> x_eval = evaluate_split("x");
		std::pair<float, float> y_eval = evaluate_split("y");
		std::pair<float, float> z_eval = evaluate_split("z");




		std::string axis;
		float split_pos;
		float best_cost = FLT_MAX;

		if (x_eval.first < y_eval.first && x_eval.first < z_eval.first) {
			axis = "x";
			split_pos = x_eval.second;
			best_cost = x_eval.first;
		}
		else if (y_eval.first < x_eval.first && y_eval.first < z_eval.first) {
			axis = "y";
			split_pos = y_eval.second;
			best_cost = y_eval.first;
		}
		else {
			axis = "z";
			split_pos = z_eval.second;
			best_cost = z_eval.first;
		}

		// Don't split if it will make things worse
		if (best_cost >= cost())
			return;


		std::vector<int> left_indices;
		std::vector<int> right_indices;

		for (int i = 0; i < triangle_indices.size(); i++) {
			int idx = triangle_indices[i];
			TrianglePrimitive triangle = master_list_triangles[idx];

			float3 triangle_center = triangle.center();

			//default to x
			float tri_check = triangle_center.x;

			if (axis == "y")
			{
				tri_check = triangle_center.y;
			}
			else if (axis == "z") {
				tri_check = triangle_center.z;
			}


			if (tri_check <= split_pos) {
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

	std::pair<float, float> evaluate_split(std::string axis)
	{

		float tests_per_axis = 5;
		float best_cost = FLT_MAX;
		float best_split = 0.0f;

		for (int s = 0; s < tests_per_axis; s++) {

			float split_t = ((float)s + 1) / (tests_per_axis + 1);

			float min_check = min.x;
			float max_check = max.x;

			if (axis == "y")
			{
				min_check = min.y;
				max_check = max.y;
			}
			else if (axis == "z") {
				min_check = min.z;
				max_check = max.z;
			}

			float pos = min_check + (max_check - min_check) * (split_t);


			BVHTree left;
			BVHTree right;

			for (int i = 0; i < triangle_indices.size(); i++) {
				int idx = triangle_indices[i];
				TrianglePrimitive triangle = master_list_triangles[idx];

				float3 triangle_center = triangle.center();

				//default to x
				float tri_check = triangle_center.x;

				if (axis == "y")
					tri_check = triangle_center.y;
				else if (axis == "z") 
					tri_check = triangle_center.z;
				

				if (tri_check <= pos) {
					left.grow_to_include(triangle);
					left.triangle_indices.push_back(idx);
				}
				else {
					right.grow_to_include(triangle);
					right.triangle_indices.push_back(idx);
				}

			}


			float cost = left.cost() + right.cost();

			if (cost < best_cost) {
				best_cost = cost;
				best_split = pos;
			}
		}

		return std::make_pair(best_cost, best_split);

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

