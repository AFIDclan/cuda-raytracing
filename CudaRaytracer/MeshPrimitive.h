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
	

	__host__ __device__ d_MeshPrimitive() {
		num_triangles = 0;
		triangles = nullptr;
	}

	__host__ __device__ d_MeshPrimitive(int num_triangles, TrianglePrimitive* triangles, d_BVHTree* bvh_top) : num_triangles(num_triangles), triangles(triangles), bvh_top(bvh_top) {}
};

class MeshPrimitive
{

public:
	MeshPrimitive(std::vector<TrianglePrimitive> triangles);
	
	int num_triangles;
	BVHTree bvh_top;

	d_MeshPrimitive* to_device();


private:
	void build_bvh();

	TrianglePrimitive* triangles;
};

