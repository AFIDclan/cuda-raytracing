#pragma once

#include <vector>
#include "TrianglePrimitive.hpp"
#include "transforms.hpp"
#include <cuda_runtime.h>

using namespace transforms;


class MeshPrimitive
{

public:
	MeshPrimitive(std::vector<TrianglePrimitive> triangles);
	

	
	int num_triangles;
	TrianglePrimitive* world_triangles;

	void set_world_rotation(float3 rotation);


private:
	void genarate_world_triangles();

	TrianglePrimitive* triangles;
	lre pose;

};

