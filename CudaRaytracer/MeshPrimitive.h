#pragma once

#include <vector>
#include "TrianglePrimitive.hpp"
#include <Eigen/Dense>

using namespace Eigen;


class MeshPrimitive
{

public:
	MeshPrimitive(std::vector<TrianglePrimitive> triangles);
	

	
	int num_triangles;
	TrianglePrimitive* world_triangles;

	void set_world_rotation(AngleAxisd rotation);


private:
	void genarate_world_triangles();

	TrianglePrimitive* triangles;
	Matrix4d H_pose;

};

