#include "MeshPrimitive.h"


MeshPrimitive::MeshPrimitive(std::vector<TrianglePrimitive> triangles)
{
	this->triangles = new TrianglePrimitive[triangles.size()];
	this->num_triangles = triangles.size();

	this->world_triangles = new TrianglePrimitive[this->num_triangles];

	for (int i = 0; i < triangles.size(); i++) {
		this->triangles[i] = triangles[i];
	}

	this->pose = lre();

	this->genarate_world_triangles();
}

void MeshPrimitive::set_world_rotation(float3 rotation)
{
	this->pose.yaw = rotation.x;
	this->pose.pitch = rotation.y;
	this->pose.roll = rotation.z;

	this->genarate_world_triangles();
}

void MeshPrimitive::genarate_world_triangles()
{

	lre local2world = invert_lre(this->pose);

	for (int i = 0; i < this->num_triangles; i++) {
		TrianglePrimitive triangle = this->triangles[i];


		float3 a = apply_lre(local2world, triangle.vertices[0]);
		float3 b = apply_lre(local2world, triangle.vertices[1]);
		float3 c = apply_lre(local2world, triangle.vertices[2]);

		float3 normal = apply_rotmat(euler2rotmat(make_float3(this->pose.yaw, this->pose.pitch, this->pose.roll)), triangle.normal);

		this->world_triangles[i] = TrianglePrimitive(a, b, c, normal, triangle.color);
	}

}
