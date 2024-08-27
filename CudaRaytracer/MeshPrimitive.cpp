#include "MeshPrimitive.h"


MeshPrimitive::MeshPrimitive(std::vector<TrianglePrimitive> triangles)
{
	this->triangles = new TrianglePrimitive[triangles.size()];
	this->num_triangles = triangles.size();

	this->world_triangles = new TrianglePrimitive[this->num_triangles];

	for (int i = 0; i < triangles.size(); i++) {
		this->triangles[i] = triangles[i];
	}

	this->H_pose = Matrix4d::Identity();

	this->genarate_world_triangles();
}

void MeshPrimitive::set_world_rotation(AngleAxisd rotation)
{
	Matrix3d new_rotation = rotation.toRotationMatrix();
	Matrix3d rotation_inv = new_rotation.inverse();

	this->H_pose.block<3, 3>(0, 0) = new_rotation;

	// Apply the rotation to 4th column of the pose matrix
	this->H_pose.block<3, 1>(0, 3) = rotation_inv * this->H_pose.block<3, 1>(0, 3);


	this->genarate_world_triangles();
}

void MeshPrimitive::genarate_world_triangles()
{

	

	Eigen::Matrix4d H_pose_inv = this->H_pose.inverse();
	Eigen::Matrix3d rotmat_pose_inv = H_pose_inv.block<3, 3>(0, 0);

	float4x4 H_pose_inv_float = eigen_mat_to_float(H_pose_inv);

	for (int i = 0; i < this->num_triangles; i++) {
		TrianglePrimitive triangle = this->triangles[i];


		float4 a = apply_matrix(H_pose_inv_float, make_float4(triangle.vertices[0].x, triangle.vertices[0].y, triangle.vertices[0].z, 1.0f));
		float4 b = apply_matrix(H_pose_inv_float, make_float4(triangle.vertices[1].x, triangle.vertices[1].y, triangle.vertices[1].z, 1.0f));
		float4 c = apply_matrix(H_pose_inv_float, make_float4(triangle.vertices[2].x, triangle.vertices[2].y, triangle.vertices[2].z, 1.0f));

		float3 normal = apply_matrix(eigen_mat_to_float(rotmat_pose_inv), triangle.normal);

		this->world_triangles[i] = TrianglePrimitive(make_float3(a.x, a.y, a.z), make_float3(b.x, b.y, b.z), make_float3(c.x, c.y, c.z), normal, triangle.color);
	}

}
