#include "Camera.h"




Camera::Camera(int width, int height, float3x3 K, float4 D) : width(width), height(height), K(K), D(D)
{
	// Define CUDA kernel launch configuration
	block_size = dim3(16, 16);
	grid_size = dim3((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

	K_inv = invert_intrinsic(K);

	pose = lre();

}

void Camera::render_scene(Scene& scene, uchar3* img_ptr, size_t pitch, bool synchronize)
{

	lre inv_pose = invert_lre(pose);

	render << <grid_size, block_size >> > (
		img_ptr,
		width,
		height,
		pitch,
		K_inv,
		pose,
		D,
		inv_pose,
		scene.d_mesh_instances,
		scene.num_mesh_instances,
		scene.d_meshes,
		scene.d_materials
		);

	if (synchronize)
		cudaDeviceSynchronize();

}
