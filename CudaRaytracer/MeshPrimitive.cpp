#include "MeshPrimitive.h"



MeshPrimitive::MeshPrimitive(std::vector<TrianglePrimitive> triangles)
{
	this->triangles = new TrianglePrimitive[triangles.size()];
	this->num_triangles = triangles.size();

	for (int i = 0; i < triangles.size(); i++) {
		this->triangles[i] = triangles[i];
	}

	this->build_bvh();
}

d_MeshPrimitive* MeshPrimitive::to_device()
{

	d_BVHTree* d_bvh_tree = BVHTree::compile_tree(this->bvh_top);

	TrianglePrimitive* d_triangles;
	cudaMalloc(&d_triangles, this->num_triangles * sizeof(TrianglePrimitive));
	cudaMemcpy(d_triangles, this->triangles, this->num_triangles * sizeof(TrianglePrimitive), cudaMemcpyHostToDevice);

	d_MeshPrimitive* host_mesh = new d_MeshPrimitive(this->num_triangles, d_triangles, d_bvh_tree);

	d_MeshPrimitive* d_mesh;

	cudaMalloc(&d_mesh, sizeof(d_MeshPrimitive));
	cudaMemcpy(d_mesh, host_mesh, sizeof(d_MeshPrimitive), cudaMemcpyHostToDevice);

	delete host_mesh;

	return d_mesh;
}

void MeshPrimitive::build_bvh()
{

	std::vector<BVHTree*>* master_list_trees = new std::vector<BVHTree*>();

	std::vector<int> triangle_indices;

	for (int i = 0; i < this->num_triangles; i++) {
		triangle_indices.push_back(i);
	}

	this->bvh_top = BVHTree(master_list_trees, this->triangles, triangle_indices);

	master_list_trees->push_back(&this->bvh_top);

	// Recursively build the BVH tree
	this->bvh_top.fill(1, 32);

}
