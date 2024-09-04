#include "Scene.h"


Scene::Scene()
{


}

void Scene::add_material(Material material)
{
	materials.push_back(material);
}

void Scene::add_mesh(MeshPrimitive mesh)
{
	meshes.push_back(mesh);
}

void Scene::add_mesh_instance(MeshInstance mesh_instance)
{
	mesh_instances.push_back(mesh_instance);
}

void Scene::upload_to_device()
{

	// Free any existing memory
	if (d_materials != nullptr) {
		cudaFree(d_materials);
	}

	if (d_meshes != nullptr) {
		cudaFree(d_meshes);
	}

	if (d_mesh_instances != nullptr) {
		cudaFree(d_mesh_instances);
	}

	cudaMalloc(&d_materials, sizeof(Material) * materials.size());

	for (int i = 0; i < materials.size(); i++) {
		cudaMemcpy(&d_materials[i], materials[i].to_device(), sizeof(Material), cudaMemcpyHostToDevice);
	}


	cudaMalloc(&d_meshes, sizeof(d_MeshPrimitive) * meshes.size());

	for (int i = 0; i < meshes.size(); i++) {
		cudaMemcpy(&d_meshes[i], meshes[i].to_device(), sizeof(d_MeshPrimitive), cudaMemcpyHostToDevice);
	}


	cudaMalloc(&d_mesh_instances, sizeof(MeshInstance) * mesh_instances.size());

	for (int i = 0; i < mesh_instances.size(); i++) {

		mesh_instances[i].build_inv();

		cudaMemcpy(&d_mesh_instances[i], &mesh_instances[i], sizeof(MeshInstance), cudaMemcpyHostToDevice);
	}

	num_mesh_instances = mesh_instances.size();
}

void Scene::update_mesh_instance(int index, MeshInstance mesh_instance)
{
	mesh_instances[index] = mesh_instance;

	mesh_instances[index].build_inv();

	cudaMemcpy(&d_mesh_instances[index], &mesh_instances[index], sizeof(MeshInstance), cudaMemcpyHostToDevice);
}
