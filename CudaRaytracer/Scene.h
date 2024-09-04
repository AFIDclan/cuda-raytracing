#pragma once
#include <cuda_runtime.h>
#include <vector>

#include "MeshInstance.hpp"
#include "MeshPrimitive.h"
#include "Material.hpp"


class Scene
{
	std::vector<Material> materials;
	std::vector<MeshPrimitive> meshes;
	std::vector<MeshInstance> mesh_instances;

public:
	Scene();

	void add_material(Material material);
	void add_mesh(MeshPrimitive mesh);
	void add_mesh_instance(MeshInstance mesh_instance);

	Material* d_materials;
	d_MeshPrimitive* d_meshes;
	MeshInstance* d_mesh_instances;
	int num_mesh_instances;
	void upload_to_device();
	void update_mesh_instance(int index, MeshInstance mesh_instance);
	
};

