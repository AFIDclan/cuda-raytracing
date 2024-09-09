#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "transforms.hpp"
#include "Ray.hpp"
#include "TrianglePrimitive.hpp"
#include "MeshPrimitive.h"
#include "MeshInstance.hpp"
#include "Material.hpp"



__global__ void render(uchar3* img, int width, int height, size_t pitch, const float3x3 K_inv, const lre camera_pose, const float4 D, const lre inv_camera_pose, MeshInstance* mesh_instances, int num_mesh_instances, d_MeshPrimitive* meshes, Material* materials);