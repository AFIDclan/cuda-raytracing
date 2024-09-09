#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "transforms.hpp"
#include "Ray.hpp"
#include "TrianglePrimitive.hpp"
#include "MeshPrimitive.h"
#include "MeshInstance.hpp"
#include "Material.hpp"

struct HitInfo
{
    float min = FLT_MAX;
    TrianglePrimitive triangle;
    float3 location;
    float3 normal;
    Material material;
    float2 uv;
};


__device__ HitInfo& cast_ray(Ray& ray, curandState* state, MeshInstance* mesh_instances, int num_mesh_instances, d_MeshPrimitive* meshes, Material* materials, bool lighting_pass = false, float light_distance = FLT_MAX)
{

    HitInfo hit_info;

    for (int mesh_idx = 0; mesh_idx < num_mesh_instances; mesh_idx++) {

        MeshInstance mesh_instance = mesh_instances[mesh_idx];
        d_MeshPrimitive mesh = meshes[mesh_instance.mesh_index];
        Material material = materials[mesh_instance.material_index];

        // Express the ray direction in mesh coordinates
        float3 r_direction = apply_euler(mesh_instance.rotation, ray.direction);

        r_direction.x *= mesh_instance.inv_scale.x;
        r_direction.y *= mesh_instance.inv_scale.y;
        r_direction.z *= mesh_instance.inv_scale.z;

        // Express the ray origin in mesh coordinates
        float3 r_origin = apply_lre(mesh_instance.pose, ray.origin);

        // Eg. Scale of 2 --> multiply the origin times 0.5 and make the object appear 2x the size
        r_origin.x *= mesh_instance.inv_scale.x;
        r_origin.y *= mesh_instance.inv_scale.y;
        r_origin.z *= mesh_instance.inv_scale.z;

        Ray r_ray = Ray(
            r_origin,
            r_direction,
            ray.pixel
        );


        int stack[32];
        int stack_index = 0;

        // Start with the root node
        stack[stack_index++] = 0;

        while (stack_index > 0) {
            int node_index = stack[--stack_index];
            d_BVHTree current_bvh = mesh.bvh_top[node_index];

            // We are assuming this ray intersects with the bounding box of the node since it was pushed onto the stack

            if (current_bvh.child_index_a > 0) {
                // If the node has children, push them onto the stack

                float dist_a = mesh.bvh_top[current_bvh.child_index_a].ray_intersects(r_ray);
                float dist_b = mesh.bvh_top[current_bvh.child_index_b].ray_intersects(r_ray);

                if (dist_a < dist_b) {
                    if (dist_b < hit_info.min) stack[stack_index++] = current_bvh.child_index_b;
                    if (dist_a < hit_info.min) stack[stack_index++] = current_bvh.child_index_a;
                }
                else {
                    if (dist_a < hit_info.min) stack[stack_index++] = current_bvh.child_index_a;
                    if (dist_b < hit_info.min) stack[stack_index++] = current_bvh.child_index_b;
                }


            }
            else {
                // Leaf node: check for intersections with triangles
                for (int i = 0; i < current_bvh.count_triangles; i++) {
                    int index = current_bvh.triangle_indices[i];

                    //float3 intersection = mesh.triangles[index].ray_intersect(r_ray);

                    //// If the intersection is at FLT_MAX, the ray did not intersect with the triangle
                    //if (intersection.x == FLT_MAX)
                    //    continue;

                    //float2 uv = mesh.triangles[index].point_inside(intersection);

                    float dist;

                    float2 uv = mesh.triangles[index].ray_inside(r_ray, dist);

                    if (uv.x != FLT_MAX) {

                        // Express the location in world coordinates
                        float3 intersection = r_ray.origin + r_ray.direction * dist;

                        hit_info.location.x = intersection.x * mesh_instance.scale.x;
                        hit_info.location.y = intersection.y * mesh_instance.scale.y;
                        hit_info.location.z = intersection.z * mesh_instance.scale.z;

                        hit_info.location = apply_lre(mesh_instance.inv_pose, hit_info.location);

                        float distance = magnitude(hit_info.location - ray.origin);

                        // Positive means the ray is facing the same direction as the normal and we hit the back of the triangle
                        float same_dir = dot(r_ray.direction, mesh.triangles[index].normal);

                        if (same_dir < 0 && (hit_info.min == FLT_MAX || distance < hit_info.min)) {
                            hit_info.min = distance;
                            hit_info.triangle = mesh.triangles[index];


                            // Express normal in world coordinates
                            hit_info.normal = apply_euler(mesh_instance.inv_rotation, hit_info.triangle.normal);

                            hit_info.normal.x *= mesh_instance.scale.x;
                            hit_info.normal.y *= mesh_instance.scale.y;
                            hit_info.normal.z *= mesh_instance.scale.z;

                            // Scaling the direction can un-normalize it
                            hit_info.normal = normalize(hit_info.normal);

                            hit_info.uv = uv;

                            hit_info.material = material;


                            //if (lighting_pass && (material.illumination > 0 || distance < light_distance))
                       /*     if (lighting_pass &&  distance < light_distance)
                            {
                                return hit_info;
                            }*/
                        }
                    }
                }
            }
        }
    }

    return hit_info;
}


// Simple CUDA kernel to invert image colors
__global__ void render(uchar3* img, int width, int height, size_t pitch, const float3x3 K_inv, const lre camera_pose, const float4 D, const lre inv_camera_pose, MeshInstance* mesh_instances, int num_mesh_instances, d_MeshPrimitive* meshes, Material* materials) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    uchar3* row = (uchar3*)((char*)img + y * pitch);

    float3 origin = make_float3(camera_pose.x, camera_pose.y, camera_pose.z);


    float3 ph = make_float3(x, y, 1.0f);

    float3 direction = apply_matrix(K_inv, ph);



    float a = direction.x;
    float b = direction.y;

    float radius = sqrt(a * a + b * b);

    float theta = atan(radius);

    float thetad = theta * (1.0 + D.x * theta + D.y * theta * theta + D.z * theta * theta * theta + D.w * theta * theta * theta * theta);

    float scale = thetad / radius;

    direction.x = scale * a;
    direction.y = scale * b;

    direction = normalize(direction);

    // Rotate by 90 deg to make y forward (world space)
    direction = make_float3(direction.x, direction.z, -direction.y);

    // Apply the camera's pose to the direction
    direction = apply_euler(make_float3(inv_camera_pose.yaw, inv_camera_pose.pitch, inv_camera_pose.roll), direction);

    //Camera Ray direction in world space
    direction = normalize(direction);

    long long seed = (y * width + x) * 1000;

    curandState state;
    curand_init(seed, 0, 0, &state);

    float3 accumulated_color = make_float3(0.0, 0.0, 0.0);

    Ray ray = Ray(
        origin,
        direction,
        make_uint2(x, y)
    );


    HitInfo hit_info = cast_ray(ray, &state, mesh_instances, num_mesh_instances, meshes, materials);


    // Hit nothing. Return the sky color
    if (hit_info.min == FLT_MAX) {

        // Pale blue sky
        row[x].x = (1.0 * 255);
        row[x].y = (0.8 * 255);
        row[x].z = (0.6 * 255);

        return;
    }

    // Move the ray to the hit location
    ray.origin = hit_info.location;

    // Apply the color of the triangle to the ray


    if (hit_info.material.texture_width > 0)
    {
        int tex_x = (hit_info.uv.x * (float)hit_info.material.texture_width);
        int tex_y = ((1.0 - hit_info.uv.y) * (float)hit_info.material.texture_height);

        tex_x = fmaxf(tex_x % hit_info.material.texture_width, 0);
        tex_y = fmaxf(tex_y % hit_info.material.texture_height, 0);

        uchar3* tex_row = (uchar3*)((char*)hit_info.material.texture + tex_y * hit_info.material.texture_pitch);

        uchar3 tex_color = tex_row[tex_x];

        // 1/255 = 0.0039215
        ray.color.x *= tex_color.x * 0.0039215f;
        ray.color.y *= tex_color.y * 0.0039215f;
        ray.color.z *= tex_color.z * 0.0039215f;
    }
    else {
        ray.color.x *= hit_info.material.albedo.x;
        ray.color.y *= hit_info.material.albedo.y;
        ray.color.z *= hit_info.material.albedo.z;
    }



    float3 light_direction = make_float3(-0.2, 0, 1);
    light_direction = normalize(light_direction);

    float light_illumination = 1.0;

    ray.direction = light_direction;
    ray.direction_inv = make_float3(1.0 / light_direction.x, 1.0 / light_direction.y, 1.0 / light_direction.z);


    // Move just slightly so we don't capture the face we just hit
    ray.origin = ray.origin + ray.direction * 1e-4;


    // Calculate the loss of light due to the angle of incidence
    //float cos_illum = dot(hit_info.normal, ray.direction);


    //ray.illumination = 0.4 * cos_illum;

    //if ((dot(hit_info.normal, light_direction) > 0))
    //{

    //    // Cast toward light source
    //    hit_info = cast_ray(ray, &state, mesh_instances, num_mesh_instances, meshes, materials, true, FLT_MAX);


    //    // Check if we hit a light source
    //    if (hit_info.min == FLT_MAX)
    //    {
    //        ray.illumination = 1.0 * cos_illum;
    //    }
    //}

    ray.illumination = 1.0;

    //ray.illumination /= 2.0;

    //ray.illumination += cast_toward_lights(hit_info, make_float3(-3.0, 0.25, 3.0), 100.0, ray, &state, mesh_instances, num_mesh_instances, meshes, materials);
    //ray.illumination += cast_toward_lights(hit_info, make_float3(16.0, 0.25, 3.0), 100.0, ray, &state, mesh_instances, num_mesh_instances, meshes, materials);

    ray.illumination = fminf(1.0, ray.illumination);
    ray.illumination = fmaxf(0.4, ray.illumination);

    row[x].x = (ray.illumination * ray.color.x * 255);
    row[x].y = (ray.illumination * ray.color.y * 255);
    row[x].z = (ray.illumination * ray.color.z * 255);


}
