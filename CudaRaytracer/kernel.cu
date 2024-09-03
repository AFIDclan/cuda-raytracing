#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <cuda_runtime.h>
#include <iostream>

#include "utils.hpp"
#include "Ray.hpp"
#include "TrianglePrimitive.hpp"
#include "MeshPrimitive.h"
#include "MeshInstance.hpp"

#include "OBJLoader.hpp"
#include <windows.h> // For SetCursorPos


__device__ Ray& raytrace(Ray& ray, MeshInstance* mesh_instances, int num_mesh_instances, d_MeshPrimitive* meshes)
{

    float hit_min = FLT_MAX;
    TrianglePrimitive hit_triangle;
    float3 hit_location;
    float3 hit_normal;

    for (int mesh_idx = 0; mesh_idx < num_mesh_instances; mesh_idx++) {

		MeshInstance mesh_instance = mesh_instances[mesh_idx];
        d_MeshPrimitive mesh = meshes[mesh_instance.mesh_index];

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


        int stack[64];
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
                    if (dist_b < hit_min) stack[stack_index++] = current_bvh.child_index_b;
                    if (dist_a < hit_min) stack[stack_index++] = current_bvh.child_index_a;
                }
                else {
                    if (dist_a < hit_min) stack[stack_index++] = current_bvh.child_index_a;
                    if (dist_b < hit_min) stack[stack_index++] = current_bvh.child_index_b;
                }


            }
            else {
                // Leaf node: check for intersections with triangles
                for (int i = 0; i < current_bvh.count_triangles; i++) {
                    int index = current_bvh.triangle_indices[i];

                    float3 intersection = mesh.triangles[index].ray_intersect(r_ray);

                    // If the intersection is at FLT_MAX, the ray did not intersect with the triangle
                    if (intersection.x == FLT_MAX)
                        continue;

                    bool inside = mesh.triangles[index].point_inside(intersection);

                    if (inside) {
                        float distance = magnitude(intersection - r_ray.origin);

						// Positive means the ray is facing the same direction as the normal and we hit the back of the triangle
						float same_dir = dot(r_ray.direction, mesh.triangles[index].normal);

                        if (same_dir < 0 && (hit_min == -1.0f || distance < hit_min)) {
                            hit_min = distance;
                            hit_triangle = mesh.triangles[index];
                            

                            // Express normal in world coordinates
                            hit_normal = apply_euler(mesh_instance.inv_rotation, hit_triangle.normal);

                            hit_normal.x *= mesh_instance.scale.x;
                            hit_normal.y *= mesh_instance.scale.y;
                            hit_normal.z *= mesh_instance.scale.z;

							// Scaling the direction can un-normalize it
							hit_normal = normalize(hit_normal);

                            // Express the location in world coordinates
                            hit_location = apply_lre(mesh_instance.inv_pose, intersection);

                            hit_location.x *= mesh_instance.scale.x;
                            hit_location.y *= mesh_instance.scale.y;
                            hit_location.z *= mesh_instance.scale.z;
                        }
                    }
                }
            }
        }
    }

    if (hit_min != FLT_MAX) {

		
		// Move the ray to the hit location
        ray.origin = hit_location;

        // Reflect around normal
        ray.direction = ray.direction - (2.0f * (dot(ray.direction, hit_normal) * hit_normal));
		
		// Calculate the inverse of the direction
		ray.direction_inv.x = 1.0f / ray.direction.x;
		ray.direction_inv.y = 1.0f / ray.direction.y;
		ray.direction_inv.z = 1.0f / ray.direction.z;
   

        // Move just slightly so we don't capture the face we just hit
        ray.origin = ray.origin + ray.direction * 1e-4;

		// Calculate the loss of light due to the angle of incidence
        float cos_illum = dot(hit_normal, ray.direction);

		// Apply the color of the triangle to the ray
        ray.color.x *= (hit_triangle.color.x * cos_illum);
        ray.color.y *= (hit_triangle.color.y * cos_illum);
        ray.color.z *= (hit_triangle.color.z * cos_illum);


     } else {
        // Pale blue sky
        ray.color.x *= 1.0;
        ray.color.y *= 0.8;
        ray.color.z *= 0.6;

        ray.illumination = 1.0;

        ray.terminated = true;
    }

    return ray;
}


// Simple CUDA kernel to invert image colors
__global__ void render(uchar3* img, int width, int height, size_t pitch, const float3x3 K_inv, const lre camera_pose, MeshInstance* mesh_instances, int num_mesh_instances, d_MeshPrimitive* meshes) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
		return;
	}

    float3 origin = make_float3(camera_pose.x, camera_pose.y, camera_pose.z);

	float3 ph = make_float3(x, y, 1.0f);
	float3 direction = apply_matrix(K_inv, ph);


    // Rotate by 90 deg to make y forward (world space)
    direction = make_float3(direction.x, direction.z, -direction.y);

    // Apply the camera's pose to the direction
    direction = apply_euler(make_float3(camera_pose.yaw, camera_pose.pitch, camera_pose.roll), direction);
     
    //Camera Ray direction in world space
    direction = normalize(direction);

    Ray ray = Ray(
        origin,
        direction,
        make_uint2(x, y)
    );

    for (int i = 0; i < 4; i++)
    {
        if (ray.terminated)
            break;

        ray = raytrace(ray, mesh_instances, num_mesh_instances, meshes);
    }

    
    //ray = raytrace(ray, meshes, num_meshes);

    //if (!ray.terminated)
    //{
    //    ray.illumination = dot(ray.direction, direction);
    //}
   

    uchar3* row = (uchar3*)((char*)img + y * pitch);
    row[x].x = (ray.color.x * ray.illumination * 255);
    row[x].y = (ray.color.y * ray.illumination * 255);
    row[x].z = (ray.color.z * ray.illumination * 255);
    
	
}

void display_image(uchar3* d_img, int width, int height, size_t pitch, double fps)
{
    // Wrap the CUDA memory in an OpenCV GpuMat
    cv::cuda::GpuMat img_gpu(height, width, CV_8UC3, d_img, pitch);

    // Download the processed image back to host memory
    cv::Mat img_cpu;
    img_gpu.download(img_cpu);

    // Convert FPS to string and overlay it on the image
    std::string fps_text = "FPS: " + std::to_string(fps);
    cv::putText(img_cpu, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

    // Display the image using OpenCV
    cv::imshow("Image", img_cpu);

    // Capture key pressed
    int key = cv::waitKey(1);

    // If the key pressed is 'q', then exit the loop
    if (key == 'q') {
        exit(0);
    }
}

struct MouseParams
{
    int last_x;
    int last_y;
    bool has_last = false;

    bool is_down = false;

    lre *pose;
};

void on_mouse(int event, int x, int y, int, void* param)
{
    // Cast the param back to the correct type
    MouseParams* mouse_state = static_cast<MouseParams*>(param);
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        mouse_state->is_down = true;
    } else if (event == cv::EVENT_LBUTTONUP)
    {
        mouse_state->is_down = false;
    } else if (event == cv::EVENT_MOUSEMOVE)
    {
        
        if (mouse_state->has_last && mouse_state->is_down)
        {
            int dx = x - mouse_state->last_x;
            int dy = y - mouse_state->last_y;

            mouse_state->pose->yaw += dx * -0.001;
            mouse_state->pose->pitch += dy * 0.001;
        }

        mouse_state->last_x = x;
        mouse_state->last_y = y;
        mouse_state->has_last = true;
    }
}

int main() {

	//transforms::test_all();

	//exit(0);

    // Image dimensions

    double fps = 0.0;

    int64 start_time = 0;
    int64 end_time = 0;


    int width = 1280;
    int height = 720;

    float3x3 K = {
        800.0, 0.0, width / 2,
        0,     800.0, height / 2,
        0,     0,   1
    };

	float3x3 K_inv = invert_intrinsic(K);

    lre camera_pose = lre();


    camera_pose.x = 0;
    camera_pose.y = -8;
    camera_pose.z = 0;





    MeshPrimitive cow = OBJLoader::load("./cow.obj");
    MeshPrimitive teapot = OBJLoader::load("./teapot.obj");
    MeshPrimitive cube = OBJLoader::load("./cube.obj");

    teapot.bvh_top.print_stats();

    d_MeshPrimitive* d_meshes;

    cudaMalloc(&d_meshes, sizeof(d_MeshPrimitive) * 3);

    cudaMemcpy(&d_meshes[0], cow.to_device(), sizeof(d_MeshPrimitive), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_meshes[1], teapot.to_device(), sizeof(d_MeshPrimitive), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_meshes[2], cube.to_device(), sizeof(d_MeshPrimitive), cudaMemcpyHostToDevice);

	MeshInstance cow_instance = MeshInstance(0);

    cow_instance.pose.x = -2;
	cow_instance.pose.pitch = 3.141592 / 2;
    cow_instance.scale = make_float3(0.2, 0.2, 0.2);


    MeshInstance cube_instance = MeshInstance(2);

    cube_instance.pose.z = -2;
    cube_instance.scale = make_float3(10.0, 10.0, 1.0);

	MeshInstance teapot_instance = MeshInstance(1);

	teapot_instance.pose.x = 2;
	teapot_instance.pose.pitch = 3.141592 / 2;

	

    cow_instance.build_inv();
    cube_instance.build_inv();
	teapot_instance.build_inv();


	MeshInstance* d_mesh_instances;

	cudaMalloc(&d_mesh_instances, sizeof(MeshInstance) * 3);

	cudaMemcpy(&d_mesh_instances[0], &cow_instance, sizeof(MeshInstance), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_mesh_instances[1], &cube_instance, sizeof(MeshInstance), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_mesh_instances[2], &teapot_instance, sizeof(MeshInstance), cudaMemcpyHostToDevice);




    //MeshPrimitive teapot = OBJLoader::load("C:/workspace/CudaRaytracer/cube.obj");

	//teapot.set_world_position(make_float3(0, 0, 8));
	//teapot.set_world_position(make_float3(0, 8, -2));

	//teapot.set_world_rotation(make_float3(0, 0, 0));

   




    // Allocate CUDA memory for the image
    uchar3* d_img;
    size_t pitch;
    cudaMallocPitch(&d_img, &pitch, width * sizeof(uchar3), height);

    // Define CUDA kernel launch configuration
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

	float angle = 0.0f;

    MouseParams mouse_state;
    mouse_state.pose = &camera_pose;
    cv::namedWindow("Image");
    cv::setMouseCallback("Image", on_mouse, &mouse_state);

    // Loop while program is running
    while (true) {

		angle += 0.005f;

        // Start measuring time
        start_time = cv::getTickCount();

        teapot_instance.pose.yaw = angle;
		teapot_instance.build_inv();

        cudaMemcpy(&d_mesh_instances[2], &teapot_instance, sizeof(MeshInstance), cudaMemcpyHostToDevice);

        // Launch the CUDA kernel to invert colors
        render << <grid_size, block_size >> > (d_img, width, height, pitch, K_inv, camera_pose, d_mesh_instances, 3, d_meshes);
        cudaDeviceSynchronize();

        // End measuring time
        end_time = cv::getTickCount();
        double time_taken = (end_time - start_time) / cv::getTickFrequency();
        fps = 1.0 / time_taken;

        display_image(d_img, width, height, pitch, fps);
    }

    // Free CUDA memory
    cudaFree(d_img);

    return 0;
}
