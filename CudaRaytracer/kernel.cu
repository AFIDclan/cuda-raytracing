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

#include "OBJLoader.hpp"


// Simple CUDA kernel to invert image colors
__global__ void raytrace(uchar3* img, int width, int height, size_t pitch, const float3x3 K_inv, const lre camera_pose, TrianglePrimitive* triangles, int count_triangles, d_BVHTree* bvh_tree) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
		return;
	}

    float3 origin = make_float3(camera_pose.x, camera_pose.y, camera_pose.z);

	float3 ph = make_float3(x, y, 1.0f);
	float3 direction = apply_matrix(K_inv, ph);

	direction = apply_euler(make_float3(0, 3.141592/2, 0), direction);
    direction = apply_euler(make_float3(camera_pose.yaw, camera_pose.pitch, camera_pose.roll), direction);
     
    //Normalize
    direction = normalize(direction);

	Ray ray(origin, direction, make_uint2(x, y));

    float hit_min = -1.0f;
    uchar3 color = make_uchar3(0, 0, 0);



    int stack[64];
    int stack_index = 0;

    // Start with the root node
    stack[stack_index++] = 0; 

    while (stack_index > 0) {
        int node_index = stack[--stack_index];
        d_BVHTree current_bvh = bvh_tree[node_index];

        // Check if the current node intersects with the ray
		float bb_dist = current_bvh.ray_intersects(ray);
        if (bb_dist < FLT_MAX) {

			// If the current node is further than the closest hit, skip it
			if (bb_dist > hit_min && hit_min != -1.0f) {
				continue;
			}

            if (current_bvh.child_index_a > 0) {
                // If the node has children, push them onto the stack

				// TODO: Push the closest child first
                stack[stack_index++] = current_bvh.child_index_b;
                stack[stack_index++] = current_bvh.child_index_a;
            }
            else {
                // Leaf node: check for intersections with triangles
               for (int i = 0; i < current_bvh.count_triangles; i++) {
                    int index = current_bvh.triangle_indices[i];

                    float3 intersection = triangles[index].ray_intersect(ray);

					// If the intersection is at FLT_MAX, then the ray did not intersect with the triangle
                    if (intersection.x == FLT_MAX)
                        continue;

                    bool inside = triangles[index].point_inside(intersection);

                    if (inside) {
                        float distance = magnitude(intersection - ray.origin);

                        if (hit_min == -1.0f || distance < hit_min) {
                            hit_min = distance;
                            float brightness = dot(ray.direction, triangles[index].normal);
                            color = make_uchar3(brightness * triangles[index].color.x, brightness * triangles[index].color.y, brightness * triangles[index].color.z);
                        }
                    }
               }
            }
        }
    }

    if (hit_min > 0.0f) {
        uchar3* row = (uchar3*)((char*)img + y * pitch);
        row[x] = color;
    }
    else {
        uchar3* row = (uchar3*)((char*)img + y * pitch);
        row[x].x = (direction.x * 255);
        row[x].y = (direction.y * 255);
        row[x].z = (direction.z * 255);
    }
	
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

    MeshPrimitive teapot = OBJLoader::load("C:/workspace/CudaRaytracer/teapot.obj");
    //MeshPrimitive teapot = OBJLoader::load("C:/workspace/CudaRaytracer/cube.obj");

	teapot.set_world_position(make_float3(0, 8, -2));
	//teapot.set_world_position(make_float3(0, 0, 10));
	teapot.set_world_rotation(make_float3(0, 3.141592/2, 0));

    TrianglePrimitive* d_triangles;
    cudaMalloc(&d_triangles, teapot.num_triangles * sizeof(TrianglePrimitive));
    cudaMemcpy(d_triangles, teapot.world_triangles, teapot.num_triangles * sizeof(TrianglePrimitive), cudaMemcpyHostToDevice);


    d_BVHTree* d_bvh_tree = BVHTree::compile_tree(teapot.bvh_top);

    // Allocate CUDA memory for the image
    uchar3* d_img;
    size_t pitch;
    cudaMallocPitch(&d_img, &pitch, width * sizeof(uchar3), height);

    // Define CUDA kernel launch configuration
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

	float angle = 0.0f;

    // Loop while program is running
    while (true) {

		angle += 0.001f;


        // Start measuring time
        start_time = cv::getTickCount();


        //teapot.set_world_rotation(make_float3(0, angle, 0));
        //cudaMemcpy(d_triangles, teapot.world_triangles, teapot.num_triangles * sizeof(TrianglePrimitive), cudaMemcpyHostToDevice);

		//camera_pose.pitch = angle;

        // Launch the CUDA kernel to invert colors
        raytrace << <grid_size, block_size >> > (d_img, width, height, pitch, K_inv, camera_pose, d_triangles, teapot.num_triangles, d_bvh_tree);
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
