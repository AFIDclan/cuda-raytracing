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



// Simple CUDA kernel to invert image colors
__global__ void raytrace(uchar3* img, int width, int height, size_t pitch, const float3x3 K_inv, const lre camera_pose, TrianglePrimitive* triangles, int count_triangles) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
		return;
	}

    //float3 origin = make_float3(camera_pose.x, camera_pose.y, camera_pose.z);
    float3 origin = make_float3(0,0,0);

	float3 ph = make_float3(x, y, 1.0f);
	float3 direction = apply_matrix(K_inv, ph);

	//Normalize
    direction = normalize(direction);

	//direction = apply_lre(camera_pose, direction);

	Ray ray(origin, direction, make_uint2(x, y));

	for (int i = 0; i < count_triangles; i++) {
        float3 intersection = triangles[i].ray_intersect(ray);

        if (intersection.x == 0.0f && intersection.y == 0.0f && intersection.z == 0.0f) 
			continue;

        bool inside = triangles[i].point_inside(intersection);

        if (inside) {
            uchar3* row = (uchar3*)((char*)img + y * pitch);
            row[x] = triangles[i].color;

            return;
        }
        
    }
	

    uchar3* row = (uchar3*)((char*)img + y * pitch);
    row[x].x = direction.x*255;
    row[x].y = direction.y*255;
    row[x].z = direction.z*255;
    
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

	std::vector<TrianglePrimitive> vec_tris;

	vec_tris.push_back(TrianglePrimitive(make_float3(-1.0f, 1.0f, 6.0f), make_float3(1.0f, 1.0f, 6.5f), make_float3(0.0f, -1.0f, 6.0f), make_uchar3(255, 128, 0)));
    vec_tris.push_back(TrianglePrimitive(make_float3(-3.0f, 2.0f, 6.0f), make_float3(-2.0f, 2.0f, 6.5f), make_float3(-2.5f, -1.0f, 6.0f), make_uchar3(128, 200, 0)));


    MeshPrimitive mesh = MeshPrimitive(vec_tris);



    TrianglePrimitive* d_triangles;
    cudaMalloc(&d_triangles, mesh.num_triangles * sizeof(TrianglePrimitive));
    cudaMemcpy(d_triangles, mesh.world_triangles, mesh.num_triangles * sizeof(TrianglePrimitive), cudaMemcpyHostToDevice);


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

        //AngleAxisd rotation(angle, Vector3d::UnitZ());

        //mesh.set_world_rotation(rotation);
        //cudaMemcpy(d_triangles, mesh.world_triangles, mesh.num_triangles * sizeof(TrianglePrimitive), cudaMemcpyHostToDevice);

        // Launch the CUDA kernel to invert colors
        raytrace << <grid_size, block_size >> > (d_img, width, height, pitch, K_inv, camera_pose, d_triangles, mesh.num_triangles);
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
