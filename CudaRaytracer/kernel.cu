#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <Eigen/Dense>


using namespace Eigen;

struct Ray {
    float3 origin;
    float3 direction;
    uint2 pixel;
    uchar3 color;
    float illumination;

    // Constructor
    __device__ Ray(float3 o, float3 d, uint2 p)
        : origin(o), direction(d), pixel(p), color(make_uchar3(255, 255, 255)), illumination(0.0f) {}
};

struct float4x4 {
    float m[4][4];
};

struct float3x3 {
    float m[3][3];
};


float3x3 eigen_mat_to_float(const Eigen::Matrix3d& matrix) {
    float3x3 result;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result.m[i][j] = static_cast<float>(matrix(i, j));
        }
    }

    return result;
}


float4x4 eigen_mat_to_float(const Eigen::Matrix4d& matrix) {
    float4x4 result;

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            result.m[i][j] = static_cast<float>(matrix(i, j));
        }
    }

    return result;
}

__device__ float4 apply_matrix(const float4x4& matrix, const float4& vec) {
    float4 result;
    result.x = matrix.m[0][0] * vec.x + matrix.m[0][1] * vec.y + matrix.m[0][2] * vec.z + matrix.m[0][3] * vec.w;
    result.y = matrix.m[1][0] * vec.x + matrix.m[1][1] * vec.y + matrix.m[1][2] * vec.z + matrix.m[1][3] * vec.w;
    result.z = matrix.m[2][0] * vec.x + matrix.m[2][1] * vec.y + matrix.m[2][2] * vec.z + matrix.m[2][3] * vec.w;
    result.w = matrix.m[3][0] * vec.x + matrix.m[3][1] * vec.y + matrix.m[3][2] * vec.z + matrix.m[3][3] * vec.w;
    return result;
}

__device__ float3 apply_matrix(const float3x3& matrix, const float3& vec) {
    float3 result;
    result.x = matrix.m[0][0] * vec.x + matrix.m[0][1] * vec.y + matrix.m[0][2] * vec.z;
    result.y = matrix.m[1][0] * vec.x + matrix.m[1][1] * vec.y + matrix.m[1][2] * vec.z;
    result.z = matrix.m[2][0] * vec.x + matrix.m[2][1] * vec.y + matrix.m[2][2] * vec.z;
    return result;
}


// Simple CUDA kernel to invert image colors
__global__ void raytrace(uchar3* img, int width, int height, size_t pitch, const float3x3 H_px_to_ph, const float4x4 H_ph_to_world) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
		return;
	}

	float3 ph = make_float3(x, y, 1.0f);
	ph = apply_matrix(H_px_to_ph, ph);

	//Normalize
	float mag = sqrt(ph.x * ph.x + ph.y * ph.y + ph.z * ph.z);
	ph.x /= mag;
	ph.y /= mag;
	ph.z /= mag;

	float4 origin = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
	float4 direction = make_float4(ph.x, ph.y, ph.z, 1.0f);

	origin = apply_matrix(H_ph_to_world, origin);
	direction = apply_matrix(H_ph_to_world, direction);

	origin.x /= origin.w;
	origin.y /= origin.w;
	origin.z /= origin.w;

	direction.x /= direction.w;
	direction.y /= direction.w;
	direction.z /= direction.w;

	Ray ray(make_float3(origin.x, origin.y, origin.z), make_float3(direction.x, direction.y, direction.z), make_uint2(x, y));


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
    int width = 640;
    int height = 480;
    double fps = 0.0;

    int64 start_time = 0;
    int64 end_time = 0;

    /*
    K = np.array([
    [800, 0, 640],
    [0, 800, 360],
    [0, 0, 1]
    ])
    */

	Matrix3d K;
	K << 400, 0, width/2,
		 0,   400, height/2,
		 0,   0,   1;


	Matrix3d H_px_to_ph = K.inverse();
	Matrix4d H_ph_to_world = Matrix4d::Identity();


	float3x3 H_px_to_ph_float = eigen_mat_to_float(H_px_to_ph);
	float4x4 H_ph_to_world_float = eigen_mat_to_float(H_ph_to_world);

    // Allocate CUDA memory for the image
    uchar3* d_img;
    size_t pitch;
    cudaMallocPitch(&d_img, &pitch, width * sizeof(uchar3), height);

    // Define CUDA kernel launch configuration
    dim3 block_size(10, 10);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);



    Vector3d vectorA(1.0, 2.0, 3.0);

    // Loop while program is running
    while (true) {
        // Start measuring time
        start_time = cv::getTickCount();

        // Launch the CUDA kernel to invert colors
        raytrace << <grid_size, block_size >> > (d_img, width, height, pitch, H_px_to_ph_float, H_ph_to_world_float);
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
