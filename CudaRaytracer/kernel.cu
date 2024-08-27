#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <Eigen/Dense>


using namespace Eigen;

__host__ __device__ float3 normalize(float3 v) {
	float mag = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
	return make_float3(v.x / mag, v.y / mag, v.z / mag);
}

__host__ __device__ float3 cross(float3 a, float3 b) {
	return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__host__ __device__ float dot(float3 a, float3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ float3 operator+(float3 a, float3 b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ float3 operator-(float3 a, float3 b) {
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ float3 operator*(float3 a, float b) {
	return make_float3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ float3 operator*(float b, float3 a) {
	return make_float3(a.x * b, a.y * b, a.z * b);
}

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

struct Triangle {
    float3 vertices[3];
    float3 normal;
    uchar3 color;

    // Constructor
    __host__ __device__ Triangle(float3 a, float3 b, float3 c, uchar3 color)
    : color(color) {
		vertices[0] = a;
		vertices[1] = b;
		vertices[2] = c;

		float3 v0 = vertices[2] - vertices[0];
		float3 v1 = vertices[1] - vertices[0];
		normal = normalize(cross(v0, v1));
    }


    __host__ __device__ Triangle(float3 a, float3 b, float3 c, float3 normal, uchar3 color)
        : normal(normal), color(color) {
        vertices[0] = a;
        vertices[1] = b;
        vertices[2] = c;
    }

    __host__ __device__ Triangle() : normal(make_float3(0.0f, 0.0f, 0.0f)), color(make_uchar3(255, 255, 255)) {
        vertices[0] = make_float3(0.0f, 0.0f, 0.0f);
        vertices[1] = make_float3(0.0f, 0.0f, 0.0f);
        vertices[2] = make_float3(0.0f, 0.0f, 0.0f);
    }

    __host__ __device__ float3 ray_intersect(const Ray& ray) {

        float denom = dot(ray.direction, normal);

        if (abs(denom) < 1e-6) {
            return make_float3(0.0f, 0.0f, 0.0f);
        }

        float t = dot(vertices[0] - ray.origin, normal) / denom;

        if (t < 0.0f) {
            return make_float3(0.0f, 0.0f, 0.0f);
        }

        float3 point = ray.origin + t * ray.direction;

        return point;
    }

    __host__ __device__ bool point_inside(const float3& point) const {

        float3 v0 = vertices[2] - vertices[0];
        float3 v1 = vertices[1] - vertices[0];
        float3 v2 = point - vertices[0];

        float dot00 = dot(v0, v0);
        float dot01 = dot(v0, v1);
        float dot02 = dot(v0, v2);
        float dot11 = dot(v1, v1);
        float dot12 = dot(v1, v2);

        float invDenom = 1.0f / (dot00 * dot11 - dot01 * dot01);
        float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
        float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

        return (u >= 0.0f) && (v >= 0.0f) && (u + v < 1.0f);
    }

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
__global__ void raytrace(uchar3* img, int width, int height, size_t pitch, const float3x3 H_px_to_ph, const float4x4 H_ph_to_world, Triangle* triangles, int count_triangles) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
		return;
	}

	float3 ph = make_float3(x, y, 1.0f);
	ph = apply_matrix(H_px_to_ph, ph);

	//Normalize
	ph = normalize(ph);

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

    /*
    K = np.array([
    [800, 0, 640],
    [0, 800, 360],
    [0, 0, 1]
    ])
    */

	Matrix3d K;

 //   int width = 640;
 //   int height = 480;

	//K << 400, 0, width/2,
	//	 0,   400, height/2,
	//	 0,   0,   1;

    int width = 1280;
    int height = 720;

    K << 800, 0, width / 2,
        0, 800, height / 2,
        0, 0, 1;

	Matrix3d H_px_to_ph = K.inverse();
	Matrix4d H_ph_to_world = Matrix4d::Identity();


	float3x3 H_px_to_ph_float = eigen_mat_to_float(H_px_to_ph);
	float4x4 H_ph_to_world_float = eigen_mat_to_float(H_ph_to_world);



	int count_triangles = 2;
	Triangle* triangles = new Triangle[count_triangles];

	triangles[0] = Triangle(make_float3(-1.0f, 1.0f, 6.0f), make_float3(1.0f, 1.0f, 6.5f), make_float3(0.0f, -1.0f, 6.0f), make_uchar3(255, 128, 0));
	triangles[1] = Triangle(make_float3(-3.0f, 2.0f, 6.0f), make_float3(-2.0f, 2.0f, 6.5f), make_float3(-2.5f, -1.0f, 6.0f), make_uchar3(128, 200, 0));

    Triangle* d_triangles;
    cudaMalloc(&d_triangles, count_triangles * sizeof(Triangle));
    cudaMemcpy(d_triangles, triangles, count_triangles * sizeof(Triangle), cudaMemcpyHostToDevice);


    // Allocate CUDA memory for the image
    uchar3* d_img;
    size_t pitch;
    cudaMallocPitch(&d_img, &pitch, width * sizeof(uchar3), height);

    // Define CUDA kernel launch configuration
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);


    // Loop while program is running
    while (true) {
        // Start measuring time
        start_time = cv::getTickCount();

        // Launch the CUDA kernel to invert colors
        raytrace << <grid_size, block_size >> > (d_img, width, height, pitch, H_px_to_ph_float, H_ph_to_world_float, d_triangles, count_triangles);
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
