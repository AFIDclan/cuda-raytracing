#pragma once

#include <cuda_runtime.h>
#include <cassert>
#include <iostream>


namespace transforms {

	struct lre {
		float x, y, z, yaw, pitch, roll;

		__host__ __device__ lre() : x(0.0f), y(0.0f), z(0.0f), yaw(0.0f), pitch(0.0f), roll(0.0f) {}
	};

	struct float4x4 {
		float m[4][4];
	};

	struct float3x3 {
		float m[3][3];
	};




	static __host__ __device__ float3x3 invert_rotmat(const float3x3& rotmat) {
		return float3x3{
			rotmat.m[0][0], rotmat.m[1][0], rotmat.m[2][0],
			rotmat.m[0][1], rotmat.m[1][1], rotmat.m[2][1],
			rotmat.m[0][2], rotmat.m[1][2], rotmat.m[2][2]
		};
	}

	static __host__ __device__ float3 apply_rotmat(const float3x3& rotmat, const float3& vec) {
		float3 result;
		result.x = rotmat.m[0][0] * vec.x + rotmat.m[0][1] * vec.y + rotmat.m[0][2] * vec.z;
		result.y = rotmat.m[1][0] * vec.x + rotmat.m[1][1] * vec.y + rotmat.m[1][2] * vec.z;
		result.z = rotmat.m[2][0] * vec.x + rotmat.m[2][1] * vec.y + rotmat.m[2][2] * vec.z;
		return result;
	}


	static __host__ __device__ float4x4 invert_homo(const float4x4& H) {
		float4x4 result;

		// Extract the rotation part (upper-left 3x3)
		float3x3 R = {
			H.m[0][0], H.m[0][1], H.m[0][2],
			H.m[1][0], H.m[1][1], H.m[1][2],
			H.m[2][0], H.m[2][1], H.m[2][2]
		};

		// Invert the rotation matrix
		float3x3 R_inv = invert_rotmat(R);

		// Extract and invert the translation part (upper-right 3x1)
		float3 t = make_float3(-H.m[0][3], -H.m[1][3], -H.m[2][3]);
		float3 t_inv = apply_rotmat(R_inv, t);

		// Construct the inverted homogeneous matrix
		result.m[0][0] = R_inv.m[0][0]; result.m[0][1] = R_inv.m[0][1]; result.m[0][2] = R_inv.m[0][2]; result.m[0][3] = t_inv.x;
		result.m[1][0] = R_inv.m[1][0]; result.m[1][1] = R_inv.m[1][1]; result.m[1][2] = R_inv.m[1][2]; result.m[1][3] = t_inv.y;
		result.m[2][0] = R_inv.m[2][0]; result.m[2][1] = R_inv.m[2][1]; result.m[2][2] = R_inv.m[2][2]; result.m[2][3] = t_inv.z;
		result.m[3][0] = 0.0f;         result.m[3][1] = 0.0f;         result.m[3][2] = 0.0f;         result.m[3][3] = 1.0f;

		return result;
	}

	static __host__ __device__ float4x4 matmul(float4x4 a, float4x4 b) {
		float4x4 result;

		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				result.m[i][j] = 0.0f;
				for (int k = 0; k < 4; ++k) {
					result.m[i][j] += a.m[i][k] * b.m[k][j];
				}
			}
		}

		return result;
	}

	static __host__ __device__ float4x4 compose_homo(float4x4 H1, float4x4 H2)
	{
		return matmul(H2, H1);
	}


	static __host__ __device__ float3 rotmat2euler(float3x3 rotmat)
	{
		float a = rotmat.m[1][2];
		if (a > 1) a = 1;
		else if (a < -1) a = -1;

		return make_float3(atan2f(rotmat.m[1][0], rotmat.m[1][1]), asinf(a), atan2f(-rotmat.m[0][2], rotmat.m[2][2]));
	}


	static __host__ __device__ float3x3 euler2rotmat(float3 euler)
	{
		float sy = sinf(euler.x);
		float cy = cosf(euler.x);
		float sp = sinf(euler.y);
		float cp = cosf(euler.y);
		float sr = sinf(euler.z);
		float cr = cosf(euler.z);

		return float3x3{
			cr * cy + sr * sp * sy, -cr * sy + sr * sp * cy, -sr * cp,
			cp * sy, cp * cy, sp,
			sr * cy - cr * sp * sy, -sr * sy - cr * sp * cy, cr * cp
		};

	}



	static __host__ __device__ float4 euler2quat(float3 euler) {

		float sy = sinf(euler.x * 0.5);
		float cy = cosf(euler.x * 0.5);
		float sp = sinf(euler.y * 0.5);
		float cp = cosf(euler.y * 0.5);
		float sr = sinf(euler.z * 0.5);
		float cr = cosf(euler.z * 0.5);

		return make_float4(
			sy * sp * sr + cy * cp * cr,
			cy * sp * cr + sy * cp * sr,
			-sy * sp * cr + cy * cp * sr,
			cy * sp * sr - sy * cp * cr
		);
	}

	static __host__ __device__ float3 apply_quat(float4 q, float3 v) {

		float a = -v.x * q.y - v.y * q.z - v.z * q.w;
		float b = v.x * q.x + v.y * q.w - v.z * q.z;
		float c = v.y * q.x + v.z * q.y - v.x * q.w;
		float d = v.z * q.x + v.x * q.y - v.y * q.y;

		return make_float3(q.x * b - q.y * a - q.z * d + q.w * c,
			q.x * c - q.z * a - q.w * b + q.y * d,
			q.x * d - q.w * a - q.y * c + q.z * b);
		
	}

	static __host__ __device__ float4x4 lre2homo(lre v)
	{
		float3 shift = make_float3(-v.x, -v.y, -v.z);
		float3x3 R = euler2rotmat(make_float3(v.yaw, v.pitch, v.roll));

		float4x4 result = {
			R.m[0][0], R.m[0][1], R.m[0][2], shift.x,
			R.m[1][0], R.m[1][1], R.m[1][2], shift.y,
			R.m[2][0], R.m[2][1], R.m[2][2], shift.z,
			0.0f, 0.0f, 0.0f, 1.0f
		};

		return result;
	}

	static __host__ __device__ lre homo2lre(float4x4 H)
	{
		float3x3 rotmat = {
			H.m[0][0], H.m[0][1], H.m[0][2],
			H.m[1][0], H.m[1][1], H.m[1][2],
			H.m[2][0], H.m[2][1], H.m[2][2]
		};

		float3 euler = rotmat2euler(rotmat);
		float3 shift = make_float3(H.m[0][3], H.m[1][3], H.m[2][3]);
		shift = apply_rotmat(invert_rotmat(rotmat), shift);

		lre out;
		out.x = -shift.x;
		out.y = -shift.y;
		out.z = -shift.z;
		out.yaw = euler.x;
		out.pitch = euler.y;
		out.roll = euler.z;

		return out;
	}


	static __host__ __device__ float3 apply_euler(float3 euler, float3 v) {
		return apply_quat(euler2quat(euler), v);
	}

	static __host__ __device__ float3 apply_lre(lre lre, float3 v) {
		float3 subtracted = make_float3(v.x - lre.x, v.y - lre.y, v.z - lre.z);
		return apply_euler(make_float3(lre.yaw, lre.pitch, lre.roll), subtracted);
	}

	static __host__ __device__ lre compose_lre(lre lre1, lre lre2) {
		return homo2lre(compose_homo(lre2homo(lre1), lre2homo(lre2)));
	}

	static __host__ __device__ lre invert_lre(lre lre0)
	{
		return homo2lre(invert_homo(lre2homo(lre0)));
	}





	// ==== TEST FUNCTIONS

	static void test_all()
	{
		float3 vecz = { 0, 0, 1 };
		float3 rot = make_float3(0, 3.141592 / 2, 0);
		float4 quat = euler2quat(rot);

		float3 vecy = apply_quat(quat, vecz);



		std::cout << "vec in: " << vecz.x << ", " << vecz.y << ", " << vecz.z << std::endl;
		std::cout << "rot in: " << rot.x << ", " << rot.y << ", " << rot.z << std::endl;
		std::cout << "quat: " << quat.x << ", " << quat.y << ", " << quat.z << ", " << quat.w << std::endl;


		std::cout << "vec out: " << vecy.x << ", " << vecy.y << ", " << vecy.z << std::endl;

	}
}
