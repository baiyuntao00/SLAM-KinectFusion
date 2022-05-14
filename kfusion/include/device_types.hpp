#pragma once
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cuda.h"
#include <cstddef>
#include <types.hpp>
#include <math.h>

#define DIVUP(a, b) (a + b - 1) / b
namespace kf
{
	namespace device
	{
		using namespace cv::cuda;
		//此处定义设备内存存储方式
		struct cuMat3f
		{
			float3 data[3];
		};
		struct cuPose
		{
			cuMat3f R;
			float3 t;
		};
		struct cuIntrs
		{
			float2 f, c;
			float2 finv;
			cuIntrs(){};
			// host:
			cuIntrs(const Intrinsics &intr_)
			{
				f = make_float2(intr_.fx, intr_.fy);
				c = make_float2(intr_.cx, intr_.cy);
				finv = make_float2(1.f / intr_.fx, 1.f / intr_.fy);
			}
			// device:
			__device__ float3 reproj(int u, int v, float z) const;
			__device__ int2 proj(const float3 &p) const;
		};
		struct cuTSDF
		{
		public:
			typedef short2 elem_type;

			elem_type *const data_;
			int3 dims;
			float3 volume_range;
			float3 voxel_size;
			float trun_dist;
			int znumber;
			// host:
			cuTSDF(elem_type *data, const int3 dims_, const float3 volume_range_, const float3 voxel_size_)
				: data_(data), dims(dims_),
				  volume_range(volume_range_),
				  voxel_size(voxel_size_)
			{
				znumber = dims_.x * dims_.y;
			};
			// host and device:
			__device__ __forceinline__ elem_type *operator()(int x, int y, int z) const
			{
				return data_ + x + y * dims.x + z * znumber;
			};
			__device__ __forceinline__ elem_type *operator()(int x, int y, int z)
			{
				return data_ + x + y * dims.x + z * znumber;
			};
			__device__ __forceinline__ elem_type *operator()(int x, int y) const
			{
				return data_ + x + dims.x * y;
			};
			__device__ __forceinline__ elem_type *zstep(elem_type *const ptr) const
			{
				return ptr + znumber;
			};
		};
		struct cuICP
		{
			PtrStep<float3> cur_vmap;
			PtrStep<float3> cur_nmap;
			PtrStep<float3> pre_vmap;
			PtrStep<float3> pre_nmap;

			float min_angle;
			float max_dist_squ;
			cuPose prepose;
			cuIntrs intr;

			cuPose curpose;
			int2 size;

			cuICP(const float dist_thres_, const float angle_thres_,
				  const cuPose &prepose_)
			{
				min_angle = angle_thres_;
				max_dist_squ = dist_thres_;
				prepose = prepose_;
				// size = make_int2(sx, sy);
			}
			void setIntrs(const cuIntrs intrs_, int x, int y)
			{
				intr = intrs_;
				size = make_int2(x, y);
			};
			__device__ bool findCoresp(int x, int y, float3 &nd, float3 &d, float3 &s) const;
		};
		// tsdf_volume
		void resetVolume(cuTSDF &vpointer);
		void raycast(const cuIntrs &intr, const cuPose &pose, const cuTSDF &vol, GpuMat &vmap, GpuMat &nmap);
		void integrate(const cuIntrs &intr, const cuPose &pose, cuTSDF &volume, const GpuMat &dmap, const GpuMat &cmap);
		// image process
		void renderPhong(const float3 &poset, const GpuMat &vmap, const GpuMat &nmap, GpuMat &cmap);
		void renderNormals(const GpuMat &nmap, GpuMat &cmap);
		void depthTruncation(GpuMat &dmap, const float max_dist);
		void getNormalmap(const GpuMat &vmap, GpuMat &nmap);
		void getVertexmap(const GpuMat &vmag, GpuMat &nmap, const cuIntrs &params);
		void resizePointsNormals(const GpuMat &vex_big, const GpuMat &nor_big, GpuMat &vex_small, GpuMat &nor_small);
		// icp
		void rigidICP(const cuICP &icphelper, cv::Matx66d &A, cv::Vec6d &b);

	}
}
// math_function
namespace kf
{
	namespace device
	{
		//__device__ static float NaNf() { return __int_as_float(0x7fffffff); /*CUDART_NAN_F*/ };
		__device__ __inline__ float3 operator*(const cuMat3f &v1, const float3 &v2)
		{
			return make_float3(v1.data[0].x * v2.x + v1.data[1].x * v2.y + v1.data[2].x * v2.z,
							   v1.data[0].y * v2.x + v1.data[1].y * v2.y + v1.data[2].y * v2.z,
							   v1.data[0].z * v2.x + v1.data[1].z * v2.y + v1.data[2].z * v2.z);
		};
		__host__ __inline__ int3 cv2cuda(const cv::Vec3i v)
		{
			return make_int3(v(0), v(1), v(2));
		}
		__host__ __inline__ float3 cv2cuda(const cv::Vec3f &v)
		{
			return make_float3(v(0), v(1), v(2));
		};
		__host__ __inline__ cuMat3f cv2cuda(const cv::Matx33f &m)
		{
			cuMat3f res;
			for (int i = 0; i < 3; i++)
			{
				res.data[i].x = m(0, i);
				res.data[i].y = m(1, i);
				res.data[i].z = m(2, i);
			}
			return res;
		};
		__host__ __inline__ cuPose cv2cuda(const cv::Affine3f &v)
		{
			cuPose out;
			out.R = cv2cuda(v.rotation());
			out.t = cv2cuda(v.translation());
			return out;
		};
		//
		__host__ __inline__ cuPose cv2cuda(const cv::Matx33f &R, const cv::Vec3f &t)
		{
			cuPose out;
			out.R = cv2cuda(R);
			out.t = cv2cuda(t);
			return out;
		};
		// math_function
		__host__ __device__ __inline__ float3 &operator+=(float3 &v1, const float3 &v2)
		{
			v1.x += v2.x;
			v1.y += v2.y;
			v1.z += v2.z;
			return v1;
		}
		__host__ __device__ __inline__ float3 operator+(const float3 &v1, const float3 &v2)
		{
			return make_float3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
		}
		__host__ __device__ __inline__ uchar3 operator+(const uchar3 &v1, const uchar3 &v2)
		{
			return make_uchar3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
		}
		__host__ __device__ __inline__ float3 operator*(const float3 &v1, const float3 &v2)
		{
			return make_float3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
		}
		__host__ __device__ __inline__ float3 operator*(const float3 &v1, const int3 &v2)
		{
			return make_float3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
		}
		__host__ __device__ __inline__ float3 operator/(const float3 &v1, const float3 &v2)
		{
			return make_float3(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
		}
		__host__ __device__ __inline__ float3 operator/(const float &v1, const float3 &v2)
		{
			return make_float3(v1 / v2.x, v1 / v2.y, v1 / v2.z);
		}
		__host__ __device__ __inline__ float3 operator/(const float3 &v1, const int3 &v2)
		{
			return make_float3(v1.x / (float)v2.x, v1.y / (float)v2.y, v1.z / (float)v2.z);
		}
		__host__ __device__ __inline__ float3 operator/(const float3 &vec, const float &v)
		{
			return make_float3(vec.x / v, vec.y / v, vec.z / v);
		}
		__host__ __device__ __inline__ float3 &operator*=(float3 &vec, const float &v)
		{
			vec.x *= v;
			vec.y *= v;
			vec.z *= v;
			return vec;
		}
		__host__ __device__ __inline__ float3 operator-(const float3 &v1, const float3 &v2)
		{
			return make_float3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
		}
		__host__ __device__ __inline__ float3 operator*(const float3 &v1, const float &v)
		{
			return make_float3(v1.x * v, v1.y * v, v1.z * v);
		}
		__host__ __device__ __inline__ float3 operator*(const float &v, const float3 &v1)
		{
			return make_float3(v1.x * v, v1.y * v, v1.z * v);
		}
		__host__ __device__ __inline__ float dot(const float3 &v1, const float3 &v2)
		{
			return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
		}
		__host__ __device__ __inline__ uchar3 operator*(const float3 &v1, const uchar3 &v2)
		{
			return make_uchar3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
		}
		__host__ __device__ __inline__ float norm_sqr(const float3 &v)
		{
			return dot(v, v);
		}
		__host__ __device__ __inline__ float norm(const float3 &v)
		{
			return sqrt(dot(v, v));
		}
		__host__ __device__ __inline__ void normalize(float3 &v)
		{
			float t = sqrtf(dot(v, v));
			v = make_float3(v.x / t, v.y / t, v.z / t);
		}
		__host__ __device__ __inline__ float3 cross(const float3 &v1, const float3 &v2)
		{
			return make_float3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
		}
	}
}