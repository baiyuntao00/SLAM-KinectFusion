#pragma once
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cuda.h"
#include <cstddef>
#include <types.hpp>
#include <math.h>
#include <device_array.hpp>

#define DIVUP(a, b) (a + b - 1) / b
#define MAXPOINTNUM 2000000
//TODO: struct data
namespace kf
{
	namespace device
	{
		using namespace cv::cuda;
		//此处定义设备内存存储方式
        typedef uchar3 Color;
		typedef float3 Point3;
		typedef float3 Normal;
		
		struct PoseR {float3 data[3];};

		struct PoseT {PoseR R;float3 t;};

		struct Intrs
		{
			float2 f, c;
			float2 finv;
			Intrs();
			Intrs(const Intrinsics &intr_);
			__device__ float3 reproj(int u, int v, float z) const;
			__device__ int2 proj(const float3 &p) const;
		};

		struct Voxel {short tsdf;short weight;uchar3 rgb;};

		struct Volume
		{
		public:
			typedef Voxel elem_type;
			elem_type *const data_;
			int3 dims;
			float3 volume_range;
			float3 voxel_size;
			float trun_dist;
			int znumber;
			// host:
			Volume(elem_type *data, const int3 dims_, const float3 volume_range_, const float3 voxel_size_);
			// host and device:
			__device__ __forceinline__ elem_type *operator()(int x, int y, int z) const;
			__device__ __forceinline__ elem_type *operator()(int x, int y, int z);
			__device__ __forceinline__ elem_type *operator()(int x, int y) const;
			__device__ __forceinline__ elem_type *zstep(elem_type *const ptr) const;
		};

		struct ICP
		{
			PtrStep<float3> cur_vmap;
			PtrStep<float3> cur_nmap;
			PtrStep<float3> pre_vmap;
			PtrStep<float3> pre_nmap;

			float min_angle;
			float max_dist_squ;
			Intrs intr;
			PoseT curpose;
			int2 size;

			ICP(const float dist_thres_, const float angle_thres_);
			void setIntrs(const Intrs intrs_, int x, int y);
			__device__ bool findCoresp(int x, int y, float3 &nd, float3 &d, float3 &s) const;
		};
	}
}
//TODO: .cu function
namespace kf
{
	namespace device
	{
		// tsdf_volume
		void resetVolume(Volume &vpointer);
		void raycast(const Intrs &intr, const PoseT &pose,const PoseR &Rinv, const Volume &vol, GpuMat &vmap, GpuMat &nmap);
		void integrate(const Intrs &intr, const PoseT &pose, Volume &volume, const GpuMat &dmap, const GpuMat &cmap);
		// image process
		void renderPhong(const float3 &poset, const GpuMat &vmap, const GpuMat &nmap, GpuMat &cmap);
		void renderNormals(const GpuMat &nmap, GpuMat &cmap);
		void depthTruncation(GpuMat &dmap, const float max_dist);
		void getNormalmap(const GpuMat &vmap, GpuMat &nmap);
		void getVertexmap(const GpuMat &vmag, GpuMat &nmap, const Intrs &params);
		void resizePointsNormals(const GpuMat &vex_big, const GpuMat &nor_big, GpuMat &vex_small, GpuMat &nor_small);
		// icp
		void rigidICP(const ICP &icphelper, cv::Matx66d &A, cv::Vec6d &b);
		// file
		//void extract_points(const Volume &vpointer, int *points_num_label, float *points, float *normals,unsigned char*colors);
		size_t extract_points(const Volume& vpointer, PtrSz<Point3> parray,  const PoseT&);
	}
}

//TODO:  mathfunction
namespace kf
{
	namespace device
	{
		//__device__ static float NaNf() { return __int_as_float(0x7fffffff); /*CUDART_NAN_F*/ };
		__device__ __inline__ float3 operator*(const PoseR &v1, const float3 &v2)
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
		__host__ __inline__ PoseR cv2cuda(const cv::Matx33f &m)
		{
			PoseR res;
			for (int i = 0; i < 3; i++)
			{
				res.data[i].x = m(0, i);
				res.data[i].y = m(1, i);
				res.data[i].z = m(2, i);
			}
			return res;
		};
		__host__ __inline__ PoseT cv2cuda(const cv::Affine3f &v)
		{
			PoseT out;
			out.R = cv2cuda(v.rotation());
			out.t = cv2cuda(v.translation());
			return out;
		};
		//
		__host__ __inline__ PoseT cv2cuda(const cv::Matx33f &R, const cv::Vec3f &t)
		{
			PoseT out;
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