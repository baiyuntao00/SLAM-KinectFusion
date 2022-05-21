#pragma once
#include <device_types.hpp>
#include "cuda_runtime_api.h"

#define MAX_WEIGHT 64
#define DIVSHORTMAX 0.0000305185f // (1.f / SHRT_MAX);
#define SHORTMAX 32767			  // SHRT_MAX;

//TODO: member function
namespace kf
{
        namespace device
        {
        //Intrs
        Intrs:: Intrs(){};
        Intrs::Intrs(const Intrinsics &intr_)
		{
			f = make_float2(intr_.fx, intr_.fy);
			c = make_float2(intr_.cx, intr_.cy);
			finv = make_float2(1.f / intr_.fx, 1.f / intr_.fy);
		}
        __device__ int2 Intrs::proj(const float3 &p) const
        {
            int2 coo;
            coo.x = __float2int_rn(__fdividef(p.x, p.z) * f.x + c.x);
            coo.y = __float2int_rn(__fdividef(p.y, p.z) * f.y + c.y);
            return coo;
        };
        __device__ float3 Intrs::reproj(int u, int v, float z) const
        {
                float x = __fdividef(z * (u - c.x),f.x);
                float y = __fdividef(z * (v - c.y),f.y);
                return make_float3(x, y, z);
        };

        //Volume
        Volume::Volume(elem_type *data, const int3 dims_, const float3 volume_range_, const float3 voxel_size_)
        : data_(data),dims(dims_),volume_range(volume_range_),voxel_size(voxel_size_)
        {znumber = dims_.x * dims_.y;};
        __device__ __forceinline__ Volume::elem_type *Volume::operator()(int x, int y, int z) const
		{return data_ + x + y * dims.x + z * znumber;};
		__device__ __forceinline__ Volume::elem_type *Volume::operator()(int x, int y, int z)
		{return data_ + x + y * dims.x + z * znumber;};
		__device__ __forceinline__ Volume::elem_type *Volume::operator()(int x, int y) const
		{return data_ + x + dims.x * y;};
		__device__ __forceinline__ Volume::elem_type *Volume::zstep(elem_type *const ptr) const
        {return ptr + znumber;};

        //ICP
        ICP::ICP(const float dist_thres_, const float angle_thres_,const PoseT &prepose_)
        {
          min_angle = angle_thres_;
          max_dist_squ = dist_thres_;
          prepose = prepose_;
        }
        void ICP::setIntrs(const Intrs intrs_, int x, int y)
        {
          intr = intrs_;
          size = make_int2(x, y);
        };
    }
}
 //TODO: math and other functions more speed
namespace kf
{
        namespace device
        {
        __device__ static float __m_nan() { return __int_as_float(0x7fffffff); /*CUDART_NAN_F*/ };
        static float __m_fepsilon() { return 1.192092896e-07f/*FLT_EPSILON*/; };
        static float __m_fmin() { return 1.175494351e-38f/*FLT_MIN*/; };
        static float __m_fmax() { return 3.402823466e+38f/*FLT_MAX*/; };
        __device__ __inline__ float __m_norm(const float3 &v){ return __fsqrt_rn(dot(v, v));};
		__device__ __inline__ void __m_normalize(float3 &v)
		{
			float t = __fsqrt_rn(dot(v, v));
			v = make_float3(__fdividef(v.x,t),__fdividef(v.y,t), __fdividef(v.z,t));
		}

        //Volume
        __device__ void set_tsdf_weight(Volume::elem_type &value, float tsdf, int weight)
        {
            value.tsdf = max(-SHORTMAX, min(SHORTMAX, static_cast<int>(tsdf * SHORTMAX)));
            value.weight=weight;
        };

        template<class T>//使用在体素和点云
        __device__ void set_rgb(T &value, uchar3 rgb) { value.rgb=rgb; };
        template<class T>
        __device__ void set_point(T &value, float3 pos,float3 nor) 
        {   value.pos=pos; value.normal=nor; };
        template<class T>
        __device__ float3 get_pos(T value) { return value.pos; };
        template<class T>
        __device__ float3 get_normal(T value) { return value.normal; };
        template<class T>
        __device__ uchar3 get_rgb(T value) { return value.rgb; };

        __device__ float voxel_tsdf(Volume::elem_type value){return  static_cast<float>(value.tsdf)*DIVSHORTMAX;};
        __device__ int voxel_weight(Volume::elem_type value){return value.weight;}; 
    }
}