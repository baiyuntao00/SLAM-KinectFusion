#include <device_types.hpp>
#include "cuda_runtime_api.h"

#define MAX_WEIGHT 64
#define DIVSHORTMAX 0.0000305185f // (1.f / SHRT_MAX);
#define SHORTMAX 32767			  // SHRT_MAX;

namespace kf
{
        namespace device
        {
        //TODO:cuIntrs
        __device__ int2 cuIntrs::proj(const float3 &p) const
        {
            int2 coo;
            coo.x = __float2int_rn(__fdividef(p.x, p.z) * f.x + c.x);
            coo.y = __float2int_rn(__fdividef(p.y, p.z) * f.y + c.y);
            return coo;
        };
        __device__ float3 cuIntrs::reproj(int u, int v, float z) const
        {
                float x = __fdividef(z * (u - c.x),f.x);
                float y = __fdividef(z * (v - c.y),f.y);
                return make_float3(x, y, z);
        };

        //TODO:cuTSDF
        __device__ cuTSDF::elem_type pack_tsdf(float tsdf, int weight)
        {
            const int new_value = max(-SHORTMAX, min(SHORTMAX, static_cast<int>(tsdf * SHORTMAX)));
            return make_short2(static_cast<short>(new_value), static_cast<short>(weight));
        }
        __device__ float tsdf_x(cuTSDF::elem_type value)
        {
            return  static_cast<float>(value.x)*DIVSHORTMAX;
        }
        __device__ int tsdf_y(cuTSDF::elem_type value){return static_cast<int>(value.y);}
        
        //TODO: other
        __device__ static float __m_nan() { return __int_as_float(0x7fffffff); /*CUDART_NAN_F*/ };

        __device__ __inline__ float __m_norm(const float3 &v)
		{
			return __fsqrt_rn(dot(v, v));
		}
		__device__ __inline__ void __m_normalize(float3 &v)
		{
			float t = __fsqrt_rn(dot(v, v));
			v = make_float3(__fdividef(v.x,t),
                            __fdividef(v.y,t), 
                            __fdividef(v.z,t));
		}
        static float __m_fepsilon() { return 1.192092896e-07f/*FLT_EPSILON*/; };
        static float __m_fmin() { return 1.175494351e-38f/*FLT_MIN*/; };
        static float __m_fmax() { return 3.402823466e+38f/*FLT_MAX*/; };
        
    }
}