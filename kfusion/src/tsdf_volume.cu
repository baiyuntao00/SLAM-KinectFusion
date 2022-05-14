#include <device_memory.hpp>
#include <device_types.hpp>
#include <device_utils.cuh>
#include <safe_call.hpp>

//TODO:volume fusion
namespace kf
{
namespace device
{
    using namespace cv::cuda;
    __global__ void kernal_resetVolume(cuTSDF tsdf)
    {
         int x = threadIdx.x + blockIdx.x * blockDim.x;
         int y = threadIdx.y + blockIdx.y * blockDim.y;

         cuTSDF::elem_type *beg = tsdf(x, y);
         cuTSDF::elem_type *end = beg + tsdf.znumber * tsdf.dims.z;

         for(cuTSDF::elem_type* pos = beg; pos != end; pos = tsdf.zstep(pos))
             *pos =pack_tsdf(0.f, 0);
     }
       void resetVolume(cuTSDF& vpointer)
       {
        const dim3 blocks(32, 8);
        const dim3 grids(32,32);
        kernal_resetVolume << <grids, blocks >> > (vpointer);
        cudaSafeCall (cudaGetLastError());
       }
       //integrate
       struct tsdfhelper
       {
           const cuIntrs intr;
           const cuPose pose;
           cuTSDF vpointer;
           tsdfhelper(const cuIntrs &proj_, const cuPose &pose_,const cuTSDF& vpointer_):
           intr(proj_),pose(pose_),vpointer(vpointer_){};
           __device__ void operator()(const PtrStepSz<float> dmap, PtrStepSz<uchar3> cmap) const
           {
       			const int x = blockIdx.x * blockDim.x + threadIdx.x;
       			const int y = blockIdx.y * blockDim.y + threadIdx.y;
                
       			if (x >= vpointer.dims.x || y >= vpointer.dims.y)
       				return;
       
       			float3 v_z0 = make_float3(x, y, 0) * vpointer.voxel_size;
       			float3 camera_pos =pose.R * (v_z0 - pose.t);
       			float3 zstep = make_float3(pose.R.data[2].x, pose.R.data[2].y, pose.R.data[2].z)* vpointer.voxel_size.x;
                cuTSDF::elem_type* vptr=vpointer(x,y);
                for (int z = 1; z < vpointer.dims.z; ++z)
       			{
                    vptr = vpointer.zstep(vptr);
                    camera_pos += zstep;
       				if (camera_pos.z <= 0)
       					continue;
       				const int2 uv = intr.proj(camera_pos);
       				if (uv.x < 0 || uv.x >= dmap.cols || uv.y < 0 || uv.y >= dmap.rows)
       					continue;
       				const float depth = dmap(uv.y, uv.x);
       				if (depth <= 0)
       					continue;
       				const float3 xylambda = intr.reproj(uv.x,uv.y,1.f);
       				// 计算得到公式7中的 lambda
       				const float lambda = __m_norm(xylambda);
                    const float sdf = (-1.f) * (__fdividef(1.f, lambda) * __m_norm(camera_pos) - depth);
                    if (sdf >= -vpointer.trun_dist) {
       					const float tsdf = fmin(1.f, __fdividef(sdf, vpointer.trun_dist));
       					const float pre_tsdf = tsdf_x(*vptr);
       					const int pre_weight = tsdf_y(*vptr);
       
       					const int add_weight = 1;
       
       					const int new_weight = min(pre_weight + add_weight, MAX_WEIGHT);
       					const float new_tsdf= __fdividef(__fmaf_rn(pre_tsdf, pre_weight, tsdf), pre_weight + add_weight);
                        *vptr = pack_tsdf(new_tsdf, new_weight);
                        //success!
                       }
                   }
           }
       };
       __global__ void kernel_integrate(const tsdfhelper ther, const PtrStepSz<float> dmap, PtrStepSz<uchar3> cmap) 
       { ther(dmap,cmap); };
       void integrate(const cuIntrs &intr, const cuPose &pose,cuTSDF &vpointer, const GpuMat &dmap,const GpuMat &cmap)
       {
        tsdfhelper ther(intr, pose, vpointer);

        dim3 block(32, 8);
        dim3 grid(DIVUP(vpointer.dims.x, block.x), DIVUP(vpointer.dims.y, block.y));
        kernel_integrate<<<grid, block>>>(ther, dmap,cmap);
        cudaSafeCall ( cudaDeviceSynchronize() );
       }
    };
}

//TODO:raycast
namespace kf
{
namespace device
{
       __device__ __forceinline__ void intersect(const float3 ray_org, const  float3 ray_dir, /*float3 box_min,*/ const  float3 box_max, float &tnear, float &tfar)
	    {
            const float3 box_min = make_float3(0.f, 0.f, 0.f);

            // compute intersection of ray with all six bbox planes
            float3 invR = make_float3(1.f / ray_dir.x, 1.f / ray_dir.y, 1.f / ray_dir.z);
            float3 tbot = invR * (box_min - ray_org);
            float3 ttop = invR * (box_max - ray_org);
    
            // re-order intersections to find smallest and largest on each axis
            float3 tmin = make_float3(fminf(ttop.x, tbot.x), fminf(ttop.y, tbot.y), fminf(ttop.z, tbot.z));
            float3 tmax = make_float3(fmaxf(ttop.x, tbot.x), fmaxf(ttop.y, tbot.y), fmaxf(ttop.z, tbot.z));
    
            // find the largest tmin and the smallest tmax
            tnear = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
            tfar = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));
	    }
       __device__ float interpolate(const cuTSDF& vpointer, const float3& p_voxels)
        {
            float3 cf = p_voxels;

            //rounding to negative infinity
            int3 g = make_int3(__float2int_rd (cf.x), __float2int_rd (cf.y), __float2int_rd (cf.z));

            if (g.x < 0 || g.x >= vpointer.dims.x - 1 || g.y < 0 || g.y >= vpointer.dims.y - 1 || g.z < 0 || g.z >= vpointer.dims.z - 1)
                return __m_nan();

            float a = cf.x - g.x;
            float b = cf.y - g.y;
            float c = cf.z - g.z;

            float tsdf = 0.f;
            tsdf += tsdf_x(*vpointer(g.x + 0, g.y + 0, g.z + 0)) * (1 - a) * (1 - b) * (1 - c);
            tsdf += tsdf_x(*vpointer(g.x + 0, g.y + 0, g.z + 1)) * (1 - a) * (1 - b) *      c;
            tsdf += tsdf_x(*vpointer(g.x + 0, g.y + 1, g.z + 0)) * (1 - a) *      b  * (1 - c);
            tsdf += tsdf_x(*vpointer(g.x + 0, g.y + 1, g.z + 1)) * (1 - a) *      b  *      c;
            tsdf += tsdf_x(*vpointer(g.x + 1, g.y + 0, g.z + 0)) *      a  * (1 - b) * (1 - c);
            tsdf += tsdf_x(*vpointer(g.x + 1, g.y + 0, g.z + 1)) *      a  * (1 - b) *      c;
            tsdf += tsdf_x(*vpointer(g.x + 1, g.y + 1, g.z + 0)) *      a  *      b  * (1 - c);
            tsdf += tsdf_x(*vpointer(g.x + 1, g.y + 1, g.z + 1)) *      a  *      b  *      c;
            return tsdf;
        }
    struct raycasthelper
    {
        const cuTSDF vpointer;
        const cuIntrs intr;
        const cuPose pose;
        float step_len;
        float3 voxel_size_inv;
        float3 gradient_delta;
        
        raycasthelper(const cuIntrs& reproj_,const cuTSDF& vpointer_,const cuPose& pose_):
        intr(reproj_),vpointer(vpointer_),pose(pose_){
            step_len = vpointer_.voxel_size.x;
            gradient_delta = vpointer_.voxel_size*0.5f;
            voxel_size_inv = 1.f / vpointer.voxel_size;
        };
        __device__ float voxel2tsdf(const float3& p) const
        {
            //rounding to nearest even
            int x = __float2int_rn(p.x * voxel_size_inv.x);
            int y = __float2int_rn(p.y * voxel_size_inv.y);
            int z = __float2int_rn(p.z * voxel_size_inv.z);
            if(x>=vpointer.dims.x-1||y>=vpointer.dims.y-1||z>=vpointer.dims.z-1
            ||x<1||y<1||z<1)
            {
                return __m_nan();
            }
            else
                return tsdf_x(*vpointer(x, y, z));
        };
        __device__ float3 compute_normal(const float3& p) const
        {
            float3 n;
 
            float Fx1 = interpolate(vpointer, make_float3(p.x + gradient_delta.x, p.y, p.z) * voxel_size_inv);
            float Fx2 = interpolate(vpointer, make_float3(p.x - gradient_delta.x, p.y, p.z) * voxel_size_inv);
            n.x = __fdividef(Fx1 - Fx2, gradient_delta.x);
 
            float Fy1 = interpolate(vpointer, make_float3(p.x, p.y + gradient_delta.y, p.z) * voxel_size_inv);
            float Fy2 = interpolate(vpointer, make_float3(p.x, p.y - gradient_delta.y, p.z) * voxel_size_inv);
            n.y = __fdividef(Fy1 - Fy2, gradient_delta.y);
 
            float Fz1 = interpolate(vpointer, make_float3(p.x, p.y, p.z + gradient_delta.z) * voxel_size_inv);
            float Fz2 = interpolate(vpointer, make_float3(p.x, p.y, p.z - gradient_delta.z) * voxel_size_inv);
            n.z = __fdividef(Fz1 - Fz2, gradient_delta.z);
            normalize(n);
            return n;
        };
        __device__ void operator()(PtrStepSz<float3>vmap, PtrStepSz<float3> nmap)const
        {
            {
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;
                if (x >= vmap.cols || y >= vmap.rows)
                    return;
                const float3 pixel_position =intr.reproj(x, y, 1.f);
                const float3 ray_org = pose.t;
                float3 ray_dir = pose.R * pixel_position;
                __m_normalize(ray_dir);

                float _near, _far;
                intersect(ray_org, ray_dir, vpointer.volume_range, _near, _far);
                float ray_len = fmax(_near, 0.f);
                if (ray_len >= _far)
                    return;

                const float3 vsetp = ray_dir * vpointer.voxel_size;
                ray_len += step_len;
                float3 nextp = ray_org + ray_dir * ray_len;
                float tsdf_next = voxel2tsdf(nextp);
                float3 vertex = make_float3(0, 0, 0);
                float3 normal = make_float3(0, 0, 0);
                for (; ray_len < _far; ray_len += step_len)
                {
                    nextp += vsetp;
                    float tsdf_cur = tsdf_next;
                    
                    tsdf_next = voxel2tsdf(nextp);
                    if (isnan(tsdf_next))
                        continue;
                    if (tsdf_cur < 0.f && tsdf_next > 0.f)
                        break;
                    if (tsdf_cur > 0.f && tsdf_next < 0.f)
                    {
                        float Ts = ray_len - __fdividef(vpointer.voxel_size.x * tsdf_cur,tsdf_cur-tsdf_next);
                        
                        vertex = ray_org + ray_dir * Ts;
                        normal = compute_normal(vertex);
        
                        if (!isnan(normal.x * normal.y * normal.z))
                        {
                             nmap(y,x) = normal;
                             vmap(y,x) = vertex;
                             break;
                        }
                    }
                }   
            }
        }
    };
    __global__ void kernal_raycast(const raycasthelper rcher, PtrStepSz<float3> vmap,  PtrStepSz<float3> nmap)
    { rcher(vmap, nmap); };
    void raycast(const cuIntrs&reproj, const cuPose& pose, const cuTSDF &vpointer, GpuMat&vmap, GpuMat &nmap)
    {
        dim3 block(32, 8);
        dim3 grid(DIVUP(vmap.cols, block.x),DIVUP(vmap.rows, block.y));

        raycasthelper rcher(reproj, vpointer, pose);
        
        kernal_raycast << <grid, block >> > (rcher, vmap, nmap);
        cudaSafeCall (cudaGetLastError());      
    }
};
}