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
    __global__ void kernal_resetVolume(Volume vpointer)
    {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        Volume::elem_type *beg = vpointer(x, y);
        Volume::elem_type *end = beg + vpointer.znumber * vpointer.dims.z;

        for(Volume::elem_type* pos = beg; pos != end; pos = vpointer.zstep(pos))
        {
            set_tsdf_weight(*pos, 0.f, 0);
            set_rgb(*pos, make_uchar3(0,0,0));
        }
     } 
       void resetVolume(Volume& vpointer)
       {
        const dim3 blocks(32, 8);
        const dim3 grids(32,32);
        kernal_resetVolume << <grids, blocks >> > (vpointer);
        cudaSafeCall (cudaGetLastError());
       }
       //integrate
       struct tsdfhelper
       {
           const Intrs intr;
           const PoseT pose;
           Volume vpointer;
           tsdfhelper(const Intrs &proj_, const PoseT &pose_,const Volume& vpointer_):
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
                Volume::elem_type* vptr=vpointer(x,y);
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
       				const float3 xylambda = intr.reproj(uv.x,uv.y, 1.f);
       				// lambda
       				const float lambda = __m_norm(xylambda);
                    const float sdf = (-1.f) * (__fdividef(1.f, lambda) * __m_norm(camera_pos) - depth);
                    if (sdf >= -vpointer.trun_dist) {
       					const float tsdf = fmin(1.f, __fdividef(sdf, vpointer.trun_dist));
       					const float pre_tsdf = voxel_tsdf(*vptr);
       					const int pre_weight = voxel_weight(*vptr);
       
       					const int add_weight = 1;
       
       					const int new_weight = min(pre_weight + add_weight, MAX_WEIGHT);
       					const float new_tsdf= __fdividef(__fmaf_rn(pre_tsdf, pre_weight, tsdf), pre_weight + add_weight);
                        set_tsdf_weight(*vptr, new_tsdf, new_weight);

                        float thres_color=__fdividef(vpointer.trun_dist, 2);
                        if (sdf <= thres_color && sdf >= -thres_color) 
                        {
                            uchar3 model_color = get_rgb(*vptr);
                            const uchar3 pixel_cmap = cmap(uv.y, uv.x);
                            float c = __int2float_rn(new_weight + add_weight);

                            float m = new_weight * model_color.x + pixel_cmap.x;
                            model_color.x=static_cast<uchar>(__fdividef(m,c));
                            m = new_weight * model_color.y + pixel_cmap.y;
                            model_color.y=static_cast<uchar>(__fdividef(m,c));
                            m = new_weight * model_color.z + pixel_cmap.z;
                            model_color.z=static_cast<uchar>(__fdividef(m,c));
                            set_rgb(*vptr, model_color);   
                        }
                       }
                   }
           }
       };
       __global__ void kernel_integrate(const tsdfhelper ther, const PtrStepSz<float> dmap, PtrStepSz<uchar3> cmap) 
       { ther(dmap,cmap); };
       void integrate(const Intrs &intr, const PoseT &pose,Volume &vpointer, const GpuMat &dmap,const GpuMat &cmap)
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
       __device__ float interpolate(const Volume& vpointer, const float3& p_voxels)
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
            tsdf += voxel_tsdf(*vpointer(g.x + 0, g.y + 0, g.z + 0)) * (1 - a) * (1 - b) * (1 - c);
            tsdf += voxel_tsdf(*vpointer(g.x + 0, g.y + 0, g.z + 1)) * (1 - a) * (1 - b) *      c;
            tsdf += voxel_tsdf(*vpointer(g.x + 0, g.y + 1, g.z + 0)) * (1 - a) *      b  * (1 - c);
            tsdf += voxel_tsdf(*vpointer(g.x + 0, g.y + 1, g.z + 1)) * (1 - a) *      b  *      c;
            tsdf += voxel_tsdf(*vpointer(g.x + 1, g.y + 0, g.z + 0)) *      a  * (1 - b) * (1 - c);
            tsdf += voxel_tsdf(*vpointer(g.x + 1, g.y + 0, g.z + 1)) *      a  * (1 - b) *      c;
            tsdf += voxel_tsdf(*vpointer(g.x + 1, g.y + 1, g.z + 0)) *      a  *      b  * (1 - c);
            tsdf += voxel_tsdf(*vpointer(g.x + 1, g.y + 1, g.z + 1)) *      a  *      b  *      c;
            return tsdf;
        }
    struct raycasthelper
    {
        const Volume vpointer;
        const Intrs intr;
        const PoseT pose;
        float step_len;
        float3 voxel_size_inv;
        float3 gradient_delta;
        
        raycasthelper(const Intrs& reproj_,const Volume& vpointer_,const PoseT& pose_):
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
                return voxel_tsdf(*vpointer(x, y, z));
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
            __m_normalize(n);
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
    void raycast(const Intrs&reproj, const PoseT& pose, const Volume &vpointer, GpuMat&vmap, GpuMat &nmap)
    {
        dim3 block(32, 8);
        dim3 grid(DIVUP(vmap.cols, block.x),DIVUP(vmap.rows, block.y));

        raycasthelper rcher(reproj, vpointer, pose);
        
        kernal_raycast << <grid, block >> > (rcher, vmap, nmap);
        cudaSafeCall (cudaGetLastError());      
    }
};
}
//TODO: extrace point cloud
namespace kf
{
    namespace device
    {
        __global__ void extract_points_kernel(const Volume vpointer, Array<Point3RGB> varray, int *point_num)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
        
            if (x >= vpointer.dims.x - 1 || y >=  vpointer.dims.y - 1)
                return;
        
            for (int z = 0; z < vpointer.dims.z - 1; ++z) 
            {
                const float tsdf = voxel_tsdf(*vpointer(x,y,z));
                if (tsdf == 0 || tsdf <= -0.99f || tsdf >= 0.99f)
                    continue;
        
                const int wex = voxel_weight(*vpointer(x+1, y, z));
                const int wey = voxel_weight(*vpointer(x, y+1, z));
                const int wez = voxel_weight(*vpointer(x, y, z+1));
                if (wex <= 0 || wey <= 0 || wez <= 0)
                    continue;
        
                const float tsdfx = voxel_tsdf(*vpointer(x+1, y, z));
                const float tsdfy = voxel_tsdf(*vpointer(x, y+1, z));
                const float tsdfz = voxel_tsdf(*vpointer(x, y, z+1));
        
                const bool is_surface_x = ((tsdf > 0) && (tsdfx < 0)) || ((tsdf < 0) && (tsdfx > 0));
                const bool is_surface_y = ((tsdf > 0) && (tsdfy < 0)) || ((tsdf < 0) && (tsdfy > 0));
                const bool is_surface_z = ((tsdf > 0) && (tsdfz < 0)) || ((tsdf < 0) && (tsdfz > 0));
        
                if (is_surface_x || is_surface_y || is_surface_z) {
                    float3 normal;
                    normal.x = (tsdfx - tsdf);
                    normal.y = (tsdfy - tsdf);
                    normal.z = (tsdfz - tsdf);
                    if (__m_norm(normal) == 0)
                        continue;
                    __m_normalize(normal);
        
                    int count = 0;
                    if (is_surface_x) count++;
                    if (is_surface_y) count++;
                    if (is_surface_z) count++;
                    int index = atomicAdd(point_num, count);
        
                    const uchar3 color = get_rgb(*vpointer(x,y,z));

                    float3 position = make_float3((static_cast<float>(x) + 0.5f) *vpointer.voxel_size.x,
                        (static_cast<float>(y) + 0.5f) *vpointer.voxel_size.y,
                        (static_cast<float>(z) + 0.5f) *vpointer.voxel_size.z);
                    if (is_surface_x) {
                        position.x = position.x - (tsdf / (tsdfx - tsdf)) * vpointer.voxel_size.x;
                        
                        set_point(*varray(index),position, normal);
                        set_rgb(*varray(index),color);
                        index ++;
                    }
                    if (is_surface_y) {
                        position.y -= (tsdf / (tsdfy - tsdf)) *vpointer.voxel_size.y;
        
                        set_point(*varray(index),position, normal);
                        set_rgb(*varray(index),color);
                        index ++;
                    }
                    if (is_surface_z) {
                        position.z -= (tsdf / (tsdfz - tsdf)) * vpointer.voxel_size.z;
        
                        set_point(*varray(index),position, normal);
                        set_rgb(*varray(index),color);
                        index ++;
                    }
                }
            }
        }
        void extract_points(const Volume& vpointer, Array<Point3RGB> &varray)
        {
            int *dev_points_num;
            cudaMalloc(&dev_points_num, sizeof(int));
            cudaMemset(dev_points_num, 0, sizeof(int));

            dim3 threads(32, 32);
            dim3 blocks((vpointer.dims.x + threads.x - 1) / threads.x,
                (vpointer.dims.y + threads.y - 1) / threads.y);
            extract_points_kernel << <blocks, threads >> > (vpointer, 
                                                            varray,
                                                            dev_points_num);
                                                            
            cudaSafeCall(cudaGetLastError());
            cudaThreadSynchronize();

            int *points_num_label = new int;
            cudaMemcpy(points_num_label, dev_points_num, sizeof(int), cudaMemcpyDeviceToHost);
            varray.elem_num = *points_num_label;
            cudaFree(dev_points_num);
            delete[] points_num_label;
        }
    }
}