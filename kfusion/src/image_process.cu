#include <device_types.hpp>
#include <safe_call.hpp>
namespace kf
{
	namespace device
	{
		using namespace cv::cuda;
		__global__ void kernal_depthTruncation(PtrStepSz<float> dmap, const float max_dist)
		{
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			int y = blockIdx.y * blockDim.y + threadIdx.y;

			if (x < dmap.cols && y < dmap.rows)
				dmap(y, x) *= 0.001f;
			if (dmap(y, x) > max_dist)
				dmap(y, x) = 0.f;
		}
		void depthTruncation(GpuMat &dmap, const float max_dist)
		{
			dim3 block(32, 8);
			dim3 grid(1, 1, 1);
			grid.x = DIVUP(dmap.cols, block.x);
			grid.y = DIVUP(dmap.rows, block.y);

			kernal_depthTruncation << <grid, block >> > (dmap, max_dist);
			cudaSafeCall(cudaGetLastError());
		}
		//
		__global__ void kernel_getVertexmap(const PtrStepSz<float> dmap, PtrStep<float3> vmap, const Intrs params)
		{
			const int x = blockIdx.x * blockDim.x + threadIdx.x;
			const int y = blockIdx.y * blockDim.y + threadIdx.y;

			if (x >= dmap.cols || y >= dmap.rows)
				return;

			float depth_value = dmap(y, x);

			if (isnan(depth_value))
				vmap(y, x) = make_float3(0.f, 0.f, 0.f);
			else
				vmap(y, x) = params.reproj(x, y, depth_value);
		}

		void getVertexmap(const GpuMat& dmap, GpuMat& vmap, const Intrs& params)
		{
			dim3 block(32, 8);
			dim3 grid(1, 1, 1);
			grid.x = DIVUP(dmap.cols, block.x);
			grid.y = DIVUP(dmap.rows, block.y);

			kernel_getVertexmap << < grid, block >> > (dmap, vmap, params);
			cudaSafeCall(cudaGetLastError());
			cudaThreadSynchronize();
		}
		//
		__global__ void kernel_getNormalmap(const PtrStepSz<float3> vmap, PtrStep<float3> nmap)         
		{
			const int x = blockIdx.x * blockDim.x + threadIdx.x;
			const int y = blockIdx.y * blockDim.y + threadIdx.y;

			if (x < 1 || x >= vmap.cols - 1 || y < 1 || y >= vmap.rows - 1)
				return;

			const float3 vex_left(vmap(y,x - 1));
			const float3 vex_right(vmap(y,x + 1));
			const float3 vex_up(vmap(y-1,x));
			const float3 vex_down(vmap(y+1,x));

			float3 normal;
			if (vex_left.z == 0 || vex_right.z == 0 || vex_up.z == 0 || vex_down.z == 0)
				normal = make_float3(0.f, 0.f, 0.f);
			else {
				float3 axis_x = make_float3(vex_left.x - vex_right.x, vex_left.y - vex_right.y, vex_left.z - vex_right.z);
				float3 axis_y = make_float3(vex_up.x - vex_down.x, vex_up.y - vex_down.y, vex_up.z - vex_down.z);

				normal = cross(axis_x, axis_y);
				float rev = -1;
				if (normal.z > 0)
					normal *= rev;
			}
			normalize(normal);
			nmap(y,x) = normal;
		}
		void getNormalmap(const GpuMat& vmag, GpuMat& nmap)
		{
			dim3 block(32, 8);
			dim3 grid(1, 1, 1);
			grid.x = DIVUP(vmag.cols, block.x);
			grid.y = DIVUP(vmag.rows, block.y);

			kernel_getNormalmap << < grid, block >> > (vmag, nmap);
			cudaSafeCall(cudaGetLastError());
		}
		__global__ void kernel_resizePointsNormals(const PtrStepSz<float3> vex_big, const PtrStep<float3> nor_big, PtrStepSz<float3> vex_small, PtrStep<float3> nor_small)
		{
			int x = threadIdx.x + blockIdx.x * blockDim.x;
			int y = threadIdx.y + blockIdx.y * blockDim.y;

			if (x >= vex_small.cols || y >= vex_small.rows)
				return;
			vex_small(y, x) = nor_small(y, x) = make_float3(0.f, 0.f, 0.f);

			int xs = x * 2;
			int ys = y * 2;

			float3 d00 = vex_big(ys + 0, xs + 0);
			float3 d01 = vex_big(ys + 0, xs + 1);
			float3 d10 = vex_big(ys + 1, xs + 0);
			float3 d11 = vex_big(ys + 1, xs + 1);

			if (!isnan(d00.x * d01.x * d10.x * d11.x))
			{
				float3 d = (d00 + d01 + d10 + d11) * 0.25f;
				vex_small(y, x) = make_float3(d.x, d.y, d.z);

				float3 n00 = nor_big(ys + 0, xs + 0);
				float3 n01 = nor_big(ys + 0, xs + 1);
				float3 n10 = nor_big(ys + 1, xs + 0);
				float3 n11 = nor_big(ys + 1, xs + 1);

				float3 n = (n00 + n01 + n10 + n11)*0.25f;
				nor_small(y, x) = make_float3(n.x, n.y, n.z);
			}
		}
		void resizePointsNormals(const GpuMat& vex_big, const GpuMat& nor_big, GpuMat& vex_small, GpuMat& nor_small)
		{
			dim3 block(32, 8);
			dim3 grid(1, 1, 1);
			grid.x = DIVUP(vex_small.cols, block.x);
			grid.y = DIVUP(vex_small.rows, block.y);
			
			kernel_resizePointsNormals << < grid, block >> > (vex_big, nor_big, vex_small, nor_small);
			cudaSafeCall(cudaGetLastError());
		}
		//
		__global__ void kernel_renderNormals(const PtrStepSz<float3> nmap, PtrStep<uchar3> cmap)
		{
			const int x = blockIdx.x * blockDim.x + threadIdx.x;
			const int y = blockIdx.y * blockDim.y + threadIdx.y;

			if (x >= nmap.cols || y >= nmap.rows)
				return;
			cmap(y,x).x=(uchar)(fabs(nmap(y,x).x)*255);
			cmap(y,x).y=(uchar)(fabs(nmap(y,x).y)*255);
			cmap(y,x).z=(uchar)(fabs(nmap(y,x).z)*255);
		}
		void renderNormals(const GpuMat &nmap, GpuMat& cmap)
		{
			dim3 block(32, 8);
			dim3 grid(1, 1, 1);
			grid.x = DIVUP(nmap.cols, block.x);
			grid.y = DIVUP(nmap.rows, block.y);
			// step 2 启动 GPU 核函数
			kernel_renderNormals << < grid, block >> > (nmap, cmap);
			cudaSafeCall(cudaGetLastError());
		}
		
		__global__ void kernel_renderPhong(const float3 poset, const PtrStepSz<float3> vmap,const PtrStepSz<float3> nmap, PtrStep<uchar3> cmap)
		{
			const int x = blockIdx.x * blockDim.x + threadIdx.x;
			const int y = blockIdx.y * blockDim.y + threadIdx.y;
			// 合法性检查: 判断是否在当前图层图像范围内
			if (x >= vmap.cols || y >= vmap.rows)
				return;
			float3 vertex = vmap(y, x);
			float3 normal = nmap(y, x);
			if (normal.x == 0 && normal.y == 0 && normal.z == 0)
				return;
			if (vertex.x == 0 && vertex.y == 0 && vertex.z == 0)
				return;

			float3 kd = make_float3(0.3843f, 0.4745f, 0.580f);
			const float3 light_pos= make_float3(500.f, 500.f, -500.f);
			const float3 eye_pos= poset;
			const float light_intensity = 0.9;
			 
		    float3 eye_pose_direction = eye_pos - vmap(y, x);
			float3 light_direction = light_pos - vmap(y, x);
			normalize(eye_pose_direction);
			normalize(light_direction);

			const float3 ambinent_light= make_float3(0.1f, 0.1f, 0.1f);
			float light_cos = dot(normal, light_direction);
			if (light_cos <= 0) {
				light_cos = -light_cos;
			}

			float light_coffi = light_intensity * light_cos;
			float3 diffuse_light = kd * light_coffi;
            float3 h_ = light_direction + eye_pose_direction;
			normalize(h_);
			float h_cos = dot(normal, h_);
			if (h_cos < 0) {
				h_cos = -h_cos;
			}

			light_coffi = light_intensity * pow(h_cos, 10);
			float specular_light_x = 0.5*light_coffi;
			float specular_light_y = 0.5*light_coffi;
			float specular_light_z = 0.5*light_coffi;

			float3 k_color = make_float3(0.f,0.f,0.f);
			k_color.x= fmin(1.f, ambinent_light.x + diffuse_light.x + specular_light_x);
			k_color.y= fmin(1.f, ambinent_light.y + diffuse_light.y + specular_light_y);
			k_color.z= fmin(1.f, ambinent_light.z + diffuse_light.z + specular_light_z);

			cmap(y, x) = make_uchar3((uchar)(k_color.x*255),
									 (uchar)(k_color.y*255),
									 (uchar)(k_color.z*255));
		}
		void renderPhong(const float3 &poset, const GpuMat &vmap,const GpuMat &nmap,GpuMat &cmap)
		{
			dim3 block(32, 8);
			dim3 grid(1, 1, 1);
			grid.x = DIVUP(vmap.cols, block.x);
			grid.y = DIVUP(vmap.rows, block.y);
			// step 2 启动 GPU 核函数
			kernel_renderPhong << < grid, block >> > (poset, vmap, nmap, cmap);
			cudaSafeCall(cudaGetLastError());
		}
	}
}