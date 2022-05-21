#include <device_memory.hpp>
#include <device_types.hpp>
#include <safe_call.hpp>

namespace kf
{
namespace device 
{

	// some icp params
    const int  BLOCK_SIZE_X=32;
    const int  BLOCK_SIZE_Y=32;
    const int  BUFFER_SIZE=27;
    const size_t BIT_LENGTH = BUFFER_SIZE* sizeof(float);

	template<int SIZE>
	static __device__ __forceinline__ void reduce(volatile double* buffer)
	{
		const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
		double value = buffer[thread_id];
		if (SIZE >= 1024) {
			if (thread_id < 512) buffer[thread_id] = value = value + buffer[thread_id + 512];
			__syncthreads();
		}
		if (SIZE >= 512) {
			if (thread_id < 256) buffer[thread_id] = value = value + buffer[thread_id + 256];
			__syncthreads();
		}
		if (SIZE >= 256) {
			if (thread_id < 128) buffer[thread_id] = value = value + buffer[thread_id + 128];
			__syncthreads();
		}
		if (SIZE >= 128) {
			if (thread_id < 64) buffer[thread_id] = value = value + buffer[thread_id + 64];
			__syncthreads();
		}
		if (thread_id < 32) {
			if (SIZE >= 64) buffer[thread_id] = value = value + buffer[thread_id + 32];
			if (SIZE >= 32) buffer[thread_id] = value = value + buffer[thread_id + 16];
			if (SIZE >= 16) buffer[thread_id] = value = value + buffer[thread_id + 8];
			if (SIZE >= 8) buffer[thread_id] = value = value + buffer[thread_id + 4];
			if (SIZE >= 4) buffer[thread_id] = value = value + buffer[thread_id + 2];
			if (SIZE >= 2) buffer[thread_id] = value = value + buffer[thread_id + 1];
		} 
	}
	__device__ bool ICP::findCoresp(const int x,const int y, float3& n, float3& d, float3& s)const
	{
		bool result=false;
		if (x < size.x && y < size.y) 
		{
			float3 normal_current;
			normal_current.x = cur_nmap(y, x).x;
			if (!isnan(normal_current.x))
			{
				float3 vertex_current = cur_vmap(y, x);

				float3 vertex_current_global = curpose.R * vertex_current + curpose.t;

				float3 vertex_current_camera = prepose.R * (vertex_current_global - prepose.t);

				int2 pixel= intr.proj(vertex_current_camera);

				if (pixel.x >= 0 && pixel.y >= 0 && pixel.x < size.x && pixel.y < size.y &&vertex_current_camera.z >= 0)
				{
					float3 normal_previous_global;

					normal_previous_global.x = pre_nmap(pixel.y, pixel.x).x;
					if (!isnan(normal_previous_global.x)) {
						// 获取对应顶点
						float3 vertex_previous_global;
						vertex_previous_global = pre_vmap(pixel.y, pixel.x);

						const float distance = norm(vertex_previous_global- vertex_current_global);
						if (distance <= max_dist_squ) {
							normal_current.y = cur_nmap(y, x).y;
							normal_current.z = cur_nmap(y, x).z;
							float3 normal_current_global = curpose.R * normal_current;

							normal_previous_global.y = pre_nmap(pixel.y, pixel.x).y;
							normal_previous_global.z = pre_nmap(pixel.y, pixel.x).z;

							float3 sinangle = cross(normal_current_global, normal_previous_global);
							const float sine = norm(sinangle);
							if (sine <= min_angle)
							{
								n = normal_previous_global;
								d = vertex_previous_global;
								s = vertex_current_global;
								result = true;
							}
						}
					};
				};
			}
		}
		return result;

	}
	__global__ void kernel_rigidICP(const ICP icphelper, float* global_buffer,const int pitch)
	{
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;
		float3 n;       // 目标点的法向, KinectFusion中为上一帧的点云对应的法向量
		float3 d;        // 目标点,      KinectFusion中为上一帧的点云
		float3 s;     // 源点,        KinectFusion中为当前帧的点云
		float row[7];

		if(icphelper.findCoresp(x,y,n,d,s))
		{
			*(float3*)&row[0] =cross(s,n);
			*(float3*)&row[3] = n;

			row[6] = dot(n, d - s);
		}
		else
			row[0] = row[1] = row[2] = row[3] = row[4] = row[5] = row[6] = 0.f;

		__shared__ double smem[BLOCK_SIZE_X * BLOCK_SIZE_Y];

		const int tid = threadIdx.y * blockDim.x + threadIdx.x;

		int shift = 0;
		for (int i = 0; i < 6; ++i) { // Rows
			for (int j = i; j < 7; ++j) { // Columns and B
				__syncthreads();
				smem[tid] = row[i] * row[j];
				__syncthreads();

				reduce<BLOCK_SIZE_X * BLOCK_SIZE_Y>(smem);
				if (tid == 0)
					global_buffer[(shift++)*pitch+gridDim.x * blockIdx.y + blockIdx.x]=smem[0];
			}
		}// 归约累加
	}

	__global__ void reduction_kernel(float* global_buffer,const int pitch, const int length, float* output)
	{
		float sum = 0.0;

		for (int t = threadIdx.x; t < length; t += 512)
			sum += *(global_buffer+blockIdx.x*pitch+t);

		__shared__ double smem[512];

		smem[threadIdx.x] = sum;

		__syncthreads();

		reduce<512>(smem);

		if (threadIdx.x == 0)
			output[blockIdx.x] = smem[0];
	};

	// 使用GPU并行计算矩阵A和向量b
	void rigidICP(const ICP& icphelper, cv::Matx66d&A, cv::Vec6d&b)
	{
		dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
		dim3 grid(static_cast<unsigned int>(std::ceil(icphelper.size.x / BLOCK_SIZE_X)),
				  static_cast<unsigned int>(std::ceil(icphelper.size.y / BLOCK_SIZE_Y)));

		size_t pitch;
		float* global_buffer;
		cudaMallocPitch((void **)&global_buffer, &pitch, BIT_LENGTH, grid.x * grid.y);

		float* d_sum_buffer;
		cudaMalloc((void**)&d_sum_buffer, BIT_LENGTH);
		float* h_sum_buffer=(float*)malloc(BIT_LENGTH);
		// 
		kernel_rigidICP << <grid, block  >> > (icphelper, global_buffer, (int)pitch);
		cudaSafeCall(cudaGetLastError());
		reduction_kernel << <27, 512 >> > (global_buffer, (int)pitch, grid.x * grid.y, d_sum_buffer);
		cudaSafeCall(cudaGetLastError());

		cudaMemcpy(h_sum_buffer, d_sum_buffer, BIT_LENGTH, cudaMemcpyDeviceToHost);

		int shift = 0;
		for (int i = 0; i < 6; ++i) { 
			for (int j = i; j < 7; ++j) { 
				double value = h_sum_buffer[shift++];
				if (j == 6)
					b(i) = value;
				else
					A(i, j) = A(j, i) = value;
			}
		}
		delete []h_sum_buffer;
		cudaFree(d_sum_buffer);
		cudaFree(global_buffer);
	}
}
}

