#include <device_types.hpp>
#include <cuda_array.hpp>
#include <device_memory.hpp>
#include <fstream>
namespace kf
{
	typedef cv::cuda::GpuMat GpuMat;
	class TSDFVolume
	{
	public:
		TSDFVolume(){};
		~TSDFVolume() { release(); };
		TSDFVolume(const cv::Vec3f scene_size_, const cv::Vec3i dims_);
		//
		void setTrunDist(const float v);
		void setMaxWeight(const int w);
		void setPose(const cv::Affine3f p);
		void setIntrinsics(const Intrinsics i);
		//
		void release();
		void create(const cv::Vec3i dims);
		void reset();

		void integrate(const GpuMat& dmap,const GpuMat& cmap);
		void raycast(GpuMat &vmap, GpuMat &nmap);

		void extracePointCloud(std::string path);
		//
		cv::Vec3f VoxelSize();
		cv::Vec3f SceneSize();
		cv::Vec3i Dims();
		DeviceMemory Data();
		//
	private:
		cv::Affine3f pose;
		DeviceMemory vdata;
		cv::Vec3f scene_size;
		cv::Vec3f voxel_size;
		cv::Vec3i dims;
		Intrinsics intr;
		float trun_dist;
		int max_weight;
	};  
}
//TODO:file
namespace kf
{
	namespace file
	{
		void exportPly(const std::string &filename, const device::Point3RGB*pcdata, int point_num);
	}
}
