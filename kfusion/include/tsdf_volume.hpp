#pragma once
#include <device_types.hpp>
#include <device_array.hpp>
#include <device_memory.hpp>
#include <fstream>
namespace kf
{
	typedef cv::cuda::GpuMat GpuMat;
	typedef DeviceArray<device::Point3> Point3Map;
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

		void integrate(const cv::Affine3f& camera_pose, const GpuMat& dmap,const GpuMat& cmap);
		void raycast(const cv::Affine3f& camera_pose, GpuMat &vmap, GpuMat &nmap);

		cv::Mat fetchPointCloud();
		//
		cv::Vec3f VoxelSize();
		cv::Vec3f SceneSize();
		cv::Vec3i Dims();
		DeviceMemory Data();
		//
	private:
		cv::Affine3f volume_pose;
		DeviceMemory vdata;
		cv::Vec3f scene_size;
		cv::Vec3f voxel_size;
		cv::Vec3i dims;
		Intrinsics intr;
		Point3Map cloud;
		float trun_dist;
		int max_weight;
	};  
}