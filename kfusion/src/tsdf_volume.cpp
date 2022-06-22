#include <tsdf_volume.hpp>
namespace kf
{
	DeviceMemory TSDFVolume::Data() { return vdata; };
	cv::Vec3f kf::TSDFVolume::VoxelSize() { return voxel_size; };
	cv::Vec3f kf::TSDFVolume::SceneSize() { return scene_size; };
	cv::Vec3i kf::TSDFVolume::Dims() { return dims; };
	void kf::TSDFVolume::setTrunDist(const float v) { trun_dist = v; };
	void kf::TSDFVolume::setMaxWeight(const int w) { max_weight = w; };
	void kf::TSDFVolume::setPose(const cv::Affine3f p) { volume_pose = p; };
	void kf::TSDFVolume::setIntrinsics(const Intrinsics i) { intr = i; };

	TSDFVolume::TSDFVolume(const cv::Vec3f scene_size_, const cv::Vec3i dims_) : scene_size(scene_size_),
																				 dims(dims_)
	{
		voxel_size = cv::Vec3f(scene_size_(0) / dims_(0), scene_size_(1) / dims_(1), scene_size_(2) / dims_(2));
		create(dims_);
	}

	//
	void TSDFVolume::create(const cv::Vec3i dims_)
	{
		dims = dims_;
		int voxels_number = dims_(0) * dims_(1) * dims_(2);
		vdata.create(voxels_number * sizeof(device::Voxel));
		reset();
	}
	void TSDFVolume::reset()
	{

		device::Volume vpointer(vdata.ptr<device::Volume::elem_type>(),
								device::cv2cuda(dims),
								device::cv2cuda(scene_size),
								device::cv2cuda(voxel_size));
		device::resetVolume(vpointer);
	}
	void TSDFVolume::release()
	{
		vdata.release();
	}
	//
	void TSDFVolume::integrate(const cv::Affine3f& camera_pose, const GpuMat &dmap, const GpuMat &cmap)
	{
		device::Volume vpointer(vdata.ptr<device::Volume::elem_type>(),
								device::cv2cuda(dims),
								device::cv2cuda(scene_size),
								device::cv2cuda(voxel_size));
		//修改：
		vpointer.trun_dist = trun_dist;
		cv::Affine3f vol2cam = camera_pose.inv() * volume_pose;
		device::integrate(device::Intrs(intr), device::cv2cuda(vol2cam), vpointer, dmap, cmap);
	}
	void TSDFVolume::raycast(const cv::Affine3f& camera_pose, GpuMat &vmap, GpuMat &nmap)
	{
		device::Volume vpointer(vdata.ptr<device::Volume::elem_type>(),
								device::cv2cuda(dims),
								device::cv2cuda(scene_size),
								device::cv2cuda(voxel_size));
		cv::Affine3f cam2vol = volume_pose.inv() * camera_pose;
		device::raycast(device::Intrs(intr), device::cv2cuda(cam2vol),
						device::cv2cuda(cam2vol.rotation().inv(cv::DECOMP_SVD)), vpointer, vmap, nmap);
	}
	cv::Mat TSDFVolume::fetchPointCloud()
	{
		enum
		{
			DEFAULT_CLOUD_BUFFER_SIZE = 10 * 1000 * 1000
		};
		if (cloud.empty())
			cloud.create(DEFAULT_CLOUD_BUFFER_SIZE);

		device::Volume vpointer(vdata.ptr<device::Volume::elem_type>(),
								device::cv2cuda(dims),
								device::cv2cuda(scene_size),
								device::cv2cuda(voxel_size)); 
		size_t size = device::extract_points(vpointer, cloud, device::cv2cuda(volume_pose));
		Point3Map dev_pointcloud = Point3Map((device::Point3 *)cloud.ptr(), size);
		//
		cv::Mat points_array = cv::Mat(1, (int)size, CV_32FC3);
		dev_pointcloud.download(points_array.ptr<device::Point3>());
		dev_pointcloud.release();
		cloud.release();
		return points_array;
	}
};
