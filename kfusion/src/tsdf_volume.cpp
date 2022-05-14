#include <tsdf_volume.hpp>
namespace kf
{
	DeviceMemory TSDFVolume::Data() { return vdata; };
	cv::Vec3f kf::TSDFVolume::VoxelSize() { return voxel_size; };
	cv::Vec3f kf::TSDFVolume::SceneSize() { return scene_size; };
	cv::Vec3i kf::TSDFVolume::Dims() { return dims; };
	void kf::TSDFVolume::setTrunDist(const float v) { trun_dist = v; };
	void kf::TSDFVolume::setMaxWeight(const int w) { max_weight = w; };
	void kf::TSDFVolume::setPose(const cv::Affine3f p) { pose = p; };
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
		vdata.create(2 * voxels_number * sizeof(unsigned short));
		reset();
	}
	void TSDFVolume::reset()
	{

		device::cuTSDF vpointer((device::cuTSDF::elem_type *)(vdata.data()),
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
	void TSDFVolume::integrate(const GpuMat &dmap, const GpuMat &cmap)
	{
		device::cuTSDF vpointer((device::cuTSDF::elem_type *)(vdata.data()),
								device::cv2cuda(dims),
								device::cv2cuda(scene_size),
								device::cv2cuda(voxel_size));
		vpointer.trun_dist = trun_dist;
		cv::Affine3f pose_inv = cv::Affine3f(pose.rotation().inv(), pose.translation());
		device::integrate(device::cuIntrs(intr), device::cv2cuda(pose_inv), vpointer, dmap, cmap);
	}
	void TSDFVolume::raycast(GpuMat &vmap, GpuMat &nmap)
	{
	    device::cuTSDF vpointer((device::cuTSDF::elem_type *)(vdata.data()),
								device::cv2cuda(dims),
								device::cv2cuda(scene_size),
								device::cv2cuda(voxel_size));
		device::raycast(device::cuIntrs(intr), device::cv2cuda(pose), vpointer, vmap, nmap);
	}
}