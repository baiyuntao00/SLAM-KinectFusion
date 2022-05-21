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
		vdata.create(voxels_number * sizeof(device::Voxel));
		reset();
	}
	void TSDFVolume::reset()
	{

		device::Volume vpointer((device::Volume::elem_type *)vdata.data(),
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
		device::Volume vpointer((device::Volume::elem_type *)vdata.data(),
								device::cv2cuda(dims),
								device::cv2cuda(scene_size),
								device::cv2cuda(voxel_size));
		vpointer.trun_dist = trun_dist;
		cv::Affine3f pose_inv = cv::Affine3f(pose.rotation().inv(), pose.translation());
		device::integrate(device::Intrs(intr), device::cv2cuda(pose_inv), vpointer, dmap, cmap);
	}
	void TSDFVolume::raycast(GpuMat &vmap, GpuMat &nmap)
	{
		device::Volume vpointer((device::Volume::elem_type *)vdata.data(),
								device::cv2cuda(dims),
								device::cv2cuda(scene_size),
								device::cv2cuda(voxel_size));
		device::raycast(device::Intrs(intr), device::cv2cuda(pose), vpointer, vmap, nmap);
	}
	void TSDFVolume::extracePointCloud(std::string path)
	{
		device::Volume vpointer((device::Volume::elem_type *)vdata.data(),
								device::cv2cuda(dims),
								device::cv2cuda(scene_size),
								device::cv2cuda(voxel_size));
		CudaArray<device::Point3RGB> pc(MAXPOINTNUM);
		
		device::Array<device::Point3RGB> varray((device::Point3RGB *)pc.data());
		device::extract_points(vpointer, varray);
		device::Point3RGB *hvarray=new device::Point3RGB[varray.elem_num];
		pc.deviceTohost(hvarray);
		file::exportPly(path, hvarray, varray.elem_num);
		pc.release();
		delete[] hvarray;
	}
};
// TODO: extrace point cloud
namespace kf
{
	namespace file
	{
		void exportPly(const std::string &filename, const device::Point3RGB *pcdata, int point_num)
		{
			std::ofstream file_out{filename};
			if (!file_out.is_open())
				return;
			file_out << "ply" << std::endl;
			file_out << "format ascii 1.0" << std::endl;
			file_out << "element vertex " << point_num << std::endl;
			file_out << "property float x" << std::endl;
			file_out << "property float y" << std::endl;
			file_out << "property float z" << std::endl;
			file_out << "property float nx" << std::endl;
			file_out << "property float ny" << std::endl;
			file_out << "property float nz" << std::endl;
			file_out << "property uchar red" << std::endl;
			file_out << "property uchar green" << std::endl;
			file_out << "property uchar blue" << std::endl;
			file_out << "end_header" << std::endl;

			for (int i = 0; i < point_num; i++)
			{
				float3 vertex = pcdata[i].pos;
				float3 normal = pcdata[i].normal;
				uchar3 color = pcdata[i].rgb;

				file_out << vertex.x << " " << vertex.y << " " << vertex.z << " " << normal.x << " " << normal.y << " "
						 << normal.z << " ";
				file_out << static_cast<int>(color.z) << " " << static_cast<int>(color.y) << " " << static_cast<int>(color.x) << std::endl;
			}
		}
	}
}
