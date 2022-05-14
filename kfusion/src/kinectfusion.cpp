#include <kinectfusion.h>
#include <opencv2/opencv.hpp>
#include "opencv2/core/cuda.hpp"
// bilateralFilte()
#include <opencv2/cudaimgproc.hpp>
// pyrDown()
#include <opencv2/cudawarping.hpp>

kf::kinectfusion::kinectfusion(const kf::Intrinsics intr, const kf::kinectfuison_params params) : intr_(intr), params_(params)
{
	//初始化视频帧
	cframe = new Frame(params_.pyramidHeight, intr_);
	pframe = new Frame(params_.pyramidHeight, intr_);
	frame_count = 0;
	frame_time = 0;
	//初始化TSDF
	vdata = new TSDFVolume(params_.volu_range, params_.volu_dims);
	vdata->setMaxWeight(params_.tsdf_max_weight);
	vdata->setTrunDist(params_.volu_trun_dist);
	vdata->setIntrinsics(intr);
	//初始相机位置
	curpose.Identity();
	curpose.matrix(0, 3) = vdata->Dims()(0) / 2 * vdata->VoxelSize()(0);
	curpose.matrix(1, 3) = vdata->Dims()(1) / 2 * vdata->VoxelSize()(1);
	curpose.matrix(2, 3) = vdata->Dims()(2) / 2 * vdata->VoxelSize()(2) - params_.init_cam_model_dist;
	//初始ICP参数
	icp = ICPRegistration(params_.icp_dist_threshold, params_.icp_angle__threshold);
	icp.setIterationNum(params_.icp_iter_count);
	icp.setIntrinsics(intr);
}
kf::kinectfusion::~kinectfusion()
{
	cframe->release();
	pframe->release();
	vdata->release();
}
cv::Mat kf::kinectfusion::getRenderMap(DISPLAY_TYPES V)
{
	cv::Mat result;
	if (V == RAW_NORMALS)
	{
		device::renderNormals(pframe->nmap[0], pframe->cmap);
		pframe->cmap.download(result);
	}
	else if (V == RAW_DEPTH)
	{
		device::renderPhong(device::cv2cuda(curpose.translation()), pframe->vmap[0], pframe->nmap[0], pframe->cmap);
		pframe->cmap.download(result);
	}
	else if (V == RAYCAST)
	{
		//
	}
	else if (V == DEPTHMAP)
	{
		cframe->dmap[0].download(result);
		result /= params_.dfilter_dist;
	}
	return result;
}
void kf::kinectfusion::imageProcess(cv::Mat cmap_, cv::Mat dmap_)
{
	cframe->dmap[0].upload(dmap_);
	cframe->cmap.upload(cmap_);

	cv::cuda::Stream stream;
	for (int level = 1; level < params_.pyramidHeight; level++)
		cv::cuda::pyrDown(cframe->dmap[level - 1], cframe->dmap[level], stream);
	for (int level = 0; level < params_.pyramidHeight; level++)
	{
		GpuMat tempMat = cframe->dmap[level];
		cv::cuda::bilateralFilter(tempMat, cframe->dmap[level],
								  params_.bfilter_kernel_size,
								  params_.bfilter_color_sigma,
								  params_.bfilter_spatial_sigma,
								  cv::BORDER_DEFAULT, stream);

		device::depthTruncation(cframe->dmap[level], params_.dfilter_dist);
		tempMat.release();
	}
	stream.waitForCompletion();
	// cuda::waitAllDefaultStream();

	for (int level = 0; level < params_.pyramidHeight; level++)
	{
		device::getVertexmap(cframe->dmap[level], cframe->vmap[level], device::cuIntrs(intr_.level(level)));
		device::getNormalmap(cframe->vmap[level], cframe->nmap[level]);
	}
}

void kf::kinectfusion::pipeline(cv::Mat cmap_, cv::Mat dmap_)
{
	clock_t time_start = clock();
	// image process
	imageProcess(cmap_, dmap_);
	// icp
	if (frame_count > 0)
	{

		if (!icp.rigidTransform(curpose, cframe, pframe))
		{
			std::cout << "tracking fail!" << std::endl;
			return;
		}
		else
		{
			std::cout << curpose.matrix << std::endl;
		}
	}
	// volume
	vdata->setPose(curpose);
	vdata->integrate(cframe->dmap[0], cframe->cmap);
	// raycast
	pframe->reset();
	vdata->raycast(pframe->vmap[0], pframe->nmap[0]);
	for (int level = 1; level < params_.pyramidHeight; level++)
	{
		device::resizePointsNormals(pframe->vmap[level - 1],
									pframe->nmap[level - 1],
									pframe->vmap[level],
									pframe->nmap[level]);
	}

	frame_count++;
	std::cout << "KinectFusion: time:" << clock() - time_start << " ms" << std::endl;
	cframe->reset();
}

void kf::kinectfusion::extracePointcloud(std::string path)
{
	int pointcloud_buffer_size = 3 * 2000000;
	file::PointCloud cloud_data = device::extract_points(volume_, pointcloud_buffer_size);
	file::exportPly(file_name, cloud_data);
}
kf::kinectfuison_params kf::kinectfuison_params::default_params()
{
	kf::kinectfuison_params p;
	////image process/////////////////
	p.pyramidHeight = 3;
	p.bfilter_color_sigma = 10;
	p.bfilter_spatial_sigma = 10;
	p.bfilter_kernel_size = 5;
	p.dfilter_dist = 5.f;
	// icp/////////////////////
	p.icp_angle__threshold = 30.f;
	p.icp_dist_threshold = 0.015f;
	p.icp_iter_count = std::vector<int>{4, 5, 10};
	p.init_cam_model_dist = 1.f;
	// volume process//////////
	p.volu_dims = cv::Vec3i::all(512);
	p.volu_range = cv::Vec3f::all(3.f);
	p.volu_trun_dist = 2.1f * p.volu_range(0) / p.volu_dims(0);
	p.min_pose_move = 0.008f;
	p.tsdf_max_weight = 64;
	// volume pose
	p.volume_pose.Identity();
	// p.volume_pose = cv::Affine3f().translate((-p.volu_range[0] / 2, -p.volu_range[1] / 2, 0.5f));
	return p;
}