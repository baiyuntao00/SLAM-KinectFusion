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
	cframe = new Frame(params_.pyramid_height, intr_);
	pframe = new Frame(params_.pyramid_height, intr_);
	frame_count = 1;
	//初始化TSDF
	vdata = new TSDFVolume(params_.volu_range, params_.volu_dims);
	vdata->setMaxWeight(params_.tsdf_max_weight);
	vdata->setTrunDist(params_.volu_trun_dist);
	vdata->setIntrinsics(intr);
	vdata->setPose(params_.volu_pose);
	//初始ICP参数
	icp = ICPRegistration(params_.icp_dist_threshold, params_.icp_angle__threshold);
	icp.setIterationNum(params_.icp_iter_count);
	icp.setIntrinsics(intr);
	//
	reset();
}
kf::kinectfusion::~kinectfusion()
{
	release();
}

cv::Mat kf::kinectfusion::getRenderMap(DISPLAY_TYPES V)
{
	cv::Mat result;
	if (V == NORMAL)
	{
		device::renderNormals(pframe->nmap[0], pframe->cmap);
		pframe->cmap.download(result);
	}
	else if (V == PHONG)
	{
		device::renderPhong(device::cv2cuda(pose_record.back().translation()), pframe->vmap[0], pframe->nmap[0], pframe->cmap);
		pframe->cmap.download(result);
	}
	return result;
}
void kf::kinectfusion::imageProcess(cv::Mat cmap_, cv::Mat dmap_)
{
	cframe->dmap[0].upload(dmap_);
	cframe->cmap.upload(cmap_);

	cv::cuda::Stream stream;
	for (int level = 1; level < params_.pyramid_height; level++)
		cv::cuda::pyrDown(cframe->dmap[level - 1], cframe->dmap[level], stream);

	for (int level = 0; level < params_.pyramid_height; level++)
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

	for (int level = 0; level < params_.pyramid_height; level++)
	{
		device::getVertexmap(cframe->dmap[level], cframe->vmap[level], device::Intrs(intr_.level(level)));
		device::getNormalmap(cframe->vmap[level], cframe->nmap[level]);
	}
}

void kf::kinectfusion::pipeline(cv::Mat cmap_, cv::Mat dmap_)
{
	// image process
	auto start_time = std::chrono::system_clock::now();
	imageProcess(cmap_, dmap_);

	if (frame_count == 1)
	{
		vdata->integrate(pose_record.back(), cframe->dmap[0], cframe->cmap);
		//
		cframe->vmap.swap(pframe->vmap);
		cframe->nmap.swap(pframe->nmap);
		frame_count++;
		cframe->reset();
		return;
	}
	///////////////////////////////////////////////////////////////////////////////////////////
	// icp:求解当前帧向之前帧转换的变换
	cv::Affine3f cam_pose;
	if (!icp.rigidTransform(cam_pose, pose_record.back(), cframe, pframe))
	{
		std::cout << "tracking fail!" << std::endl;
		reset();
		return;
	}
	//存储全局
	pose_record.push_back((pose_record.back() * cam_pose));

	///////////////////////////////////////////////////////////////////////////////////////////
	// volume
	vdata->integrate(pose_record.back(), cframe->dmap[0], cframe->cmap);

	///////////////////////////////////////////////////////////////////////////////////////////
	// raycast
	pframe->reset();
	vdata->raycast(pose_record.back(), pframe->vmap[0], pframe->nmap[0]);
	for (int level = 1; level < params_.pyramid_height; level++)
	{
		device::resizePointsNormals(pframe->vmap[level - 1],
									pframe->nmap[level - 1],
									pframe->vmap[level],
									pframe->nmap[level]);
	}

	std::chrono::duration<double, std::milli> ms = std::chrono::system_clock::now() - start_time;
	std::cout << "Frame:" << frame_count <<"||Time:" <<ms.count() << "ms" << std::endl;
	// std::cout << curpose.matrix << std::endl;
	frame_count++;
	cframe->reset();
}
cv::Affine3f kf::kinectfusion::getCurCameraPose()
{
	if (pose_record.size() > 0)
		return pose_record.back();
}
void kf::kinectfusion::reset()
{
	frame_count = 1;
	cframe->reset();
	pframe->reset();
	vdata->reset();
	pose_record.clear();
	pose_record.push_back(cv::Affine3f::Identity());
}
cv::Mat kf::kinectfusion::extracePointcloud()
{
	points_array.setTo(0);
	points_array = vdata->fetchPointCloud();
	return points_array;
}
void kf::kinectfusion::savePointcloud(std::string path)
{
	int points_num = points_array.cols;
	std::ofstream file_out{path};
	if (!file_out.is_open())
		return;
	file_out << "ply" << std::endl;
	file_out << "format ascii 1.0" << std::endl;
	file_out << "element vertex " << points_num << std::endl;
	file_out << "property float x" << std::endl;
	file_out << "property float y" << std::endl;
	file_out << "property float z" << std::endl;
	file_out << "end_header" << std::endl;
	for (int i = 0; i < points_num; i++)
	{
		cv::Vec3f p = points_array.at<cv::Vec3f>(0, i);
		file_out << p(0) << " " << p(1) << " " << p(2) << std::endl;
	}
}
kf::kinectfuison_params kf::kinectfuison_params::default_params()
{
	kf::kinectfuison_params p;
	////image process/////////////////
	p.pyramid_height = 3;
	p.bfilter_color_sigma = 10;
	p.bfilter_spatial_sigma = 10;
	p.bfilter_kernel_size = 5;
	p.dfilter_dist = 5.f;
	// icp/////////////////////
	p.icp_angle__threshold = 30.f;
	p.icp_dist_threshold = 0.015f;
	p.icp_iter_count = std::vector<int>{4, 5, 10};
	// volume process//////////
	p.volu_dims = cv::Vec3i::all(512);
	p.volu_range = cv::Vec3f::all(3.f);
	p.volu_trun_dist = 2.1f * p.volu_range(0) / p.volu_dims(0);
	p.volu_pose = cv::Affine3f().translate(cv::Vec3f(-p.volu_range[0] / 2, -p.volu_range[1] / 2, 0.5f));
	p.min_pose_move = 0.008f;
	p.tsdf_max_weight = 64;
	// volume pose
	//体积场的位置在原点。
	return p;
}
void kf::kinectfusion::release()
{
	cframe->release();
	pframe->release();
	vdata->release();
}