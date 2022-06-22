#include <depth_sensor.h>
#include <fstream>

//#ifdef KINECT2
using namespace kf;
depth_sensor::~depth_sensor()
{
	release();
}

void depth_sensor::open(const std::string _path)
{
#ifdef DATASET
	data_path = _path;
	cv::glob(data_path + "/color/*.png", img_col_name);
	cv::glob(data_path + "/depth/*.png", img_dep_name);
	if (img_col_name.size() == 0 || img_dep_name.size() == 0)
	{
		std::cout << "error: no camera!" << std::endl;
		exit(0);
	}
	// read intr
	std::ifstream intr_file(data_path + "/intr.txt");
	std::vector<float> intr_;
	if (intr_file.is_open())
	{
		float t = 0;
		for (int i = 0; i < 9; i++)
		{
			intr_file >> t;
			if (t > 0.1f)
				intr_.push_back(t);
		}
		intr_file.close();
		if (intr_.size() == 5)
		{
			params.fx = intr_[0];
			params.cx = intr_[1];
			params.fy = intr_[2];
			params.cy = intr_[3];
			params.c = intr_[4];
			cv::Mat temp_size = cv::imread(img_col_name[0], 1);
			params.height = temp_size.rows;
			params.width = temp_size.cols;
		}
	}
#endif
#ifdef KINECT2
	HRESULT hr;
	hr = GetDefaultKinectSensor(&pSensor);
	if (pSensor)
	{
		hr = pSensor->Open();
		if (SUCCEEDED(hr))
		{
			// 获取多数据源到读取器
			hr = pSensor->OpenMultiSourceFrameReader(
				FrameSourceTypes::FrameSourceTypes_Color |
					FrameSourceTypes::FrameSourceTypes_BodyIndex |
					FrameSourceTypes::FrameSourceTypes_Depth,
				&pMultiSourceFrameReader);
		}
		else
			return false;
	}
	//
	if (SUCCEEDED(hr))
		hr = pSensor->get_CoordinateMapper(&pCoordinateMapper);
	IMultiSourceFrame *pMultiFrame = nullptr;
	hr = -1;
	while (FAILED(hr))
		hr = pMultiSourceFrameReader->AcquireLatestFrame(&pMultiFrame);
	SafeRelease(pMultiFrame);
	CameraIntrinsics intr_kinect = {};
	hr = pCoordinateMapper->GetDepthCameraIntrinsics(&intr_kinect);
	//
	if (!pSensor || FAILED(hr))
	{
		std::cout << "error: no camera!" << std::endl;
		exit(0);
	}
	if (SUCCEEDED(hr))
		pSensor->get_BodyFrameSource(&pBodySource);
	if (SUCCEEDED(hr))
		hr = pBodySource->OpenReader(&pBodyReader);
	body_count = 0;
	pBodySource->get_BodyCount(&body_count);
	p_cam_space = new CameraSpacePoint[width * height];
	p_color_space = new ColorSpacePoint[width * height];
#endif
#ifdef REALSENSE
	rs2::context ctx;
	auto dev_list = ctx.query_devices();
	if (dev_list.size() == 0)
	{
		std::cout << "error: no camera!" << std::endl;
		exit(0);
	}
	rs2::device dev = *dev_list.begin();
	std::cout << dev.get_info(RS2_CAMERA_INFO_NAME) << std::endl;
	rs2::config cfg;

	cfg.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_BGR8, 30);
	cfg.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16, 30);
	//补洞滤波
	rs2::hole_filling_filter hole_filter;
	hole_filter.set_option(RS2_OPTION_HOLES_FILL, 2);
	//创建数据管道
	rs2::pipeline_profile profile = pipe.start(cfg);
	auto sensor = profile.get_device().first<rs2::depth_sensor>();
	sensor.set_option(RS2_OPTION_VISUAL_PRESET, 0);
	//获取数据流
	auto color_stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>(); //声明内参
	auto depth_stream = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>(); //声明内参
	////深度向彩色对齐
	rs2_stream align_to = RS2_STREAM_COLOR;
	align = new rs2::align(align_to);
	//深度比例
	// depth_scale = sensor.as<rs2::depth_sensor>().get_depth_scale();
	//获取内参color大，depth小
	in_par = color_stream.get_intrinsics();
	params.fx = in_par.fx;
	params.fy = in_par.fy;
	params.cx = in_par.ppx;
	params.cy = in_par.ppy;
	/*intr_.fx_inv = 1.f / intr_.fx;
	intr_.fy_inv = 1.f / intr_.fy;*/
	params.height = height;
	params.width = width;
	params.c = 1.0;
#endif
}

bool depth_sensor::getFrame()
{
#ifdef KINECT2
	IMultiSourceFrame *pMultiFrame = nullptr;
	HRESULT hr = 0;
	// 获取新的一个多源数据帧
	hr = pMultiSourceFrameReader->AcquireLatestFrame(&pMultiFrame);
	if (SUCCEEDED(hr))
		hr = pMultiFrame->get_ColorFrameReference(&pColorFrameReference);
	if (SUCCEEDED(hr))
		hr = pColorFrameReference->AcquireFrame(&pColorFrame);
	if (SUCCEEDED(hr))
		hr = pMultiFrame->get_DepthFrameReference(&pDepthFrameReference);
	if (SUCCEEDED(hr))
		hr = pDepthFrameReference->AcquireFrame(&pDepthFrame);
	if (SUCCEEDED(hr))
		hr = pColorFrame->CopyConvertedFrameDataToArray(1920 * 1080 * 4, bgra_data, ColorImageFormat::ColorImageFormat_Bgra);
	if (SUCCEEDED(hr))
		hr = pDepthFrame->CopyFrameDataToArray(width * height, depth_data);
	if (SUCCEEDED(hr))
		hr = pCoordinateMapper->MapDepthFrameToColorSpace(width * height, depth_data, width * height, p_color_space);
	////深度图像空间映射到相机空间
	cv::Mat(height, width, CV_16UC1, depth_data).convertTo(depth_map, CV_32FC1);
	// depth.convertTo(depth, CV_32FC1);
	cv::Mat color_temp = cv::Mat(1080, 1920, CV_8UC4, bgra_data);
	color_map = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
	if (SUCCEEDED(hr))
	{
		for (int i = 0; i < depth_map.rows; i++)
		{
			for (int j = 0; j < color_map.cols; j++)
			{
				ColorSpacePoint colorPoint = p_color_space[i * color_map.cols + j];
				int x = static_cast<int>(std::floor(colorPoint.X + 0.5f));
				int y = static_cast<int>(std::floor(colorPoint.Y + 0.5f));
				if ((x >= 0) && (x < color_temp.cols) && (y >= 0) && (y < color_temp.rows)) //&& (depth_ >= 800) && (depth_ <= 5000))
				{
					color_map.at<cv::Vec3b>(i, j)[0] = color_temp.at<cv::Vec4b>(y, x)[0];
					color_map.at<cv::Vec3b>(i, j)[1] = color_temp.at<cv::Vec4b>(y, x)[1];
					color_map.at<cv::Vec3b>(i, j)[2] = color_temp.at<cv::Vec4b>(y, x)[2];
				}
			}
		}
	}
	// release memory
	SafeRelease(pMultiFrame);
	SafeRelease(pColorFrame);
	SafeRelease(pDepthFrame);
	SafeRelease(pColorFrameReference);
	SafeRelease(pDepthFrameReference);
	return true;
#endif
#ifdef DATASET
	if (img_dep_name.size() == 0 || img_col_name.size() == 0)
		return false;
	std::string imgFullname = img_col_name[0];
	color_map = cv::imread(imgFullname, 1);
	imgFullname = img_dep_name[0];
	cv::imread(imgFullname, -1).convertTo(depth_map, CV_32FC1);
	img_col_name.erase(img_col_name.begin());
	img_dep_name.erase(img_dep_name.begin());
	return true;
#endif
#ifdef REALSENSE
	rs2::frameset data = pipe.wait_for_frames();
	auto processed = align->process(data);
	rs2::depth_frame depthf = processed.get_depth_frame();
	rs2::video_frame colorf = processed.get_color_frame();
	//
	// in_par =depth.get_profile().as<rs2::video_stream_profile>().get_intrinsics();
	depth_map = cv::Mat(cv::Size(width, height), CV_16UC1, (void *)depthf.get_data(), cv::Mat::AUTO_STEP);
	color_map = cv::Mat(cv::Size(width, height), CV_8UC3, (void *)colorf.get_data(), cv::Mat::AUTO_STEP);
	return true;
#endif
}
void depth_sensor::release()
{
#ifdef KINECT2
	pSensor->Close();
	SafeRelease(pCoordinateMapper);
#endif
}