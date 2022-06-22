/*This is an kinectfusion-based 3D reconstruction system madde by Yuntao Bai.
Four modules: image process, icp, tsdf fusion and raycast 
3rdparts:
* cuda 11.3
* opencv_cuda_viz (vcpkg)

performance:
running at about 20 ms per frame in windows.

reference:
* Newcombe et al, KinectFusion: Real-time dense surface mapping and tracking
* https://github.com/PointCloudLibrary/pcl/tree/master/gpu/kinfu

email: ytb_shanghai@163.com (Yuntao Bai)
6.22.2022*/


#include <depth_sensor.h>
#include <kinectfusion.h>
#include <string>
#include <iostream>
// viz
#include <opencv2/viz/vizcore.hpp>
struct KinectFusionAPP
{
	depth_sensor *camera;
	kf::kinectfuison_params kfparams;
	kf::kinectfusion *kinfu;
	cv::viz::Viz3d mviz;

	void KeyboardCallback(int key)
	{
		switch (key)
		{
		case 27:
			exit(0);
			break;
		case 32:
			system("pause");
			break;
		case 't':
		case 'T':
			kinfu->extracePointcloud();
			kinfu->savePointcloud("../../pointcloud.ply");
			std::cout << "Save PLY" << std::endl;
			break;
		case 'e':
		case 'E':
			kinfu->reset();
			break;
		}
	}
	//
	KinectFusionAPP(depth_sensor *camera_)
	{
		camera = new depth_sensor();
		camera = camera_;
		kfparams = kfparams.default_params();
		kinfu = new kf::kinectfusion(camera->params, kfparams);
		cv::viz::WCube cube(cv::Vec3d::all(0.f), cv::Vec3d(kfparams.volu_range), true, cv::viz::Color::apricot());
		mviz.showWidget("cube", cube, kfparams.volu_pose);
		mviz.showWidget("world-coor", cv::viz::WCoordinateSystem(0.3));
	}
	bool execute()
	{
		do
		{
			if (!camera->getFrame())
			{
				std::cout << "no image!" << std::endl;
				break;
			}
			// kinectfusion
			kinfu->pipeline(camera->color_map, camera->depth_map);

			// 2D show (Phong or Normal)
			cv::imshow("Scene", kinfu->getRenderMap(kinfu->PHONG));
			cv::imshow("Depth", camera->depth_map / 4000.f);
			cv::imshow("Color", camera->color_map);

			// 3D show
			if (kinfu->frame_count % 5 == 0) //每5帧更新一次3D
			{
				mviz.showWidget("cloud", cv::viz::WCloud(kinfu->extracePointcloud()));
				mviz.showWidget("camera", viz::WCoordinateSystem(0.3), kinfu->getCurCameraPose());
			}

			KeyboardCallback(waitKey(10));
			//
			mviz.spinOnce(1, true);

		} while (!mviz.wasStopped());

		// output camera poses
		std::ofstream outfile("../../poses.txt");
		for (int i = 0; i < kinfu->pose_record.size(); i++)
			outfile << kinfu->pose_record[i].matrix << std::endl;
		outfile.close();
		std::cout << "end!" << std::endl;
		return true;
	}
	void release()
	{
		camera->release();
		kinfu->release();
	}
};

//
int main(int argc, char *argv[])
{
	std::cout << "KinectFusion: start" << std::endl;
	//初始化相機
	depth_sensor camera;
	camera.open("../../dataset");
	//初始化app
	KinectFusionAPP app(&camera);
	try
	{
		app.execute();
		app.release();
	}
	catch (const std::bad_alloc & /*e*/)
	{
		std::cout << "Bad alloc" << std::endl;
	}
	catch (const std::exception & /*e*/)
	{
		std::cout << "Exception" << std::endl;
	}
	return 0;
}