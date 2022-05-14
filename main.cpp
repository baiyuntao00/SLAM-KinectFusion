
//
#include <MicrosoftKinect.h>
#include <kinectfusion.h>
#include <string>
#include <iostream>

void stop() { system("pause"); };
void interaction(kf::kinectfusion *kinfu)
{
	switch (cv::waitKey(1))
	{
	case 's':
		std::exit(EXIT_SUCCESS);
		break;
	case 'p':
	{
		// kinfu->extraceCloud(std::string("C:/Users/HP/Desktop/pointcloud") + std::to_string(clock()) + ".ply");
		std::cout << "successfully export pointclud!" << std::endl;
		stop();
		break;
	}
	case 'm':
	{
		// kinfu->extraceMesh(std::string("C:/Users/HP/Desktop/mesh") + std::to_string(clock()) + ".ply");
		std::cout << "successfully export mesh!" << std::endl;
		stop();
		break;
	}
	case 'r':
	{
		// kinfu->reset();
		std::cout << "Reset!" << std::endl;
		break;
	}
	}
}
//
int main(int argc, char *argv[])
{
	kf::kinectfuison_params kfparams;
	kf::MicrosoftKinect *camera = new kf::MicrosoftKinect();
	if (!camera->initMicrosoftKinect("../../dataset"))
	{
		std::exit(EXIT_SUCCESS);
	}
	kf::kinectfusion *kinfu = new kf::kinectfusion(camera->params, kfparams.default_params());
	std::cout << "KinectFusion: Start" << std::endl;
	// init
	try
	{
		while (cv::waitKey(1))
		{
			if (!camera->getFrame())
			{
				std::cout << "KinectFusion: End" << std::endl;
				std::exit(EXIT_SUCCESS);
			}
			cv::imshow("image", camera->color_map);
			kinfu->pipeline(camera->color_map, camera->depth_map);
			cv::imshow("kinfu", kinfu->getRenderMap(kinfu->RAW_DEPTH));
			//cv::imshow("raycast", kinfu->getRayCastMap());
			//break;
			interaction(kinfu);
		}
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