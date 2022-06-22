/*Kinect 2.0, realsense D... and dataset*/
#pragma once
#include <types.hpp>
#define DATASET

#ifdef REALSENSE
#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#endif

#ifdef KINECT2
#include "Kinect.h"
#endif
#ifdef DATASET
#endif
//
using namespace std;
using namespace cv;
using namespace kf;
class depth_sensor
{
public:
        kf::Intrinsics params;
        cv::Mat_<cv::Vec3b> color_map;
        cv::Mat_<float> depth_map;
#ifdef KINECT2
        CameraSpacePoint *p_cam_space;
        ColorSpacePoint *p_color_space;
        static depth_sensor *currentInstance;
        const int width = 512;
        const int height = 424;
#endif
#ifdef REALSENSE
        const int width = 640;
        const int height = 480;
#endif
#ifdef DATASET
        const int width = 640;
        const int height = 480;
#endif

public:
        depth_sensor(){};
        depth_sensor(const string _path) { open(_path); };
        ~depth_sensor();
        bool getFrame();

        void open(const string _path);
        void release();

private:
#ifdef DATASET
        std::string data_path;
        vector<string> img_col_name;
        vector<string> img_dep_name;
#endif
#ifdef KINECT2
        ColorSpacePoint *pColorCoordinates = new ColorSpacePoint[height * width];
        UINT16 *depth_data = new UINT16[height * width];
        BYTE *bodyIndex_data = new BYTE[height * width];
        BYTE *bgra_data = new BYTE[1080 * 1920 * 4];
        IKinectSensor *pSensor;
        ICoordinateMapper *pCoordinateMapper = nullptr;
        IDepthFrameReference *pDepthFrameReference = nullptr;
        IColorFrameReference *pColorFrameReference = nullptr;
        IDepthFrame *pDepthFrame = nullptr;
        IColorFrame *pColorFrame = nullptr;
        IBodyFrameSource *pBodySource = nullptr;
        IBodyFrameReader *pBodyReader = nullptr;
        DepthSpacePoint *pDepthCoordinates = nullptr;
        IMultiSourceFrameReader *pMultiSourceFrameReader;
#endif
#ifdef KINECT2
        vector<string> img_col_name;
        vector<string> img_dep_name;
#endif
#ifdef REALSENSE
        rs2::pipeline pipe;
        rs2_intrinsics in_par; //相机内参
        rs2::align *align;
#endif
};

////util
template <class Interface>
inline void SafeRelease(Interface *&pInterfaceToRelease)
{
        if (pInterfaceToRelease != NULL)
        {
                pInterfaceToRelease->Release();
                pInterfaceToRelease = NULL;
        }
};
