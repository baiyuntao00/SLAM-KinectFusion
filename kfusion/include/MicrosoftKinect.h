#pragma once
#include <types.hpp>
#include "Kinect.h"
//
namespace kf
{
	using namespace std;
	using namespace cv;
    class MicrosoftKinect
    {
    public:
#ifdef ENABLE_KINECT2
        CameraSpacePoint *p_cam_space;
        ColorSpacePoint *p_color_space;
        static MicrosoftKinect *currentInstance;
#else

#endif
        Intrinsics params;
        const int width = 512;
        const int height = 424;
        Mat color_map;
        Mat depth_map;

    public:
        MicrosoftKinect(){};
        ~MicrosoftKinect();
        bool getFrame();

#ifdef ENABLE_KINECT2
        bool initMicrosoftKinect();
#else
        bool initMicrosoftKinect(const string _path);
#endif

    private:
#ifdef ENABLE_KINECT2
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
#else
        vector<string> img_col_name;
        vector<string> img_dep_name;
        string data_path;
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
}