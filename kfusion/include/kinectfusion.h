#pragma once
#include <types.hpp>
#include <device_types.hpp>
#include <tsdf_volume.hpp>
#include <icp_registration.hpp>
#include <chrono>
namespace kf
{
    struct kinectfuison_params
    {
        kinectfuison_params default_params();
        ////surf meaasure
        int pyramid_height;
        float dfilter_dist;
        int bfilter_kernel_size;
        float bfilter_spatial_sigma;
        float bfilter_color_sigma;
        ////pose estimation
        float icp_dist_threshold;
        float icp_angle__threshold;
        std::vector<int> icp_iter_count;
        ////volume fusion //default the ori of volume is the center of world
        cv::Vec3f volu_range;
        cv::Affine3f volu_pose;
        float volu_trun_dist;
        float init_cam_model_dist;
        cv::Vec3i volu_dims;
        float min_pose_move;
        int tsdf_max_weight;
    };
    class kinectfusion
    {
    public:
        kinectfusion(const kf::Intrinsics intr, const kf::kinectfuison_params params);
        ~kinectfusion();

        void pipeline(cv::Mat cmap_, cv::Mat dmap_);
        void reset();
        //
        enum DISPLAY_TYPES
        {
            PHONG,
            NORMAL,
        };
        //
        cv::Mat getRenderMap(DISPLAY_TYPES V = PHONG);

        
        // cv::Mat extracePointcloud();
        cv::Mat extracePointcloud();
        void savePointcloud(std::string path);

        cv::Affine3f getCurCameraPose();
        void release();

    public:
        std::string frame_time;
        int frame_count;
        std::vector<cv::Affine3f> pose_record;
   
    private:
        void imageProcess(cv::Mat cmap_, cv::Mat dmap_);

    private:
        Frame *cframe;
        Frame *pframe;
        // cv::Mat rcmap;
        TSDFVolume *vdata;
        ICPRegistration icp;
        Intrinsics intr_;
        kinectfuison_params params_;
        cv::Mat points_array;
    };
}

// TODO:file
namespace kf
{
    namespace file
    {
        void exportPly(const std::string &filename, cv::Mat pointcloud);
    }
}