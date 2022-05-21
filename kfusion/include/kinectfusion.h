#pragma once
#include <types.hpp>
#include <device_types.hpp>
#include <tsdf_volume.hpp>
#include <icp_registration.hpp>

namespace kf
{
    struct kinectfuison_params
    {
        kinectfuison_params default_params();
        ////surf meaasure
        int pyramidHeight;
        float dfilter_dist;
        int bfilter_kernel_size;
        float bfilter_spatial_sigma;
        float bfilter_color_sigma;
        ////pose estimation
        float icp_dist_threshold;
        float icp_angle__threshold;
        std::vector<int> icp_iter_count;
        ////volume fusion
        cv::Vec3f volu_range;
        float volu_trun_dist;
        float init_cam_model_dist;
        cv::Vec3i volu_dims;
        float min_pose_move;
        int tsdf_max_weight;
        //
        cv::Affine3f volume_pose;
    };
    class kinectfusion
    {
    public:
        kinectfusion(const kf::Intrinsics intr, const kf::kinectfuison_params params);
        ~kinectfusion();

        void pipeline(cv::Mat cmap_, cv::Mat dmap_);
        void reset();
        //
        enum DISPLAY_TYPES{
            RAYCAST_PHONG,
            RAYCAST_NORMAL,
            DEPTHMAP
        };
        cv::Mat getRenderMap(DISPLAY_TYPES V);
        

        void extracePointcloud(std::string path);
        void extraceSurfaceMesh(std::string path);
        
    public:
        int frame_count;
        float frame_time;
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
        cv::Affine3f curpose;
    };
}