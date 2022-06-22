#pragma once
#include <device_types.hpp>

namespace kf
{
    typedef cv::cuda::GpuMat GpuMat;
    class ICPRegistration
    {
    public:
        ICPRegistration(){};
        ICPRegistration(const float d,const float a);
        ~ICPRegistration(){};

        void setMaxDistThres(const float max_dist_);
        void setMaxAngleThres(const float max_angle_);
        void setIterationNum(const std::vector<int> &iters);
        void setIntrinsics(const Intrinsics intrs_);

        bool rigidTransform(cv::Affine3f &curpose, const cv::Affine3f prepose, const Frame *cframe, const Frame *pframe);

    private:
        std::vector<int> iters;
        Intrinsics intrs;
        float angle_thres;
        float dist_thres;
    };
}