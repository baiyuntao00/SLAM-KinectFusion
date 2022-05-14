#pragma once
#include <string>
#include <math.h>
#include <iostream>
#include <opencv2/opencv.hpp>
//
namespace kf
{
    typedef cv::cuda::GpuMat GpuMat;
    struct Intrinsics
    {
        int width, height;
        float fx, fy, cx, cy;
        float c = 1;
        Intrinsics level(const size_t level) const
        {
            if (level == 0)
                return *this;

            const float scale_factor = powf(0.5f, static_cast<float>(level));
            return Intrinsics{width >> level, height >> level,
                              fx * scale_factor, fy * scale_factor,
                              (cx + 0.5f) * scale_factor - 0.5f,
                              (cy + 0.5f) * scale_factor - 0.5f};
        }
    };
    struct Frame
    {
        std::vector<GpuMat> dmap;
        std::vector<GpuMat> vmap;
        std::vector<GpuMat> nmap;
        GpuMat cmap;

        Frame(const int pyr_height_, const Intrinsics &intr) : dmap(pyr_height_),
                                                               vmap(pyr_height_),
                                                               nmap(pyr_height_)
        {
            cmap = cv::cuda::createContinuous(intr.height, intr.width, CV_8UC3);
            cmap.setTo(0);
            for (int level = 0; level < pyr_height_; ++level)
            {
                // 生成对应的GpuMat数据
                vmap[level] = cv::cuda::createContinuous(intr.level(level).height, intr.level(level).width, CV_32FC3);
                nmap[level] = cv::cuda::createContinuous(intr.level(level).height, intr.level(level).width, CV_32FC3);
                dmap[level] = cv::cuda::createContinuous(intr.level(level).height, intr.level(level).width, CV_32FC1);
                // 然后清空为0
                vmap[level].setTo(0);
                nmap[level].setTo(0);
                dmap[level].setTo(0);
            } // 遍历每一层金字塔
        };
        void reset()
        {
            cmap.setTo(0);
            for (int i = 0; i < dmap.size(); i++)
            {
                dmap[i].setTo(0);
                nmap[i].setTo(0);
                vmap[i].setTo(0);
            }
        };
        void release()
        {
            cmap.release();
            for (int i = 0; i < dmap.size(); i++)
            {
                dmap[i].release();
                nmap[i].release();
                vmap[i].release();
            }
        };
        Frame& operator=(Frame&& data) noexcept
		{
			vmap = std::move(data.vmap);
			nmap = std::move(data.nmap);
			cmap = std::move(data.cmap);
			return *this;
		};
    };
    inline float deg2rad (float alpha) { return alpha * 0.017453293f; }
}
