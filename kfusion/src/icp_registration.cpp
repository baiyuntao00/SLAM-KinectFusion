#include <icp_registration.hpp>

kf::ICPRegistration::ICPRegistration(const float d, const float a) : dist_thres(d)
{
    angle_thres = sinf(deg2rad(a));
};

void kf::ICPRegistration::setMaxDistThres(const float v) { dist_thres = v; };

void kf::ICPRegistration::setMaxAngleThres(const float v) { angle_thres = deg2rad(v); };

void kf::ICPRegistration::setIterationNum(const std::vector<int> &iters_) { iters = iters_; };

void kf::ICPRegistration::setIntrinsics(const Intrinsics intrs_) { intrs = intrs_; };

bool kf::ICPRegistration::rigidTransform(cv::Affine3f &camera_pose, const cv::Affine3f prepose, const Frame *cframe, const Frame *pframe)
{
    camera_pose.Identity();
    device::ICP icphelper(dist_thres,angle_thres);

    for (int level = iters.size() - 1; level >= 0; level--)
    {
        icphelper.cur_vmap = cframe->vmap[level];
        icphelper.cur_nmap = cframe->nmap[level];
        icphelper.pre_vmap = pframe->vmap[level];
        icphelper.pre_nmap = pframe->nmap[level];
        icphelper.setIntrs(device::Intrs(intrs.level(level)), intrs.level(level).width, intrs.level(level).height);
        for (int i = 0; i < iters[level]; i++)
        {
            cv::Matx66d A; /* 行优先 */
            cv::Vec6d b;
            icphelper.curpose = device::cv2cuda(camera_pose);
            device::rigidICP(icphelper, A, b);
            //
            double det_A = cv::determinant(A);
            if (fabs(det_A) < 1e-15 || std::isnan(det_A))
                return false;
            cv::Vec6d result;
            cv::solve(A, b, result, cv::DECOMP_SVD);
            // updata
            cv::Affine3f Tinc(cv::Vec3d(result.val), cv::Vec3d(result.val + 3));
            camera_pose = camera_pose * Tinc;
        }
    }
    return true;
}