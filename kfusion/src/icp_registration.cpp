#include <icp_registration.hpp>

kf::ICPRegistration::ICPRegistration(const float d, const float a) : dist_thres(d)
{
    angle_thres = sinf(deg2rad(a));
};

void kf::ICPRegistration::setMaxDistThres(const float v) { dist_thres = v; };

void kf::ICPRegistration::setMaxAngleThres(const float v) { angle_thres = deg2rad(v); };

void kf::ICPRegistration::setIterationNum(const std::vector<int> &iters_) { iters = iters_; };

void kf::ICPRegistration::setIntrinsics(const Intrinsics intrs_) { intrs = intrs_; };

bool kf::ICPRegistration::rigidTransform(cv::Affine3f &curpose, const Frame *cframe, const Frame *pframe)
{
    cv::Matx33f R = curpose.rotation();
    cv::Vec3f t = curpose.translation();

    device::cuICP icphelper(dist_thres,
                            angle_thres,
                            device::cv2cuda(R.inv(), t));

    for (int level = iters.size() - 1; level >= 0; level--)
    {
        icphelper.cur_vmap = cframe->vmap[level];
        icphelper.cur_nmap = cframe->nmap[level];
        icphelper.pre_vmap = pframe->vmap[level];
        icphelper.pre_nmap = pframe->nmap[level];
        icphelper.setIntrs(device::cuIntrs(intrs.level(level)), intrs.level(level).width, intrs.level(level).height);
        for (int i = 0; i < iters[level]; i++)
        {
            cv::Matx66d A; /* 行优先 */
            cv::Vec6d b;
            icphelper.curpose = device::cv2cuda(R, t);
            device::rigidICP(icphelper, A, b);
            //
            double det_A = cv::determinant(A);
            if (fabs(det_A) < 1e-15 || std::isnan(det_A))
                return false;

            cv::Vec6d result;
            cv::solve(A, b, result, cv::DECOMP_SVD);
            float rx = result(0);
            float ry = result(1);
            float rz = result(2);
            // Update rotation -- 恢复出原始的旋转增量
            cv::Matx33f rotaX(1, 0, 0,
                              0, cos(rx), -sin(rx),
                              0, sin(rx), cos(rx));
            cv::Matx33f rotaY(cos(ry), 0, sin(ry),
                              0, 1, 0,
                              -sin(ry), 0, cos(ry));
            cv::Matx33f rotaZ(cos(rz), -sin(rz), 0,
                              sin(rz), cos(rz), 0,
                              0, 0, 1);
            //
            cv::Matx33f rota = rotaZ * rotaY * rotaX;

            cv::Vec3f cam_tran_incremental(result(3), result(4), result(5));
            // updata
            t = rota * t + cam_tran_incremental;
            R = rota * R;
        }
    }
    curpose = cv::Affine3f(R, t);
    return true;
}