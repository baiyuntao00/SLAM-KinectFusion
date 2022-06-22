# Online 3D Reconstruction-KinectFusion

This is an kinectfusion-based 3D reconstruction system madde by Yuntao Bai.

3rdparts:
* cuda 11.3
* opencv_cuda_viz (vcpkg)

performance:
running at about 20 ms per frame in windows PC with NVIDIA 1650Ti.

reference:
* Newcombe et al, KinectFusion: Real-time dense surface mapping and tracking
* https://github.com/PointCloudLibrary/pcl/tree/master/gpu/kinfu

data: The dataset in "dataset" repository can be used to test the system. In addition, the system also supports the input of Kinect 2.0 and Realsense
![image3](https://github.com/baiyuntao00/KinectFusion/raw/main/doc/3D.png)
