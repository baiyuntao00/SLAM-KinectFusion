#pragma once 

#include <types.hpp> 
#include <fstream>

namespace file
{
	struct PointCloud
	{
		cv::Mat vertices;
		// Normal directions
		cv::Mat normals;
		// RGB color values
		cv::Mat color;
		// Total number of valid points
		int num_points;
	};
	struct SurfaceMesh
	{
		// Triangular faces
		cv::Mat triangles;
		// Colors of the vertices
		cv::Mat colors;
		// Total number of vertices
		int num_vertices;
		// Total number of triangles
		int num_triangles;
	};
	void exportPly(const std::string& filename, const PointCloud& point_cloud);
	//void exportPly(const std::string& filename, const SurfaceMesh& point_cloud);
	////
};
#endif
