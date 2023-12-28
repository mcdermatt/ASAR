#include <iostream>
#include <thread>
#include <pcl/common/angles.h> // for pcl::deg2rad
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
// #include <vtk-9.2/vtkSmartPointer.h>
#include <pcl/visualization/pcl_visualizer.h> //fatal error: vtkSmartPointer.h: No such file or directory
#include <pcl/console/parse.h>
// #include <pcl/visualization/cloud_viewer.h> //fatal error: vtkSmartPointer.h: No such file or directory
// #include <vtkObjectFactoryRegistryCleanup.h> //not needed


using namespace std::chrono_literals;

int main()
{
std::cout << "at least you didn't crash" << std::endl;
}