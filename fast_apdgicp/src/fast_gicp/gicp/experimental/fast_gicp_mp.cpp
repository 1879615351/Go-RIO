#include <fast_gicp/gicp/experimental/fast_gicp_mp.hpp>
#include <fast_gicp/gicp/experimental/fast_gicp_mp_impl.hpp>

// template class fast_gicp::FastGICPMultiPoints<pcl::PointXYZ, pcl::PointXYZ>;
template class fast_gicp::FastGICPMultiPoints<pcl::PointXYZI, pcl::PointXYZI>;
template class fast_gicp::FastGICPMultiPoints<pcl::PointXYZINormal, pcl::PointXYZINormal>;
