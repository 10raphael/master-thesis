#include <opencv2/opencv.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>

#include <pcl/octree/octree.h>
#include "scannet_utils.hpp"

#include <pcl/visualization/pcl_visualizer.h>

void octree_label_prop(const pcl::PointCloud<pcl::PointXYZLNormal>::Ptr &cur_cloud,
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZLNormal> &octree, pcl::PointCloud<pcl::PointXYZLNormal>::Ptr &fused_cloud,
    std::unordered_map<int, std::unordered_map<int, float>> &label_cnts, float radius, int &global_ptid){
    
    if (octree.getInputCloud() == nullptr) {
        octree.setInputCloud(fused_cloud);
        // std::cout << "INITIAL MAP SIZE: " << fused_cloud->size() << std::endl;
    }

    for (const auto &point : cur_cloud->points) {
        // fused_cloud->points.push_back(point);
        // std::cout << "points to be added: " << cur_cloud->size() << std::endl;
        octree.addPointToCloud(point, fused_cloud);
    }
    
    // https://pcl.readthedocs.io/projects/tutorials/en/latest/octree.html
    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;
    
    for (size_t i = 0; i < cur_cloud->size(); ++i) {
        pcl::PointXYZLNormal &cur_point = cur_cloud->points[i];
        // std::cout << global_ptid << std::endl;
        
        // int cur_ptid = global_ptid++;
        int cur_ptid;
        cur_ptid = global_ptid++;

        // std::cout << cur_ptid << std::endl; 
        
        Eigen::Vector3f cur_normal(cur_point.normal_x, cur_point.normal_y, cur_point.normal_z);

        if (octree.radiusSearch(cur_point, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0) {
            
            // own label
            auto label = cur_cloud->points[i].label;
            label_cnts[cur_ptid][label]++;

            // +1 for all the neighbors
            for (size_t j = 0; j < pointIdxRadiusSearch.size(); ++j) {
                // neigboring labels
                int idx = pointIdxRadiusSearch[j];
                label_cnts[idx][label]++;

                float sq_dist = std::sqrt(pointRadiusSquaredDistance[j]);

                float sigma = 100;
                float w_dist = exp(-sq_dist / (2 * sigma * sigma));
                
                // const pcl::PointXYZLNormal &neighbor_pt = fused_cloud->points[idx];
                // Eigen::Vector3f neighbor_norm(neighbor_pt.normal_x, neighbor_pt.normal_y, neighbor_pt.normal_z);

                // absolute value of normalized dot product of cur and idx normal
                // float w_norm = std::abs(cur_normal.dot(neighbor_norm) / (cur_normal.norm() * neighbor_norm.norm()));
                label_cnts[idx][label] += w_dist; // w_norm;

            }
        }
    }

    // check all pts for updates
    for (size_t i = 0; i < fused_cloud->size(); ++i) {
        pcl::PointXYZLNormal &point = fused_cloud->points[i];

        // label disdtribution of current point
        auto &cur_pt_distr = label_cnts[i];

        float max_votes = 0;
        int best_label = point.label;

        for (const auto& [label, votes] : cur_pt_distr) {
            
            if (votes > max_votes) { // besser als >=
                max_votes = votes;
                best_label = label;
                
            }
        }
        point.label = best_label;
        
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "num frames?" << std::endl;
        return -1;
    }

    int num_frames = std::stoi(argv[1]);
    std::string dataset_path = "/Users/raphael/vsc/mth/ScanNet/SensReader/python/output_scene0204_00";

    std::string mask_in = "/Users/raphael/Downloads/emsanet_out/out204/scannet/inference_outputs_r34_NBt1D/scannet/valid/scannet_semantic/pred_path_semantic/"; // pred_path_semantic
    // std::string mask_in = "/Users/raphael/vsc/mth/ScanNet/SensReader/python/output_scene0568_00/scannet-20-360p_ms_newcorr/";
    // std::string mask_in = "/Users/raphael/vsc/ScanNet/scene0549_00/scene0549_00/label_nyu40id/";

    std::string intr_path = dataset_path + "/intrinsic";
    intr intr = read_intr(intr_path);
    
    float depth_shift = 1000.0f;
    
    // propagation params
    float radius = 0.11f; // 0.11f
    float octree_res = radius/2;
    
    // filtering parameters
    float VG_radius = 0.03f;

    int SOR_MeanK = 12; // 10-50
    float SOR_StddevMulThresh = 0.01f; // 0.5-2.0
    
    double hpr_radius = 1e8;
    
    int spdf_k = 6; // min 6 
    float spdf_radius = octree_res/2; // min 0 
    float spdf_sigma = SOR_StddevMulThresh*2; // min 0 
    int spdf_itMax = 1; // min 1 

    float r_norm = 0.03f;
    
    pcl::PointCloud<pcl::PointXYZLNormal>::Ptr full_cloud(new pcl::PointCloud<pcl::PointXYZLNormal>);
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZLNormal> octree(octree_res);

    std::unordered_map<int, std::unordered_map<int, float>> label_cnts;

    std::vector<float> times;

    static int global_ptid = 0;
    for (int i = 0; i < num_frames; ++i) {
        
        if (i%5 == 0){

            std::string i_str = std::to_string(i);
            
            std::string rgb_path = dataset_path + "/color/" + i_str + ".jpg";
            std::string depth_path = dataset_path + "/depth/" + i_str + ".png"; 

            std::string mask_path = mask_in + i_str + ".png";

            std::string pose_path = dataset_path + "/pose/" + i_str + ".txt";

            // using semantic masks
            rgb_path = mask_path;

            cv::Mat rgb_img = cv::imread(rgb_path, cv::IMREAD_GRAYSCALE); // IMREAD_COLOR
            if (rgb_img.empty()) {
                std::cerr << "no image/mask: " << rgb_path << std::endl;
                continue;
            }

            cv::Mat depth_img = cv::imread(depth_path, cv::IMREAD_UNCHANGED);
            if (depth_img.empty()) {
                std::cerr << "no depth image for pcd " << depth_path << std::endl;
                continue;
            }

            Eigen::Matrix4f pose = get_pose(pose_path);
            
            Eigen::Vector3d camera_location = pose.cast<double>().block<3,1>(0, 3);

            pcl::PointCloud<pcl::PointXYZLNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZLNormal>);
            cloud->width = depth_img.cols;
            cloud->height = depth_img.rows;
            cloud->is_dense = false;
            cloud->points.resize(cloud->width * cloud->height);

            for (int y = 0; y < depth_img.rows; ++y) {
                for (int x = 0; x < depth_img.cols; ++x) {
                    pcl::PointXYZLNormal& point = cloud->at(y * depth_img.cols + x);

                    unsigned short depth_value = depth_img.at<unsigned short>(y, x);
                    if (depth_value == 0) {
                        point.x = point.y = point.z = std::numeric_limits<float>::quiet_NaN();
                    } else {
                        point.z = static_cast<float>(depth_value)/ depth_shift;
                        point.x = (x - intr.mx_depth) * point.z / intr.fx_depth;
                        point.y = (y - intr.my_depth) * point.z / intr.fy_depth;
                    }

                    Eigen::Vector4f depth_coord(point.x, point.y, point.z, 1.0f);
                    
                    // to color img frame
                    int color_x = static_cast<int>((depth_coord.x() * intr.fx_color / depth_coord.z()) + intr.mx_color);
                    int color_y = static_cast<int>((depth_coord.y() * intr.fy_color / depth_coord.z()) + intr.my_color);

                    if (color_x >= 0 && color_x < rgb_img.cols && color_y >= 0 && color_y < rgb_img.rows) {

                        char label = rgb_img.at<char>(color_y, color_x);
                        point.label = label;

                        // cv::Vec3b color = rgb_img.at<cv::Vec3b>(color_y, color_x);
                        // point.r = color[2]; 
                        // point.g = color[1]; 
                        // point.b = color[0]; 
                    } 
                }
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            // std::cout << "original # pts: " << cloud->points.size() << std::endl;

            VG_filter(cloud, VG_radius);
            SOR_filter(cloud, SOR_MeanK, SOR_StddevMulThresh);

            pcl::PointCloud<pcl::PointXYZLNormal>::Ptr vis_cloud(new pcl::PointCloud<pcl::PointXYZLNormal>);
            HPR(cloud, camera_location, hpr_radius, vis_cloud);
            // std::cout << "# pts after HPR: " << vis_cloud->points.size() << std::endl;
        
            pcl::PointCloud<pcl::PointXYZLNormal>::Ptr transf_cloud(new pcl::PointCloud<pcl::PointXYZLNormal>);
            pcl::transformPointCloud(*vis_cloud, *transf_cloud, pose);

            transf_cloud->width = transf_cloud->points.size();
            transf_cloud->height = 1;
            transf_cloud->points.resize (transf_cloud->width * transf_cloud->height);

            pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
            compute_normals(transf_cloud, r_norm, normals); 

            if (transf_cloud->points.size() != normals->points.size()) {
                std::cout << "normals not calculated" << std::endl;
            }
            
            for (int j = normals->points.size() - 1; j >= 0; --j) {
                // magnitude for the normals that are not nan are 1
                transf_cloud->points[j].normal_x = normals->points[j].normal_x;
                transf_cloud->points[j].normal_y = normals->points[j].normal_y;
                transf_cloud->points[j].normal_z = normals->points[j].normal_z;
                
                // assume pt is irrelevant if it has no normals
                // https://en.cppreference.com/w/cpp/numeric/math/isnan
                // std::cout << "Normal " << j << ": (" << transf_cloud->points[j].normal_x << ", " << transf_cloud->points[j].normal_y << ", " << transf_cloud->points[j].normal_z << ")" << std::endl;

                if (std::isnan(normals->points[j].normal_x) || std::isnan(normals->points[j].normal_y) || std::isnan(normals->points[j].normal_z)) {
                    // https://stackoverflow.com/questions/44921987/removing-points-from-a-pclpointcloudpclpointxyzrgbLPointXYZRGBLPointXYZRGBL/63109642#63109642
                    transf_cloud->points.erase(transf_cloud->points.begin() + j);
                    normals->points.erase(normals->points.begin() + j);
                }
            }
            
            std::vector<int> idx_spdf;
            pcl::PointCloud<pcl::PointXYZLNormal>::Ptr spdf_cloud = SpDF_filter(transf_cloud, spdf_k, spdf_sigma, spdf_radius, spdf_itMax, idx_spdf);
            // std::cout << "# pts after SpDF: " << spdf_cloud->points.size() << std::endl;
            // *full_cloud += *spdf_cloud;
            octree_label_prop(spdf_cloud, octree, full_cloud, label_cnts, radius, global_ptid);
            
            auto t2 = std::chrono::high_resolution_clock::now();
            float t = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
            
            std::cout << "added cloud " << i << "; new points: " << transf_cloud->points.size() << "; t=" << t/1000 << " s" << std::endl;
            
            times.push_back(t);
            
        }
    }

    full_cloud->width = full_cloud->points.size();
    full_cloud->height = 1;
    full_cloud->points.resize (full_cloud->width * full_cloud->height);

    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGBL>);
    
    color_pts(full_cloud, colored_cloud);

    // avg runtime
    float tot_t = 0.0f;
    for (float t : times) {
        tot_t += t;
    }
    float avg_t = tot_t / times.size();

    float var = 0.0f;
    for (float t : times) {
        var += (t - avg_t) * (t - avg_t);
    }
    var /= times.size();
    
    float std_dev = std::sqrt(var);

    std::cout << "full cloud size: " << colored_cloud->points.size() << std::endl;
    std::cout << "avg time per pcd: " << avg_t/1e3 << " +- " << std_dev/1e3 << "s" << std::endl;

    pcl::io::savePCDFileBinaryCompressed("0819_0549_pred.pcd", *colored_cloud);

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Full Point Cloud Viewer"));
    viewer->setBackgroundColor(255, 255, 255);
    viewer->addPointCloud<pcl::PointXYZRGBL>(colored_cloud, "full point cloud");
    viewer->spin();

    return 0;
}
