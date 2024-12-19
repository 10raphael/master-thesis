#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>

#include <pcl/octree/octree.h>

#include "kitti_utils.hpp"

#include <pcl/visualization/pcl_visualizer.h>

void octree_label_prop(const pcl::PointCloud<pcl::PointXYZLNormal>::Ptr &cur_cloud,
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZLNormal> &octree, pcl::PointCloud<pcl::PointXYZLNormal>::Ptr &fused_cloud,
    std::unordered_map<int, std::unordered_map<u_int16_t, float>> &label_cnts, int &global_ptid, float radius, const std::vector<std::pair<uint16_t, float>>& label_score){
    
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

        int label = cur_cloud->points[i].label;
        
        float cur_score = label_score[i].second;

        // int cur_ptid;
        // cur_ptid = global_ptid++;

        // std::cout << cur_ptid << std::endl; 
        
        Eigen::Vector3f cur_normal(cur_point.normal_x, cur_point.normal_y, cur_point.normal_z);

        if (octree.radiusSearch(cur_point, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0) {
          
            // +1 for all the neighbors
            // auto label = cur_cloud->points[i].label;
            // label_cnts[idx][label]++;

            // for (const auto &idx : pointIdxRadiusSearch) {
            for (size_t j = 0; j < pointIdxRadiusSearch.size(); ++j) {
                int idx = pointIdxRadiusSearch[j];

                // uncomment this to replace the confidence scores with +1 (simple voting)
                // label_cnts[idx][label]++;
            
                label_cnts[idx][label] += cur_score;
                // https://en.wikipedia.org/wiki/Kernel_smoother
                float sq_dist = std::sqrt(pointRadiusSquaredDistance[j]);

                float sigma = 200;
                float w_dist = exp(-sq_dist / (2 * sigma * sigma));

                // const pcl::PointXYZLNormal &neighbor_pt = fused_cloud->points[idx];
                // Eigen::Vector3f neighbor_norm(neighbor_pt.normal_x, neighbor_pt.normal_y, neighbor_pt.normal_z);

                // absolute value of normalized dot product of cur and idx normal
                // float w_norm = std::abs(cur_normal.dot(neighbor_norm) / (cur_normal.norm() * neighbor_norm.norm()));
                
                // float w_tot = w_norm*w_dist; 
                label_cnts[idx][label] += w_dist; 
                
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
                // std::cout << "HERE" << std::endl;
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
    std::string in_folder = "/Users/raphael/vsc/semantic-kitti/dataset/sequences/00"; 
    std::string poses_path = "/Users/raphael/vsc/semantic-kitti/dataset/sequences/00/poses.txt"; 

    // octree tuning parameters
    float radius = 0.35f;
    float octree_res = radius/2; 

    // float filter_size = 0.025f; 

    // SpDf params
    int spdf_k = 6; // min 6
    float spdf_radius = 0.5f; // min 0
    float spdf_sigma = 0.025f; // min 0
    int spdf_itMax = 1; // min 1
    
    Eigen::Matrix4f Tr = parse_calib(in_folder + "/calib.txt");
    std::vector<Eigen::Matrix4f> poses = load_poses(poses_path, Tr);
    
    if (poses.size() == 0) {
        std::cerr << "No poses!" << std::endl;
        return -1;
    }
    
    Eigen::Matrix4f init_pose = poses[0];
    Eigen::Matrix4f init_pose_inv = init_pose.inverse();
    
    for (auto& pose : poses) {
        pose = init_pose_inv * pose;
    }

    pcl::PointCloud<pcl::PointXYZLNormal>::Ptr full_cloud(new pcl::PointCloud<pcl::PointXYZLNormal>);
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZLNormal> octree(octree_res);

    // (global_ptid, (label, count))
    std::unordered_map<int, std::unordered_map<uint16_t, float>> label_cnts;

    std::vector<float> times;

    static int global_ptid = 0;

    for (int i = 0; i < num_frames; ++i) {

        if (i >= poses.size()) {
            std::cerr << "too few frames" << std::endl;
            return -1;
        }

        // https://cplusplus.com/reference/cstdio/snprintf/

        char buffer[256];
        std::snprintf(buffer, sizeof(buffer), "%06d.bin", i);
        std::string bin_path = in_folder + "/velodyne/" + buffer;

        std::snprintf(buffer, sizeof(buffer), "%06d.label", i);
        std::string label_path = in_folder + "/predictions_1/" + buffer;

        pcl::PointCloud<pcl::PointXYZLNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZLNormal>);
        load_pcd(bin_path, label_path, cloud);

        std::vector<std::pair<uint16_t, float>> label_score = make_label_score(label_path, cloud->points.size());

        Eigen::Matrix4f pose = poses[i];

        auto t1 = std::chrono::high_resolution_clock::now();
        // std::cout << "before filter: " << cloud->points.size() << endl;
        // VG_filter(cloud, filter_size);

        std::vector<int> dist_idx;
        dist_filter(cloud, 15.0f, dist_idx);
        filter_label_score(label_score, dist_idx);
        
        // transform to one consistnet frame, everything wrt first frame

        pcl::PointCloud<pcl::PointXYZLNormal>::Ptr transf_cloud(new pcl::PointCloud<pcl::PointXYZLNormal>);
        pcl::transformPointCloud(*cloud, *transf_cloud, pose);

        std::vector<int> idx_spdf;
        pcl::PointCloud<pcl::PointXYZLNormal>::Ptr spdf_cloud = SpDF_filter(transf_cloud, spdf_k, spdf_sigma, spdf_radius, spdf_itMax, idx_spdf);
        // std::cout << spdf_cloud->points.size() << " kept points" << endl;
        filter_label_score(label_score, idx_spdf);
        updt_label_cnts(label_cnts, label_score, global_ptid);

        // NO PROPAGATION, JUST AGGREGATING LABELED CLOUDS
        // *full_cloud += *spdf_cloud;
        
        // LABEL PROPAGATION
        octree_label_prop(spdf_cloud, octree, full_cloud, label_cnts, global_ptid, radius, label_score);

        auto t2 = std::chrono::high_resolution_clock::now();

        float t = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

        times.push_back(t);

        std::cout << "added (w/propagated labels) pointcloud #" << i << "; processing t: " << t / 1000 << "s" << endl;
    }
    
    full_cloud->width = full_cloud->points.size();
    full_cloud->height = 1;
    full_cloud->points.resize (full_cloud->width * full_cloud->height);
    
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGBL>);
    
    // convert cloud from PointXYZLNormal to PointXYZRGBL, and color it
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
    std::cout << "avg time per pcd: " << avg_t/1e3 << " +- " << std_dev/1e3 << "s" << std::endl;
    
    std::string out_folder = "output";
    std::filesystem::create_directories(out_folder);
    
    std::string out_pcd = out_folder + "/0818_05_prop.pcd";
    
    if (pcl::io::savePCDFileBinaryCompressed(out_pcd, *colored_cloud) == -1) {
        PCL_ERROR("could not write .pcd \n");
        return -1;
    }

    // std::cout << "# of points label_cnts: " << label_cnts.size() << endl;

    std::cout << "Saved " << colored_cloud->points.size() << " data points to " << out_pcd << endl;
    // std::cout << out_pcd << " ; paramteres: " << radius << ", octree resolution " << octree_res << ", filter size " << filter_size << endl;
    // std::cout << out_pcd << "; SpDF paramteres: k " << spdf_k << ", radius " << spdf_radius << ", sigma " << spdf_sigma << ", itMax " << spdf_itMax << endl;
    
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("PCLVisualizer"));
    viewer->setBackgroundColor(255, 255, 255);
    viewer->addPointCloud<pcl::PointXYZRGBL>(colored_cloud, "full point cloud");
    viewer->spin();

    return 0;
}
