#include <pcl/filters/voxel_grid.h>

#include <pcl/filters/extract_indices.h>

#include <unordered_set>

#include "pointmatcher/PointMatcher.h"

typedef PointMatcher<float> PM;
typedef PM::DataPoints DP;

// https://pcl.readthedocs.io/projects/tutorials/en/latest/voxel_grid.html
void VG_filter(const pcl::PointCloud<pcl::PointXYZLNormal>::Ptr& in_cloud, float leaf_size) {

    pcl::VoxelGrid<pcl::PointXYZLNormal> voxel_grid;
    pcl::PointCloud<pcl::PointXYZLNormal>::Ptr out_cloud(new pcl::PointCloud<pcl::PointXYZLNormal>);
    voxel_grid.setInputCloud(in_cloud);
    voxel_grid.setLeafSize(leaf_size, leaf_size, leaf_size);
    voxel_grid.filter(*in_cloud);
}

void load_pcd(const std::string &bin_path, const std::string &label_path, pcl::PointCloud<pcl::PointXYZLNormal>::Ptr &cloud) {
    // https://en.cppreference.com/w/cpp/io/ios_base/openmode
    // https://stackoverflow.com/questions/46719183/c-using-ifstream-to-read-file
    std::ifstream input(bin_path, std::ios::binary);
    std::ifstream label_input(label_path, std::ios::binary);

    if (!input) {
        std::cerr << ".bin folder? " << bin_path << std::endl;
        return;
    }
    if (!label_input) {
        std::cerr << ".label folder? " << label_path << std::endl;
        return;
    }
    
    float point[4];
    uint32_t full_label;

    // https://hackingcpp.com/cpp/std/file_streams.html
    while (input.read(reinterpret_cast<char*>(&point), sizeof(point)) && label_input.read(reinterpret_cast<char*>(&full_label), sizeof(uint32_t))){ 
        pcl::PointXYZLNormal pcl_pt;
        pcl_pt.x = point[0];
        pcl_pt.y = point[1];
        pcl_pt.z = point[2];

        // https://stackoverflow.com/questions/44098765/split-unsigned-32-bit-integer-into-two-16-bit-numbers-that-can-be-rebuilt
        // 32 bit, 16 bit label, 16 bit instance id,
        uint16_t label = full_label & 0xFFFF; 
        // uint16_t instance_id = full_label >> 16;
        
        pcl_pt.label = label;

        cloud->push_back(pcl_pt);
    }
    input.close();
    label_input.close();
}

Eigen::Matrix4f parse_calib(const std::string& path) {
    std::ifstream file(path);
    Eigen::Matrix4f Tr = Eigen::Matrix4f::Identity();

    // https://www.geeksforgeeks.org/read-a-file-line-by-line-in-cpp/
    if (!file.is_open()) {
        std::cerr << "no calib.txt, check path: " << path << std::endl;
        return Tr;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.rfind("Tr:", 0) == 0) {

            // https://cplusplus.com/reference/string/string/substr/

            std::istringstream ss(line.substr(4)); // skip Tr:_
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 4; ++j) {
                    ss >> Tr(i, j);
                }
            }
            Tr(3, 3) = 1.0;
            break;
        }
    }

    file.close();
    return Tr;
}

// get poses from poses.txt
std::vector<Eigen::Matrix4f> load_poses(const std::string& path, Eigen::Matrix4f& Tr){ 
    std::vector<Eigen::Matrix4f> poses;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "no poses.txt, check path: " << path << std::endl;
        return poses;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                ss >> pose(i, j);
            }
        }

        pose = Tr.inverse() * pose * Tr;
        poses.push_back(pose);
    }

    file.close();
    return poses;
}

// https://github.com/PRBonn/semantic-kitti-api/blob/6379deb98c82f36902ccc35301baf28cc5cd1db0/config/semantic-kitti.yaml#L37

std::unordered_map<uint16_t, std::vector<int>> color_map = {
    {0, {0, 0, 0}}, {1, {0, 0, 255}}, {10, {245, 150, 100}}, {11, {245, 230, 100}}, {13, {250, 80, 100}}, {15, {150, 60, 30}}, {16, {255, 0, 0}},
    {18, {180, 30, 80}}, {20, {255, 0, 0}}, {30, {30, 30, 255}}, {31, {200, 40, 255}}, {32, {90, 30, 150}}, {40, {255, 0, 255}}, {44, {255, 150, 255}},
    {48, {75, 0, 75}}, {49, {75, 0, 175}}, {50, {0, 200, 255}}, {51, {50, 120, 255}}, {52, {0, 150, 255}}, {60, {170, 255, 150}}, {70, {0, 175, 0}},
    {71, {0, 60, 135}}, {72, {80, 240, 150}}, {80, {150, 240, 255}}, {81, {0, 0, 255}}, {99, {255, 255, 50}}, {252, {245, 150, 100}},
    {253, {200, 40, 255}}, {254, {30, 30, 255}}, {255, {90, 30, 150}}, {256, {255, 0, 0}}, {257, {250, 80, 100}}, {258, {180, 30, 80}}, {259, {255, 0, 0}}
};

void color_pts(pcl::PointCloud<pcl::PointXYZLNormal>::Ptr &in_cloud,
                 pcl::PointCloud<pcl::PointXYZRGBL>::Ptr &color_cloud){
    
    // CHANGE POINT TYPE
    color_cloud->resize(in_cloud->size());

    // COLOR THE CLOUS
    for (size_t i = 0; i < in_cloud->size(); ++i) {
        pcl::PointXYZRGBL point;
        point.x = in_cloud->points[i].x;
        point.y = in_cloud->points[i].y;
        point.z = in_cloud->points[i].z;
        point.label = in_cloud->points[i].label;

        uint16_t label = in_cloud->points[i].label;
        
        if (color_map.find(label) != color_map.end()) {
            // BGR map
            point.r = color_map.at(label)[2];
            point.g = color_map.at(label)[1];
            point.b = color_map.at(label)[0];
        } else {
            std::cout << "point not colored!" << std::endl;
            point.r = 0;
            point.g = 0;
            point.b = 0;
        }
        
        color_cloud->points[i] = point;
    }
}

// https://github.com/norlab-ulaval/libpointmatcher/blob/master/doc/DataFilters.md#spdfhead
pcl::PointCloud<pcl::PointXYZLNormal>::Ptr SpDF_filter(const pcl::PointCloud<pcl::PointXYZLNormal>::Ptr& transf_cloud,
    int k, float sigma, float radius, int itMax, std::vector<int>& idx_kept) {
    auto t1 = std::chrono::high_resolution_clock::now();
    
    DP cloud;
    cloud.features = PM::Matrix::Zero(3, transf_cloud->size());
    
    for (size_t i = 0; i < transf_cloud->size(); ++i) {
        cloud.features.col(i) << (*transf_cloud)[i].x, (*transf_cloud)[i].y, (*transf_cloud)[i].z;
    }
    
    auto spdf = PM::get().DataPointsFilterRegistrar.create(
        "SpectralDecompositionDataPointsFilter",
        {   
            {"k", std::to_string(k)},         
            {"sigma", std::to_string(sigma)}, 
            {"radius", std::to_string(radius)},
            {"itMax", std::to_string(itMax)},

            {"keepNormals", "1"},
            {"keepLabels", "1"},
            {"keepLambdas", "0"},
            {"keepTensors", "0"},
        }
    );
    
    DP filtered_cloud = spdf->filter(cloud);
    
    pcl::PointCloud<pcl::PointXYZLNormal>::Ptr spdf_cloud(new pcl::PointCloud<pcl::PointXYZLNormal>);
    spdf_cloud->width = filtered_cloud.features.cols();
    spdf_cloud->height = 1; 
    spdf_cloud->is_dense = false;
    spdf_cloud->points.resize(spdf_cloud->width * spdf_cloud->height);
    
    // copy coordinates
    for (size_t i = 0; i < filtered_cloud.features.cols(); ++i) {
        spdf_cloud->points[i].x = filtered_cloud.features(0, i);
        spdf_cloud->points[i].y = filtered_cloud.features(1, i);
        spdf_cloud->points[i].z = filtered_cloud.features(2, i);
    }

    // copy normals
    for (size_t i = 0; i < filtered_cloud.descriptors.cols(); ++i) {
        spdf_cloud->points[i].normal_x = filtered_cloud.descriptors(0, i);
        spdf_cloud->points[i].normal_y = filtered_cloud.descriptors(1, i);
        spdf_cloud->points[i].normal_z = filtered_cloud.descriptors(2, i);
        
        // curvature: 3 is surface, 2 is edge, 1 is vertex
    }
    

    auto t2 = std::chrono::high_resolution_clock::now();
    float t_filter = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

    idx_kept.clear();

    size_t idx_spdf = 0, idx_transf = 0;
        while (idx_spdf < spdf_cloud->points.size() && idx_transf < transf_cloud->points.size()) {
            if (spdf_cloud->points[idx_spdf].x == transf_cloud->points[idx_transf].x &&
                spdf_cloud->points[idx_spdf].y == transf_cloud->points[idx_transf].y &&
                spdf_cloud->points[idx_spdf].z == transf_cloud->points[idx_transf].z) {
                spdf_cloud->points[idx_spdf].label = transf_cloud->points[idx_transf].label;

                idx_kept.push_back(idx_transf);
                ++idx_spdf; 
                ++idx_transf; 
            } else {
                ++idx_transf; 
            }
        }

    auto t3 = std::chrono::high_resolution_clock::now();

    float t_copy = std::chrono::duration_cast<std::chrono::milliseconds>( t3 - t2 ).count();

    // std::cout << "time copy to out_cloud: " << t_copy / 1000 << "s" << std::endl;

    // std::cout << "idx_kept: " << idx_kept.size() << std::endl;

    return spdf_cloud;
}

// https://stackoverflow.com/a/48595186
void dist_filter(pcl::PointCloud<pcl::PointXYZLNormal>::Ptr& cloud, float max_dist, std::vector<int>& idx_kept){
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    pcl::ExtractIndices<pcl::PointXYZLNormal> extract;

    idx_kept.clear();

    for (size_t i = 0; i < cloud->points.size(); ++i)
    {
        const auto& point = cloud->points[i];
        float dist = sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
        if (dist <= max_dist)
        {
            inliers->indices.push_back(i);
        }
    }
    
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(false);  
    extract.filter(*cloud);

    idx_kept = inliers->indices;
}

std::vector<std::pair<uint16_t, float>> make_label_score(const std::string& label_path, size_t cloud_size) {
    std::vector<std::pair<uint16_t, float>> label_score;

    std::ifstream label_input(label_path, std::ios::binary);
    if (!label_input) {
        std::cerr << "no labels? " << label_path << std::endl;
        return label_score;
    }
    
    std::string conf_path = label_path + "c";
    std::ifstream conf_file(conf_path, std::ios::binary);
    if (!conf_file) {
        std::cerr << "no conf scores?" << conf_path << std::endl;
        return label_score;
    }

    uint32_t full_label;
    float conf_score;
    for (size_t i = 0; i < cloud_size; ++i) {
        if (label_input.read(reinterpret_cast<char*>(&full_label), sizeof(uint32_t)) && 
            conf_file.read(reinterpret_cast<char*>(&conf_score), sizeof(float))) {
            
            uint16_t label = full_label & 0xFFFF;
            label_score.emplace_back(label, conf_score);

        } else {
            std::cerr << "no conf scores?" << i << std::endl;
            break;
        }
    }

    return label_score;
}

void updt_label_cnts(std::unordered_map<int,std:: unordered_map<uint16_t, float>>& label_cnts,
    const std::vector<std::pair<uint16_t, float>>& label_score, int& pt_id) {

    for (const auto& [label, score] : label_score) {
        label_cnts[pt_id][label] = score;
        // comment out  line above and comment in line below if run without conf scores
        // label_cnts[pt_id][label]++;
        ++pt_id; 
    }
}

void filter_label_score(std::vector<std::pair<uint16_t, float>>& label_score, const std::vector<int>& filter_idx) {
        
    std::unordered_set<int> idx_kept(filter_idx.begin(), filter_idx.end());
    
    // https://en.cppreference.com/w/cpp/algorithm/remove
    auto it = remove_if(
        label_score.begin(), 
        label_score.end(),
        [&idx_kept, i = 0](const std::pair<uint16_t, float>&) mutable {
            return !idx_kept.count(i++);
        }
    );
    
    label_score.erase(it, label_score.end());
}


