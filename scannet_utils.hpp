#include <pcl/surface/convex_hull.h>
#include <unordered_set>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <pcl/features/normal_3d.h>

#include "pointmatcher/PointMatcher.h"

typedef PointMatcher<float> PM;
typedef PM::DataPoints DP;

// https://github.com/ScanNet/ScanNet/blob/fcaa1773a9e186b22a4228df632b6999a1633b79/BenchmarkScripts/ScanNet200/scannet200_constants.py#L8

Eigen::Matrix4f get_pose(const std::string& pose_file) {
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    std::ifstream file(pose_file);
    if (file.is_open()) {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                file >> pose(i, j);
            }
        }
    } else {
        std::cerr << "no pose file " << pose_file << std::endl;
    }
    return pose;
}

// https://github.com/pdhimal1/HPR/blob/40253f02bcf79cf00f6392dd7d92e4793f16c62b/HPR-PCL-QT/pclviewer.cpp#L79
void HPR(pcl::PointCloud<pcl::PointXYZLNormal>::Ptr cloud, Eigen::Vector3d camera_location, double init_radius, 
    pcl::PointCloud<pcl::PointXYZLNormal>::Ptr& cloud_out) {

    static double radius = init_radius;

    std::vector<Eigen::Vector3d> spherical_projection;

    // Perform spherical projection
    for (size_t pidx = 0; pidx < cloud->points.size(); ++pidx) {
        
        pcl::PointXYZLNormal currentPoint = cloud->points[pidx];
        Eigen::Vector3d currentVector(currentPoint.x, currentPoint.y, currentPoint.z);
        
        Eigen::Vector3d projected_point = currentVector - camera_location;

        double norm = projected_point.norm();
        if (norm == 0){
            norm = 0.0001;
        }
        
        spherical_projection.push_back(projected_point + 2 * (radius - norm) * projected_point / norm);
    }

    // add the last point, may be this is not necessary?
    // spherical_projection.push_back(Eigen::Vector3d(0, 0, 0));

    // This is just adding the points after performing spherical inversion to the newCloud
    pcl::PointCloud<pcl::PointXYZLNormal>::Ptr newCloud(new pcl::PointCloud<pcl::PointXYZLNormal>);
    for (const auto& pos : spherical_projection) {
        // pcl::PointXYZRGBL point(vec.x(), vec.y(), vec.z());
        pcl::PointXYZLNormal currentPoint;
        
        currentPoint.x = pos.x();
        currentPoint.y = pos.y();
        currentPoint.z = pos.z();
        // currentPoint.label = label;
        
        newCloud->push_back(currentPoint);
    }

    // std::cout << "#pts in newCloud: " << newCloud->size() << "; " << std::endl;

    // Compute the convex hull
    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull(new pcl::PointCloud <pcl::PointXYZ>);
    pcl::ConvexHull<pcl::PointXYZLNormal> chull;
    chull.setInputCloud(newCloud);
    // chull.reconstruct(*cloud_hull);

    std::vector<pcl::Vertices> hull_vertices;
    chull.reconstruct(hull_vertices);

    std::unordered_set<int> unq_idx;
    for (const auto& v : hull_vertices) {
        for (const auto& idx : v.vertices) {
            
            if (unq_idx.find(idx) == unq_idx.end()) {
                cloud_out->push_back(cloud->points[idx]);
                unq_idx.insert(idx);
            }
        }
    }

    auto diff_pts = cloud->size() - cloud_out->size();
    
    // if too many points not visible, increase radius x10
    if (diff_pts > 0.66*cloud->size()){
        radius *= 10;
        std::cout << "new radius: " << radius << std::endl;
    }
    

    // std::cout << "radius: " << radius << std::endl;
    // std::cout << "Original Cloud size " << cloud->points.size() << std::endl;
    // std::cout << "New Cloud's size " << cloud_out->points.size() << std::endl;
}

// https://pcl.readthedocs.io/projects/tutorials/en/latest/voxel_grid.html
void VG_filter(const pcl::PointCloud<pcl::PointXYZLNormal>::Ptr& in_cloud, float leaf_size) {

    pcl::VoxelGrid<pcl::PointXYZLNormal> voxel_grid;
    pcl::PointCloud<pcl::PointXYZLNormal>::Ptr out_cloud(new pcl::PointCloud<pcl::PointXYZLNormal>);
    voxel_grid.setInputCloud(in_cloud);
    voxel_grid.setLeafSize(leaf_size, leaf_size, leaf_size);
    voxel_grid.filter(*in_cloud);
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
    // times.push_back(t);

    float t_copy = std::chrono::duration_cast<std::chrono::milliseconds>( t3 - t2 ).count();

    // std::cout << "time copy to out_cloud: " << t_copy / 1000 << "s" << std::endl;
    // std::cout << "idx_kept: " << idx_kept.size() << std::endl;

    return spdf_cloud;
}

// https://github.com/ScanNet/ScanNet/blob/fcaa1773a9e186b22a4228df632b6999a1633b79/BenchmarkScripts/ScanNet200/scannet200_constants.py#L2
const std::vector<uint16_t> VALID_CLASS_IDS_20 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39};

const std::map<uint16_t, std::tuple<float, float, float>> SCANNET_COLOR_MAP_20 = {
    {0, {0., 0., 0.}}, {1, {174., 199., 232.}}, {2, {152., 223., 138.}}, {3, {31., 119., 180.}}, {4, {255., 187., 120.}}, {5, {188., 189., 34.}},
    {6, {140., 86., 75.}}, {7, {255., 152., 150.}}, {8, {214., 39., 40.}}, {9, {197., 176., 213.}}, {10, {148., 103., 189.}}, {11, {196., 156., 148.}},
    {12, {23., 190., 207.}}, {14, {247., 182., 210.}}, {15, {66., 188., 102.}}, {16, {219., 219., 141.}}, {17, {140., 57., 197.}}, 
    {18, {202., 185., 52.}}, {19, {51., 176., 203.}}, {20, {200., 54., 131.}}, {21, {92., 193., 61.}}, {22, {78., 71., 183.}}, {23, {172., 114., 82.}},
    {24, {255., 127., 14.}}, {25, {91., 163., 138.}}, {26, {153., 98., 156.}}, {27, {140., 153., 101.}}, {28, {158., 218., 229.}}, 
    {29, {100., 125., 154.}}, {30, {178., 127., 135.}}, {32, {146., 111., 194.}}, {33, {44., 160., 44.}}, {34, {112., 128., 144.}},
    {35, {96., 207., 209.}}, {36, {227., 119., 194.}}, {37, {213., 92., 176.}}, {38, {94., 106., 211.}}, {39, {82., 84., 163.}}, {40, {100., 85., 144.}}
};


cv::Vec3b id2color(uint16_t id) {
    auto it = SCANNET_COLOR_MAP_20.find(id);
    if (it != SCANNET_COLOR_MAP_20.end()) {
        auto [r, g, b] = it->second;
        return cv::Vec3b(static_cast<float>(r), static_cast<float>(g), static_cast<float>(b));
    }
    return cv::Vec3b(0, 0, 0);
}

void color_pts(pcl::PointCloud<pcl::PointXYZLNormal>::Ptr &in_cloud,
                 pcl::PointCloud<pcl::PointXYZRGBL>::Ptr &color_cloud) {
    
    // CHANGE POINT TYPE
    color_cloud->resize(in_cloud->size());

    // COLOR THE CLOUS
    for (size_t i = 0; i < in_cloud->size(); ++i) {
        pcl::PointXYZRGBL point;
        point.x = in_cloud->points[i].x;
        point.y = in_cloud->points[i].y;
        point.z = in_cloud->points[i].z;
        point.label = in_cloud->points[i].label;
        
        cv::Vec3b color = id2color(point.label);
        // cv::Vec3b color = id2color(in_cloud->points[i].label);
        point.r = color[0]; 
        point.g = color[1];     
        point.b = color[2]; 

        color_cloud->points[i] = point;
    }
}

// https://pcl.readthedocs.io/projects/tutorials/en/latest/statistical_outlier.html
void SOR_filter(const pcl::PointCloud<pcl::PointXYZLNormal>::Ptr& cloud, int& SOR_MeanK, float& SOR_setStddevMulThresh) {
    
    pcl::StatisticalOutlierRemoval<pcl::PointXYZLNormal> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(SOR_MeanK); 
    sor.setStddevMulThresh(SOR_setStddevMulThresh); 
    sor.filter(*cloud);
    
}

// https://pcl.readthedocs.io/projects/tutorials/en/latest/normal_estimation.html
void compute_normals(const pcl::PointCloud<pcl::PointXYZLNormal>::Ptr &in_cloud, float radius, pcl::PointCloud<pcl::Normal>::Ptr& normals){
    
    pcl::NormalEstimation<pcl::PointXYZLNormal, pcl::Normal> ne;
    ne.setInputCloud (in_cloud);
    
    pcl::search::KdTree<pcl::PointXYZLNormal>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZLNormal> ());
    ne.setSearchMethod(tree);

    ne.setRadiusSearch(radius);
    ne.compute(*normals);
}

struct intr {
    float fx_color;
    float fy_color;
    float mx_color;
    float my_color;
    float fx_depth;
    float fy_depth;
    float mx_depth;
    float my_depth;
};

intr read_intr(const std::string& intr_path) {
    intr intr;
    
    std::string color_txt = intr_path + "/intrinsic_color.txt";
    std::string depth_txt = intr_path + "/intrinsic_depth.txt";

    auto read_file = [](const std::string& file_path, float& fx, float& fy, float& mx, float& my) {
        std::ifstream ifs(file_path);
        if (!ifs.is_open()) {
            std::cerr << "no intrinsics @ " << file_path << std::endl;
            return;
        }

        std::string line;
        float mat[4][4];
        for (int i = 0; i < 4; ++i) {
            std::getline(ifs, line);
            std::istringstream ss(line);
            for (int j = 0; j < 4; ++j) {
                ss >> mat[i][j];
            }
        }

        // fx, 0, mx,
        // 0, fy, my,
        // 0, 0, 1;
        fx = mat[0][0];
        fy = mat[1][1];
        mx = mat[0][2];
        my = mat[1][2];
    };

    read_file(color_txt, intr.fx_color, intr.fy_color, intr.mx_color, intr.my_color);
    read_file(depth_txt, intr.fx_depth, intr.fy_depth, intr.mx_depth, intr.my_depth);
    
    return intr;
}
