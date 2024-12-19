cmake_minimum_required(VERSION 3.5)
project(mth_proj)

find_package(PCL 1.2 REQUIRED)

find_package(OpenCV REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(lprop_kitti prop_kitti.cpp)
add_executable(lprop_scannet prop_scannet.cpp)
add_executable(mIoU_scannet mIoU_scannet.cpp)

find_package(libpointmatcher REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)
include_directories(${CMAKE_CURRENT_BINARY_DIR} ${libpointmatcher_INCLUDE_DIRS} ${Boost_INCLUDE_DIR})


target_link_libraries(lprop_kitti ${PCL_LIBRARIES} ${OpenCV_LIBS} ${libpointmatcher_LIBRARIES} ${Boost_LIBRARIES})
target_link_libraries(lprop_scannet ${PCL_LIBRARIES} ${OpenCV_LIBS} ${libpointmatcher_LIBRARIES} ${Boost_LIBRARIES})
target_link_libraries(mIoU_scannet ${PCL_LIBRARIES})

# https://github.com/luxonis/depthai-core/issues/982#issuecomment-2025736490
# Check if JsonCpp target does not exist and then create it manually
if(NOT TARGET JsonCpp::JsonCpp)
  # Attempt to find JsonCpp library - modify paths as needed
  find_library(JSONCPP_LIBRARY NAMES jsoncpp PATHS /opt/homebrew/Cellar/jsoncpp/1.9.5/lib /usr/local/lib)
  find_path(JSONCPP_INCLUDE_DIR NAMES json/json.h PATHS /opt/homebrew/Cellar/jsoncpp/1.9.5/include /usr/local/include)

  # Create an imported target
  add_library(JsonCpp::JsonCpp SHARED IMPORTED)
  set_target_properties(JsonCpp::JsonCpp PROPERTIES
    IMPORTED_LOCATION "${JSONCPP_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${JSONCPP_INCLUDE_DIR}")
endif()
