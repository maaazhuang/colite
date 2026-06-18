cmake_minimum_required(VERSION 3.20)

project(
  colite
  VERSION 0.1.0
  DESCRIPTION "A compact C++ SfM and MVS reconstruction system"
  LANGUAGES C CXX
)

option(COLITE_BUILD_TESTS "Build unit tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc features2d calib3d)
find_package(Ceres REQUIRED)
find_package(glog REQUIRED)
find_library(FLANN_CPP_LIBRARY NAMES flann_cpp REQUIRED)
find_library(LZ4_LIBRARY NAMES lz4 REQUIRED)

add_subdirectory(src/colite)
add_subdirectory(app/recon_sfm)

if(COLITE_BUILD_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)
  add_subdirectory(tests)
endif()
