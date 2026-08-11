cmake_minimum_required(VERSION 3.20)

project(
  colite
  VERSION 0.1.0
  DESCRIPTION "A compact C++ SfM and MVS reconstruction system"
  LANGUAGES C CXX
)

include(GNUInstallDirs)

option(COLITE_BUILD_TESTS "Build unit tests" ON)
option(COLITE_ENABLE_CUDA_SIFT "Build the optimized CUDA SIFT feature backend when CUDA is available" ON)
option(COLITE_ENABLE_CUDA_MATCHER "Build the CUDA brute-force descriptor matcher" ON)
option(COLITE_ENABLE_CUDA_MVS "Build the optional CUDA MVS PatchMatch backend" OFF)
option(COLITE_ENABLE_SIFT_PARALLEL "Enable the persistent SIFT worker pool" ON)
option(COLITE_ENABLE_AVX2 "Enable AVX2/FMA optimized CPU SIFT kernels" ON)
option(COLITE_ENABLE_POISSON_MESHING "Build the bundled PoissonRecon mesher" ON)
option(COLITE_ENABLE_DELAUNAY_MESHING "Build the CGAL Delaunay mesher when CGAL is available" ON)
option(COLITE_ENABLE_ONNX "Build optional ONNX Runtime learned feature backends" OFF)

if(COLITE_ENABLE_CUDA_SIFT OR COLITE_ENABLE_CUDA_MATCHER OR COLITE_ENABLE_CUDA_MVS)
  # WSL non-login shells commonly omit /usr/local/cuda/bin from PATH even
  # when the toolkit is installed there.
  if(UNIX AND NOT CMAKE_CUDA_COMPILER AND
     EXISTS "/usr/local/cuda/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH
        "CUDA compiler" FORCE)
  endif()
  include(CheckLanguage)
  check_language(CUDA)
endif()

if((COLITE_ENABLE_CUDA_SIFT OR COLITE_ENABLE_CUDA_MATCHER OR COLITE_ENABLE_CUDA_MVS) AND
   CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_package(CUDAToolkit REQUIRED)
elseif(COLITE_ENABLE_CUDA_SIFT OR COLITE_ENABLE_CUDA_MATCHER OR COLITE_ENABLE_CUDA_MVS)
  message(STATUS "CUDA compiler not found: CUDA SIFT/matcher/MVS backends are disabled")
  set(COLITE_ENABLE_CUDA_SIFT OFF)
  set(COLITE_ENABLE_CUDA_MATCHER OFF)
  set(COLITE_ENABLE_CUDA_MVS OFF)
endif()

install(
  FILES
    LICENSE
    THIRD_PARTY_NOTICES.md
  DESTINATION ${CMAKE_INSTALL_DATADIR}/colite
)

install(
  DIRECTORY docs/
  DESTINATION ${CMAKE_INSTALL_DATADIR}/colite/docs
  FILES_MATCHING PATTERN "*.md"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
find_package(Ceres REQUIRED)
find_package(glog REQUIRED)
find_package(PoseLib QUIET)
find_library(FLANN_CPP_LIBRARY NAMES flann_cpp REQUIRED)
find_library(LZ4_LIBRARY NAMES lz4 REQUIRED)

if(COLITE_ENABLE_ONNX)
  find_package(onnxruntime CONFIG REQUIRED)
endif()

if(COLITE_ENABLE_DELAUNAY_MESHING)
  find_package(CGAL QUIET)
  if(NOT CGAL_FOUND)
    message(STATUS "CGAL not found: Delaunay meshing will be unavailable")
  endif()
endif()

if(COLITE_ENABLE_POISSON_MESHING)
  add_subdirectory(thirdparty/PoissonRecon)
endif()

add_subdirectory(src/colite)
add_subdirectory(src/colite/exe)

if(COLITE_BUILD_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)
  add_subdirectory(tests)
endif()
