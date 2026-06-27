add_library(colite_core
  core/camera.cpp
  core/geometry.cpp
  core/match_graph.cpp
  core/reconstruction.cpp
  core/version.cpp
  features/backends/sift_opencv_v2.cpp
  features/backends/sift_vlfeat_v2.c
  features/backends/vl_imconv.c
  features/sift_extractor.cpp
  matching/descriptor_matcher.cpp
  matching/guided_matching.cpp
  matching/two_view_geometry_batch.cpp
  matching/two_view_geometry_verifier.cpp
  matching/weak_pair_rematcher.cpp
  mvs/dense_cloud.cpp
  mvs/dense_map.cpp
  mvs/image_workspace.cpp
  mvs/mvs_pipeline.cpp
  mvs/patch_match.cpp
  sfm/global_bundle_adjuster.cpp
  sfm/image_registrar.cpp
  sfm/incremental_initializer.cpp
  sfm/incremental_mapper.cpp
  sfm/incremental_triangulator.cpp
  sfm/local_bundle_adjuster.cpp
  sfm/observation_filter.cpp
)

target_include_directories(colite_core
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/features
)

target_compile_definitions(colite_core
  PUBLIC
    COLITE_VERSION_STRING="${PROJECT_VERSION}"
)

target_link_libraries(colite_core
  PUBLIC
    Eigen3::Eigen
    Ceres::ceres
    glog::glog
    ${FLANN_CPP_LIBRARY}
    ${LZ4_LIBRARY}
    ${OpenCV_LIBS}
)

if(MSVC)
  target_compile_options(colite_core PRIVATE /W4)
else()
  target_compile_options(colite_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

set_source_files_properties(
  features/backends/sift_opencv_v2.cpp
  features/backends/sift_vlfeat_v2.c
  features/backends/vl_imconv.c
  PROPERTIES
    COMPILE_OPTIONS "-Wno-pedantic;-Wno-unused-function"
)
