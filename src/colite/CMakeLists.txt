add_library(colite_core
  core/camera.cpp
  core/geometry.cpp
  core/match_graph.cpp
  core/reconstruction.cpp
  core/version.cpp
  features/sift_extractor.cpp
  matching/descriptor_matcher.cpp
  matching/two_view_geometry_batch.cpp
  matching/two_view_geometry_verifier.cpp
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
