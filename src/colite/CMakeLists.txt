set(COLITE_CUDA_SIFT_SOURCES
  features/backends/cuda_sift_extractor_stub.cpp
)
set(COLITE_CUDA_MVS_SOURCES
  mvs/patch_match_cuda_stub.cpp
)

if(COLITE_ENABLE_CUDA_SIFT)
  set(COLITE_CUDA_SIFT_SOURCES
    features/backends/cuda_sift_extractor.cu
  )

  add_library(colite_cudasift_backend STATIC
    features/backends/cudasift/thirdparty/cudaImage.cu
    features/backends/cudasift/thirdparty/cudaSiftH.cu
  )
  target_include_directories(colite_cudasift_backend
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/features/backends/cudasift/thirdparty
  )
  target_compile_options(colite_cudasift_backend
    PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
  )
  set_target_properties(colite_cudasift_backend PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
  )
endif()

if(COLITE_ENABLE_CUDA_MVS)
  set(COLITE_CUDA_MVS_SOURCES
    mvs/patch_match_cuda.cu
  )
endif()

add_library(colite_core
  core/camera.cpp
  core/geometry.cpp
  core/match_graph.cpp
  core/reconstruction.cpp
  core/version.cpp
  controllers/recon_sfm_controller.cpp
  controllers/recon_mvs_controller.cpp
  controllers/colmap_compare_controller.cpp
  controllers/feature_extractor.cpp
  controllers/sequential_matcher.cpp
  controllers/mapper.cpp
  features/backends/sift_opencv.cpp
  features/backends/sift_vlfeat.c
  ${COLITE_CUDA_SIFT_SOURCES}
  features/sift_extractor.cpp
  io/camera_prior.cpp
  io/sfm_workspace.cpp
  matching/descriptor_matcher.cpp
  matching/guided_matching.cpp
  matching/pair_generator.cpp
  matching/two_view_geometry_batch.cpp
  matching/two_view_geometry_verifier.cpp
  matching/weak_pair_rematcher.cpp
  mvs/dense_cloud.cpp
  mvs/dense_map.cpp
  mvs/image_workspace.cpp
  mvs/mvs_pipeline.cpp
  mvs/patch_match.cpp
  ${COLITE_CUDA_MVS_SOURCES}
  sfm/global_bundle_adjuster.cpp
  sfm/image_registrar.cpp
  sfm/incremental_initializer.cpp
  sfm/incremental_mapper.cpp
  sfm/incremental_triangulator.cpp
  sfm/local_bundle_adjuster.cpp
  sfm/observation_manager.cpp
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

if(COLITE_ENABLE_CUDA_SIFT)
  target_link_libraries(colite_core PUBLIC colite_cudasift_backend)
  target_compile_definitions(colite_core PUBLIC COLITE_ENABLE_CUDA_SIFT)
endif()

if(COLITE_ENABLE_CUDA_MVS)
  target_compile_definitions(colite_core PUBLIC COLITE_ENABLE_CUDA_MVS)
  target_compile_options(colite_core
    PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;--expt-relaxed-constexpr;-Xcudafe=--diag_suppress=20012>
  )
endif()

if(MSVC)
  target_compile_options(colite_core PRIVATE /W4)
else()
  target_compile_options(colite_core
    PRIVATE
      $<$<COMPILE_LANGUAGE:C>:-Wall;-Wextra;-Wpedantic>
      $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  )
endif()

set_source_files_properties(
  features/backends/sift_opencv.cpp
  features/backends/sift_vlfeat.c
  features/backends/cuda_sift_extractor_stub.cpp
  mvs/patch_match_cuda_stub.cpp
  PROPERTIES
    COMPILE_OPTIONS "-Wno-pedantic;-Wno-unused-function"
)
