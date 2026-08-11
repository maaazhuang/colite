set(COLITE_CUDA_SIFT_SOURCES
  features/backends/cuda_sift_extractor_stub.cpp
)
set(COLITE_CUDA_MVS_SOURCES
  mvs/patch_match_cuda_stub.cpp
)
set(COLITE_CUDA_MATCHER_SOURCES
  matching/descriptor_matcher_cuda_stub.cpp
)

if(COLITE_ENABLE_CUDA_SIFT)
  set(COLITE_CUDA_SIFT_SOURCES
    features/backends/cuda_sift_extractor.cu
    features/backends/cuda_sift/cuda_sift.cu
    features/backends/cuda_sift/cuda_sift_initial.cu
    features/backends/cuda_sift/cuda_sift_pyramid.cu
    features/backends/cuda_sift/cuda_sift_extrema.cu
    features/backends/cuda_sift/cuda_sift_limit.cu
    features/backends/cuda_sift/cuda_sift_orientation.cu
    features/backends/cuda_sift/cuda_sift_descriptor.cu
  )
endif()

if(COLITE_ENABLE_CUDA_MVS)
  set(COLITE_CUDA_MVS_SOURCES
    mvs/patch_match_cuda.cu
  )
endif()

if(COLITE_ENABLE_CUDA_MATCHER)
  set(COLITE_CUDA_MATCHER_SOURCES
    matching/descriptor_matcher_cuda.cu
  )
endif()

add_library(colite_core
  core/camera.cpp
  core/geometry.cpp
  core/match_graph.cpp
  core/reconstruction.cpp
  core/parallel.cpp
  core/version.cpp
  exe/logging.cpp
  controllers/recon_sfm_controller.cpp
  controllers/sfm_camera_setup.cpp
  controllers/recon_mvs_controller.cpp
  controllers/colmap_compare_controller.cpp
  controllers/feature_extractor.cpp
  controllers/sequential_matcher.cpp
  controllers/mapper.cpp
  features/backends/sift_opencv.cpp
  features/backends/sift_vlfeat.c
  ${COLITE_CUDA_SIFT_SOURCES}
  features/aliked_extractor.cpp
  features/onnx_runtime.cpp
  features/sift_extractor.cpp
  io/camera_prior.cpp
  io/image_reader.cpp
  io/model_resource.cpp
  io/sfm_workspace.cpp
  matching/descriptor_matcher.cpp
  matching/learned_matcher.cpp
  ${COLITE_CUDA_MATCHER_SOURCES}
  matching/guided_matching.cpp
  matching/match_graph_consistency.cpp
  matching/pair_generator.cpp
  matching/vocabulary_tree.cpp
  matching/two_view_geometry_batch.cpp
  matching/two_view_geometry_verifier.cpp
  matching/weak_pair_rematcher.cpp
  mvs/dense_cloud.cpp
  mvs/dense_map.cpp
  mvs/image_workspace.cpp
  mvs/image_undistorter.cpp
  mvs/mvs_pipeline.cpp
  mvs/meshing.cpp
  mvs/mesh_simplifier.cpp
  mvs/mesh_texturer.cpp
  mvs/patch_match.cpp
  ${COLITE_CUDA_MVS_SOURCES}
  sfm/global_bundle_adjuster.cpp
  sfm/global_mapper.cpp
  sfm/global_positioner.cpp
  sfm/hierarchical_mapper.cpp
  sfm/image_registrar.cpp
  sfm/incremental_initializer.cpp
  sfm/incremental_mapper.cpp
  sfm/incremental_triangulator.cpp
  sfm/local_bundle_adjuster.cpp
  sfm/model_clusterer.cpp
  sfm/observation_manager.cpp
  sfm/observation_filter.cpp
  sfm/reconstruction_manager.cpp
  sfm/rotation_averager.cpp
  sfm/similarity_transform.cpp
  sfm/structureless_registrar.cpp
  sfm/view_graph.cpp
)

target_include_directories(colite_core
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/features
    ${CMAKE_CURRENT_SOURCE_DIR}/features/backends
)

target_compile_definitions(colite_core
  PUBLIC
    COLITE_VERSION_STRING="${PROJECT_VERSION}"
)

set_source_files_properties(features/backends/sift_vlfeat.c PROPERTIES
  LANGUAGE CXX
)

if(COLITE_ENABLE_SIFT_PARALLEL)
  target_compile_definitions(colite_core PRIVATE
    COLITE_ENABLE_SIFT_PARALLEL=1
    COLITE_VLFEAT_ENABLE_PARALLEL=1
  )
endif()

if(COLITE_ENABLE_AVX2)
  target_compile_definitions(colite_core PRIVATE
    COLITE_ENABLE_AVX2=1
    COLITE_VLFEAT_ENABLE_AVX2=1
  )
endif()

target_link_libraries(colite_core
  PUBLIC
    Eigen3::Eigen
    Ceres::ceres
    glog::glog
    ${FLANN_CPP_LIBRARY}
    ${LZ4_LIBRARY}
    ${OpenCV_LIBS}
)

if(COLITE_ENABLE_ONNX)
  target_link_libraries(colite_core PUBLIC onnxruntime::onnxruntime)
  target_compile_definitions(colite_core PUBLIC COLITE_ENABLE_ONNX=1)
endif()

if(PoseLib_FOUND)
  target_link_libraries(colite_core PRIVATE PoseLib::PoseLib)
  target_compile_definitions(colite_core PRIVATE COLITE_HAS_POSELIB=1)
endif()

if(COLITE_ENABLE_POISSON_MESHING)
  target_link_libraries(colite_core PRIVATE colite_poisson_recon)
  target_compile_definitions(colite_core PUBLIC COLITE_ENABLE_POISSON_MESHING)
endif()

if(COLITE_ENABLE_DELAUNAY_MESHING AND CGAL_FOUND)
  target_link_libraries(colite_core PRIVATE CGAL::CGAL)
  target_compile_definitions(colite_core PUBLIC COLITE_ENABLE_DELAUNAY_MESHING)
endif()

if(COLITE_ENABLE_CUDA_SIFT OR COLITE_ENABLE_CUDA_MATCHER)
  target_link_libraries(colite_core PUBLIC CUDA::cudart)
endif()

if(COLITE_ENABLE_CUDA_SIFT)
  target_compile_definitions(colite_core PUBLIC COLITE_ENABLE_CUDA_SIFT)
  set_target_properties(colite_core PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endif()

if(COLITE_ENABLE_CUDA_MATCHER)
  target_link_libraries(colite_core PUBLIC CUDA::cublas)
  target_compile_definitions(colite_core PUBLIC COLITE_ENABLE_CUDA_MATCHER)
  set_target_properties(colite_core PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
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
  features/backends/cuda_sift_extractor_stub.cpp
  matching/descriptor_matcher_cuda_stub.cpp
  mvs/patch_match_cuda_stub.cpp
  PROPERTIES
    COMPILE_OPTIONS "-Wno-pedantic;-Wno-unused-function"
)

if(MSVC)
  set(COLITE_OPENCV_SIFT_OPTIONS /utf-8)
  set(COLITE_VLFEAT_SIFT_OPTIONS /utf-8)
  if(COLITE_ENABLE_AVX2)
    list(APPEND COLITE_OPENCV_SIFT_OPTIONS /arch:AVX2)
    list(APPEND COLITE_VLFEAT_SIFT_OPTIONS /arch:AVX2)
  endif()
else()
  set(COLITE_OPENCV_SIFT_OPTIONS
    -Wno-pedantic -Wno-unused-function -fno-tree-vectorize)
  set(COLITE_VLFEAT_SIFT_OPTIONS -Wno-pedantic -Wno-unused-function)
  if(COLITE_ENABLE_AVX2)
    list(APPEND COLITE_OPENCV_SIFT_OPTIONS -mavx2 -mfma)
    list(APPEND COLITE_VLFEAT_SIFT_OPTIONS -mavx2 -mfma)
  endif()
endif()

# AVX2 只作用于不跨库边界传递 Eigen 类型的 SIFT 内核，避免调用方和
# colite_core 采用不同 Eigen 对齐 ABI（会导致 aligned pointer 被错误 free）。
set_source_files_properties(features/backends/sift_opencv.cpp PROPERTIES
  COMPILE_OPTIONS "${COLITE_OPENCV_SIFT_OPTIONS}")
set_source_files_properties(features/backends/sift_vlfeat.c PROPERTIES
  COMPILE_OPTIONS "${COLITE_VLFEAT_SIFT_OPTIONS}")
