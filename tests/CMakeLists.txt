add_executable(colite_core_tests
  core/camera_test.cpp
  core/geometry_test.cpp
  core/match_graph_test.cpp
  core/reconstruction_test.cpp
  features/sift_extractor_test.cpp
  features/sift_reference_consistency_test.cpp
  integration/recon_mvs_exe_test.cpp
  integration/recon_sfm_exe_test.cpp
  integration/sfm_workspace_and_commands_test.cpp
  integration/sfm_dependency_boundary_test.cpp
  matching/descriptor_matcher_test.cpp
  matching/guided_matching_test.cpp
  matching/pair_generator_test.cpp
  matching/two_view_geometry_batch_test.cpp
  matching/two_view_geometry_verifier_test.cpp
  matching/weak_pair_rematcher_test.cpp
  mvs/dense_cloud_test.cpp
  mvs/dense_map_test.cpp
  mvs/image_workspace_test.cpp
  mvs/mvs_pipeline_test.cpp
  mvs/patch_match_test.cpp
  sfm/global_bundle_adjuster_test.cpp
  sfm/image_registrar_test.cpp
  sfm/incremental_initializer_test.cpp
  sfm/incremental_mapper_test.cpp
  sfm/incremental_triangulator_test.cpp
  sfm/local_bundle_adjuster_test.cpp
  sfm/observation_manager_test.cpp
  sfm/observation_filter_test.cpp
)

add_library(vlfeat_original_reference OBJECT
  ${PROJECT_SOURCE_DIR}/references/vlfeat-0.9.21/vl/generic.c
  ${PROJECT_SOURCE_DIR}/references/vlfeat-0.9.21/vl/host.c
  ${PROJECT_SOURCE_DIR}/references/vlfeat-0.9.21/vl/imopv.c
  ${PROJECT_SOURCE_DIR}/references/vlfeat-0.9.21/vl/random.c
  ${PROJECT_SOURCE_DIR}/references/vlfeat-0.9.21/vl/sift.c
)

target_include_directories(vlfeat_original_reference
  PRIVATE
    ${PROJECT_SOURCE_DIR}/references/vlfeat-0.9.21
)

target_compile_definitions(vlfeat_original_reference
  PRIVATE
    VL_DISABLE_THREADS
    VL_DISABLE_SSE2
    VL_DISABLE_OPENMP
    VlSiftFilt=OrigVlSiftFilt
    VlSiftKeypoint=OrigVlSiftKeypoint
    vl_sift_new=orig_vl_sift_new
    vl_sift_delete=orig_vl_sift_delete
    vl_sift_process_first_octave=orig_vl_sift_process_first_octave
    vl_sift_process_next_octave=orig_vl_sift_process_next_octave
    vl_sift_detect=orig_vl_sift_detect
    vl_sift_calc_keypoint_orientations=orig_vl_sift_calc_keypoint_orientations
    vl_sift_calc_keypoint_descriptor=orig_vl_sift_calc_keypoint_descriptor
    vl_sift_calc_raw_descriptor=orig_vl_sift_calc_raw_descriptor
    vl_sift_keypoint_init=orig_vl_sift_keypoint_init
    vl_imconvcol_v=orig_vl_imconvcol_v
    vl_imconvcol_vf=orig_vl_imconvcol_vf
)

target_compile_options(vlfeat_original_reference
  PRIVATE
    $<$<COMPILE_LANG_AND_ID:C,GNU,Clang>:-Wno-unused-function>
)

target_link_libraries(colite_core_tests
  PRIVATE
    colite_core
    colite_command_exe
    colite_sfm_command_exe
    colite_recon_mvs_exe
    colite_recon_sfm_exe
    GTest::gtest_main
    vlfeat_original_reference
)

target_include_directories(colite_core_tests
  PRIVATE
    ${PROJECT_SOURCE_DIR}/src/colite/features/backends
    ${PROJECT_SOURCE_DIR}/references/vlfeat-0.9.21/vl
)

target_compile_definitions(colite_core_tests
  PRIVATE
    COLITE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

include(GoogleTest)
gtest_discover_tests(colite_core_tests)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_test(
  NAME MvsSweepScriptTest
  COMMAND ${Python3_EXECUTABLE}
          ${PROJECT_SOURCE_DIR}/tests/scripts/mvs_sweep_test.py
)
set_tests_properties(
  MvsSweepScriptTest
  PROPERTIES
    ENVIRONMENT "COLITE_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
)
