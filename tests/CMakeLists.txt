add_executable(colite_core_tests
  core/camera_test.cpp
  core/geometry_test.cpp
  core/match_graph_test.cpp
  core/reconstruction_test.cpp
  features/sift_extractor_test.cpp
  integration/recon_sfm_app_test.cpp
  matching/descriptor_matcher_test.cpp
  matching/two_view_geometry_batch_test.cpp
  matching/two_view_geometry_verifier_test.cpp
  sfm/global_bundle_adjuster_test.cpp
  sfm/image_registrar_test.cpp
  sfm/incremental_initializer_test.cpp
  sfm/incremental_mapper_test.cpp
  sfm/incremental_triangulator_test.cpp
  sfm/local_bundle_adjuster_test.cpp
  sfm/observation_filter_test.cpp
)

target_link_libraries(colite_core_tests
  PRIVATE
    colite_core
    recon_sfm_app
    GTest::gtest_main
)

target_compile_definitions(colite_core_tests
  PRIVATE
    COLITE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

include(GoogleTest)
gtest_discover_tests(colite_core_tests)
