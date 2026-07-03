add_library(colite_sfm_command_exe
  feature_extractor.cpp
  sequential_matcher.cpp
  mapper.cpp
  compare_colmap.cpp
)

target_link_libraries(colite_sfm_command_exe
  PUBLIC
    colite_core
)

add_library(colite_recon_sfm_exe
  recon_sfm.cpp
)

target_link_libraries(colite_recon_sfm_exe
  PUBLIC
    colite_core
)

add_library(colite_recon_mvs_exe
  recon_mvs.cpp
)

target_link_libraries(colite_recon_mvs_exe
  PUBLIC
    colite_core
)

add_library(colite_command_exe
  command.cpp
)

target_link_libraries(colite_command_exe
  PUBLIC
    colite_sfm_command_exe
    colite_recon_sfm_exe
    colite_recon_mvs_exe
)

add_executable(colite colite_main.cpp)
target_link_libraries(colite PRIVATE colite_command_exe)

add_executable(feature_extractor feature_extractor_main.cpp)
target_link_libraries(feature_extractor PRIVATE colite_sfm_command_exe)

add_executable(exhaustive_matcher exhaustive_matcher_main.cpp)
target_link_libraries(exhaustive_matcher PRIVATE colite_sfm_command_exe)

add_executable(sequential_matcher sequential_matcher_main.cpp)
target_link_libraries(sequential_matcher PRIVATE colite_sfm_command_exe)

add_executable(spatial_matcher spatial_matcher_main.cpp)
target_link_libraries(spatial_matcher PRIVATE colite_sfm_command_exe)

add_executable(vocab_tree_matcher vocab_tree_matcher_main.cpp)
target_link_libraries(vocab_tree_matcher PRIVATE colite_sfm_command_exe)

add_executable(mapper mapper_main.cpp)
target_link_libraries(mapper PRIVATE colite_sfm_command_exe)

add_executable(compare_colmap compare_colmap_main.cpp)
target_link_libraries(compare_colmap PRIVATE colite_sfm_command_exe)

add_executable(recon_sfm recon_sfm_main.cpp)
target_link_libraries(recon_sfm PRIVATE colite_recon_sfm_exe)

add_executable(recon_mvs recon_mvs_main.cpp)
target_link_libraries(recon_mvs PRIVATE colite_recon_mvs_exe)

if(MSVC)
  target_compile_options(colite PRIVATE /W4)
  target_compile_options(feature_extractor PRIVATE /W4)
  target_compile_options(exhaustive_matcher PRIVATE /W4)
  target_compile_options(sequential_matcher PRIVATE /W4)
  target_compile_options(spatial_matcher PRIVATE /W4)
  target_compile_options(vocab_tree_matcher PRIVATE /W4)
  target_compile_options(mapper PRIVATE /W4)
  target_compile_options(compare_colmap PRIVATE /W4)
  target_compile_options(recon_sfm PRIVATE /W4)
  target_compile_options(recon_mvs PRIVATE /W4)
else()
  target_compile_options(colite PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(feature_extractor PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(exhaustive_matcher PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(sequential_matcher PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(spatial_matcher PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(vocab_tree_matcher PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(mapper PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(compare_colmap PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(recon_sfm PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(recon_mvs PRIVATE -Wall -Wextra -Wpedantic)
endif()
