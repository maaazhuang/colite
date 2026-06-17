add_library(recon_sfm_app recon_sfm_app.cpp)

target_include_directories(recon_sfm_app
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(recon_sfm_app
  PUBLIC
    colite_core
)

add_executable(recon_sfm main.cpp)

target_link_libraries(recon_sfm
  PRIVATE
    recon_sfm_app
)

if(MSVC)
  target_compile_options(recon_sfm PRIVATE /W4)
else()
  target_compile_options(recon_sfm PRIVATE -Wall -Wextra -Wpedantic)
endif()
