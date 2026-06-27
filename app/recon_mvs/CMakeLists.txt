add_library(recon_mvs_app recon_mvs_app.cpp)

target_include_directories(recon_mvs_app
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(recon_mvs_app
  PUBLIC
    colite_core
)

add_executable(recon_mvs main.cpp)

target_link_libraries(recon_mvs
  PRIVATE
    recon_mvs_app
)

if(MSVC)
  target_compile_options(recon_mvs PRIVATE /W4)
else()
  target_compile_options(recon_mvs PRIVATE -Wall -Wextra -Wpedantic)
endif()
