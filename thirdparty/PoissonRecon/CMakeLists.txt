add_library(colite_poisson_recon STATIC
  PoissonRecon.cpp
  SurfaceTrimmer.cpp
)

target_include_directories(colite_poisson_recon PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(colite_poisson_recon PRIVATE RELEASE)
target_compile_options(colite_poisson_recon PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wno-write-strings>
  $<$<CXX_COMPILER_ID:MSVC>:/bigobj>
)

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(colite_poisson_recon PUBLIC OpenMP::OpenMP_CXX)
endif()
