#pragma once

#include <cstddef>
#include <filesystem>

namespace colite::controllers {

struct MapperControllerOptions {
  std::filesystem::path workspace_path;
};

struct MapperControllerSummary {
  bool success = false;
  bool cache_hit = false;
  std::size_t num_registered_images = 0;
  std::size_t num_points3d = 0;
};

class MapperController {
 public:
  explicit MapperController(MapperControllerOptions options);

  MapperControllerSummary Run() const;

 private:
  MapperControllerOptions options_;
};

}  // namespace colite::controllers
