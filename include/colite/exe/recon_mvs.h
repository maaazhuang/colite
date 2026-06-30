#pragma once

#include <iosfwd>
#include <string>
#include <vector>

#include "colite/controllers/recon_mvs_controller.h"

namespace colite::exe {

struct ReconMvsCliOptions : controllers::ReconMvsControllerOptions {
  bool show_help = false;
  bool show_version = false;
};

int RunReconMvs(const std::vector<std::string>& args,
                std::ostream& out,
                std::ostream& err);

}  // namespace colite::exe
