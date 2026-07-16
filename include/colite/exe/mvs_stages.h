#pragma once

#include <iosfwd>
#include <string>
#include <vector>

namespace colite::exe {

int RunImageUndistorter(const std::vector<std::string>& args,
                        std::ostream& out, std::ostream& err);
int RunPatchMatchStereo(const std::vector<std::string>& args,
                        std::ostream& out, std::ostream& err);
int RunStereoFusion(const std::vector<std::string>& args,
                    std::ostream& out, std::ostream& err);

}  // namespace colite::exe
