#pragma once

#include <iosfwd>
#include <string>
#include <vector>

namespace colite::exe {

int RunPoissonMesher(const std::vector<std::string>& args,
                     std::ostream& out, std::ostream& err);
int RunDelaunayMesher(const std::vector<std::string>& args,
                      std::ostream& out, std::ostream& err);
int RunMeshSimplifier(const std::vector<std::string>& args,
                      std::ostream& out, std::ostream& err);
int RunMeshTexturer(const std::vector<std::string>& args,
                    std::ostream& out, std::ostream& err);

}  // namespace colite::exe
