#pragma once

#include <iosfwd>
#include <string>
#include <vector>

namespace colite::exe {

int RunModelAnalyzer(const std::vector<std::string>&, std::ostream&, std::ostream&);
int RunModelTransformer(const std::vector<std::string>&, std::ostream&, std::ostream&);
int RunModelAligner(const std::vector<std::string>&, std::ostream&, std::ostream&);
int RunModelCropper(const std::vector<std::string>&, std::ostream&, std::ostream&);
int RunBundleAdjuster(const std::vector<std::string>&, std::ostream&, std::ostream&);
int RunPointTriangulator(const std::vector<std::string>&, std::ostream&, std::ostream&);
int RunImageRegistrator(const std::vector<std::string>&, std::ostream&, std::ostream&);
int RunModelSplitter(const std::vector<std::string>&, std::ostream&, std::ostream&);

}  // namespace colite::exe
