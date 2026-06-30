#pragma once

#include <iosfwd>
#include <string>
#include <vector>

namespace colite::exe {

int RunSequentialMatcher(const std::vector<std::string>& args,
                         std::ostream& out,
                         std::ostream& err);
int RunExhaustiveMatcher(const std::vector<std::string>& args,
                         std::ostream& out,
                         std::ostream& err);
int RunSpatialMatcher(const std::vector<std::string>& args,
                      std::ostream& out,
                      std::ostream& err);
int RunVocabTreeMatcher(const std::vector<std::string>& args,
                        std::ostream& out,
                        std::ostream& err);

}  // namespace colite::exe
