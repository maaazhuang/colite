#pragma once

#include <iosfwd>
#include <string>
#include <vector>

namespace colite::exe {

int RunMapper(const std::vector<std::string>& args,
              std::ostream& out,
              std::ostream& err);

}  // namespace colite::exe
