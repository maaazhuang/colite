#include "colite/core/version.h"

namespace colite {

std::string_view Version() {
  return COLITE_VERSION_STRING;
}

}  // namespace colite

