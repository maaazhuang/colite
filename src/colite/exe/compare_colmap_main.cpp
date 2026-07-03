#include <iostream>
#include <string>
#include <vector>

#include "colite/exe/compare_colmap.h"

int main(int argc, char** argv) {
  std::vector<std::string> args;
  args.reserve(static_cast<std::size_t>(argc));
  for (int i = 0; i < argc; ++i) {
    args.emplace_back(argv[i]);
  }
  return colite::exe::RunCompareColmap(args, std::cout, std::cerr);
}
