#include <iostream>
#include <string>
#include <vector>

#include <glog/logging.h>

#include "colite/exe/recon_sfm.h"

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  std::vector<std::string> args;
  args.reserve(static_cast<std::size_t>(argc));
  for (int i = 0; i < argc; ++i) {
    args.emplace_back(argv[i]);
  }

  const int exit_code = colite::exe::RunReconSfm(args, std::cout, std::cerr);
  google::ShutdownGoogleLogging();
  return exit_code;
}
