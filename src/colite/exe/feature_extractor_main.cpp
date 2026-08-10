#include <iostream>
#include <string>
#include <vector>

#include "colite/exe/feature_extractor.h"
#include "colite/exe/logging.h"

int main(int argc, char** argv) {
  const std::vector<std::string> args =
      colite::exe::InitializeLoggingAndBuildArgs(argc, argv);
  const int exit_code =
      colite::exe::RunFeatureExtractor(args, std::cout, std::cerr);
  colite::exe::ShutdownLogging();
  return exit_code;
}
