#include <iostream>
#include <string>
#include <vector>

#include "colite/exe/logging.h"
#include "colite/exe/recon_sfm.h"

int main(int argc, char** argv) {
  const std::vector<std::string> args =
      colite::exe::InitializeLoggingAndBuildArgs(argc, argv);

  const int exit_code = colite::exe::RunReconSfm(args, std::cout, std::cerr);
  colite::exe::ShutdownLogging();
  return exit_code;
}
