#include <iostream>
#include "colite/exe/command.h"
#include "colite/exe/logging.h"

int main(int argc, char** argv) {
  const std::vector<std::string> args =
      colite::exe::InitializeLoggingAndBuildArgs(argc, argv);
  const int exit_code = colite::exe::RunColite(args, std::cout, std::cerr);
  colite::exe::ShutdownLogging();
  return exit_code;
}
