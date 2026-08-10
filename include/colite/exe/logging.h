#pragma once

#include <string>
#include <vector>

namespace colite::exe {

// Initializes glog and removes Colite-wide logging options from argv before
// command-specific parsing. Supported options may appear before or after the
// subcommand: --log_dir, --log_level, --log_to_stderr and --v.
std::vector<std::string> InitializeLoggingAndBuildArgs(int argc, char** argv);

void ShutdownLogging();

}  // namespace colite::exe
