#include "colite/exe/logging.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <stdexcept>
#include <string>

#include <glog/logging.h>

namespace colite::exe {
namespace {

bool ParseBool(const std::string& value, const std::string& option) {
  std::string normalized = value;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](const unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  if (normalized == "1" || normalized == "true" || normalized == "yes") {
    return true;
  }
  if (normalized == "0" || normalized == "false" || normalized == "no") {
    return false;
  }
  throw std::invalid_argument(option + " expects 0/1 or true/false");
}

int ParseLogLevel(const std::string& value) {
  std::string normalized = value;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](const unsigned char c) {
                   return static_cast<char>(std::toupper(c));
                 });
  if (normalized == "INFO" || normalized == "0") return 0;
  if (normalized == "WARNING" || normalized == "1") return 1;
  if (normalized == "ERROR" || normalized == "2") return 2;
  if (normalized == "FATAL" || normalized == "3") return 3;
  throw std::invalid_argument(
      "--log_level expects INFO, WARNING, ERROR or FATAL");
}

}  // namespace

std::vector<std::string> InitializeLoggingAndBuildArgs(const int argc,
                                                       char** argv) {
  if (argc <= 0 || argv == nullptr || argv[0] == nullptr) {
    throw std::invalid_argument("Invalid process arguments");
  }

  // Colite historically relied on redirected stdout/stderr for durable test
  // logs. Keep INFO visible by default and only switch to glog files when the
  // caller explicitly supplies --log_dir without --log_to_stderr=1.
  bool log_to_stderr = true;
  bool explicit_log_to_stderr = false;
  std::string log_dir;
  int min_log_level = 0;
  int verbosity = 0;

  std::vector<std::string> args;
  args.reserve(static_cast<std::size_t>(argc));
  args.emplace_back(argv[0]);
  for (int idx = 1; idx < argc; ++idx) {
    const std::string option = argv[idx];
    auto require_value = [&]() -> std::string {
      if (++idx >= argc) {
        throw std::invalid_argument("Missing value for " + option);
      }
      return argv[idx];
    };
    if (option == "--log_dir") {
      log_dir = require_value();
    } else if (option == "--log_level") {
      min_log_level = ParseLogLevel(require_value());
    } else if (option == "--log_to_stderr") {
      log_to_stderr = ParseBool(require_value(), option);
      explicit_log_to_stderr = true;
    } else if (option == "--v") {
      verbosity = std::stoi(require_value());
      if (verbosity < 0) {
        throw std::invalid_argument("--v must be non-negative");
      }
    } else {
      args.push_back(option);
    }
  }

  if (!log_dir.empty()) {
    std::filesystem::create_directories(log_dir);
    FLAGS_log_dir = log_dir;
    if (!explicit_log_to_stderr) log_to_stderr = false;
  }
  FLAGS_logtostderr = log_to_stderr;
  FLAGS_alsologtostderr = !log_to_stderr && explicit_log_to_stderr;
  FLAGS_minloglevel = min_log_level;
  FLAGS_v = verbosity;
  google::InitGoogleLogging(argv[0]);
  return args;
}

void ShutdownLogging() { google::ShutdownGoogleLogging(); }

}  // namespace colite::exe
