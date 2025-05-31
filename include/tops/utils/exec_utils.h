#pragma once

#include <optional>
#include <vector>

namespace tops {
int execute_and_wait(const char *file_name,
                     const std::vector<const char *> &argv,
                     std::optional<int> stdin_fd = std::nullopt,
                     std::optional<int> stdout_fd = std::nullopt,
                     std::optional<int> stderr_fd = std::nullopt);
}
