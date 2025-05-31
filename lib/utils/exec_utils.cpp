#include "tops/utils/exec_utils.h"

#include "sys/wait.h"
#include "unistd.h"

namespace tops {
int execute_and_wait(const char *file_name,
                     const std::vector<const char *> &argv,
                     std::optional<int> stdin_fd, std::optional<int> stdout_fd,
                     std::optional<int> stderr_fd) {
  auto pid = fork();
  if (pid == -1) {
    return -1;
  }
  if (pid == 0) {
    if (stdin_fd) {
      if (-1 == dup2(*stdin_fd, STDIN_FILENO)) {
        return -1;
      }
    }
    if (stdout_fd) {
      if (-1 == dup2(*stdout_fd, STDOUT_FILENO)) {
        return -1;
      }
    }
    if (stderr_fd) {
      if (-1 == dup2(*stderr_fd, STDERR_FILENO)) {
        return -1;
      }
    }
    execvp(file_name, (char *const *)argv.data());
    return -1;
  } else {
    int status;
    if (!waitpid(pid, &status, 0)) {
      return -1;
    }
    return status;
  }
}
} // namespace tops
