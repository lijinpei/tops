#include "tops/utils/exec_utils.h"
#include "gtest/gtest.h"

TEST(exec_wait, exec) {
  char stdin_temp_file[] = "/tmp/tops_test_stdin_XXXXXX";
  char stdout_temp_file[] = "/tmp/tops_test_stdout_XXXXXX";
  char stderr_temp_file[] = "/tmp/tops_test_stderr_XXXXXX";
  int proc_stdin_fd = mkstemp(stdin_temp_file);
  int proc_stdout_fd = mkstemp(stdout_temp_file);
  int proc_stderr_fd = mkstemp(stderr_temp_file);
  int exec_res = tops::execute_and_wait("pwd", {}, proc_stdin_fd,
                                        proc_stdout_fd, proc_stderr_fd);
  EXPECT_EQ(exec_res, 0);
  auto stdin_size = lseek(proc_stdin_fd, 0, SEEK_END);
  auto stdout_size = lseek(proc_stdout_fd, 0, SEEK_END);
  auto stderr_size = lseek(proc_stderr_fd, 0, SEEK_END);
  EXPECT_EQ(stdin_size, 0);
  EXPECT_NE(stdout_size, 0);
  EXPECT_EQ(stderr_size, 0);
}
