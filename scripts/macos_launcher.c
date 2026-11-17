#include <errno.h>
#include <libgen.h>
#include <limits.h>
#include <netdb.h>
#include <spawn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

static FILE *log_file = NULL;

static void log_line(const char *fmt, ...) {
  if (!log_file) {
    return;
  }

  va_list args;
  va_start(args, fmt);
  vfprintf(log_file, fmt, args);
  va_end(args);
  fputc('\n', log_file);
  fflush(log_file);
}

static int port_open(const char *port) {
  struct addrinfo hints;
  struct addrinfo *result = NULL;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;

  if (getaddrinfo("127.0.0.1", port, &hints, &result) != 0) {
    return 0;
  }

  int sock = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
  if (sock < 0) {
    freeaddrinfo(result);
    return 0;
  }

  int ok = connect(sock, result->ai_addr, result->ai_addrlen) == 0;
  close(sock);
  freeaddrinfo(result);
  return ok;
}

static int run_script(const char *root) {
  char script[PATH_MAX];
  snprintf(script, sizeof(script), "%s/scripts/start-snipra.sh", root);

  char *argv[] = {"/bin/bash", script, NULL};
  pid_t pid;
  int spawn_status = posix_spawn(&pid, "/bin/bash", NULL, NULL, argv, environ);
  if (spawn_status != 0) {
    log_line("posix_spawn failed: %s", strerror(spawn_status));
    return 127;
  }

  int status = 0;
  if (waitpid(pid, &status, 0) < 0) {
    log_line("waitpid failed: %s", strerror(errno));
    return 127;
  }

  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }
  if (WIFSIGNALED(status)) {
    return 128 + WTERMSIG(status);
  }
  return 127;
}

int main(int argc, char **argv) {
  char root[PATH_MAX] = ".";
  if (argc > 0) {
    char executable[PATH_MAX];
    if (realpath(argv[0], executable) != NULL) {
      char *macos_dir = dirname(executable);
      snprintf(root, sizeof(root), "%s/../../..", macos_dir);
    }
  }

  char logs_dir[PATH_MAX];
  snprintf(logs_dir, sizeof(logs_dir), "%s/storage/logs", root);
  mkdir(logs_dir, 0755);

  char log_path[PATH_MAX];
  snprintf(log_path, sizeof(log_path), "%s/macos-app-launcher.log", logs_dir);
  log_file = fopen(log_path, "a");

  log_line("");
  log_line("== Snipra.app native launch ==");
  log_line("Root: %s", root);

  setenv(
      "PATH",
      "/opt/homebrew/opt/ffmpeg-full/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
      1);

  int status = run_script(root);
  log_line("Launcher exited with status %d", status);

  if (status != 0 && !port_open("5173")) {
    char command[PATH_MAX + 256];
    snprintf(command, sizeof(command),
             "osascript -e 'display alert \"Snipra no pudo abrir\" message "
             "\"Revisa storage/logs/macos-app-launcher.log dentro de la carpeta Snipra.\" "
             "as critical' >/dev/null 2>&1");
    system(command);
    return status;
  }

  return 0;
}
