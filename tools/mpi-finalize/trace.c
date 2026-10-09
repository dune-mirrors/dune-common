// SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

#define _POSIX_C_SOURCE 200809L
#include <mpi.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

/* No C++ destructors, MPI calls after finalize, barriers, or sleeps. */
static int trace_fd = -1;
static int world_rank = -1;
static int world_size = -1;

static void record(const char *event, int rc)
{
  char line[256];
  struct timespec now = {0, 0};
  clock_gettime(CLOCK_MONOTONIC, &now);
  int length = snprintf(line, sizeof(line),
                        "event=%s rank=%d size=%d pid=%ld monotonic=%ld.%09ld rc=%d\n",
                        event, world_rank, world_size, (long)getpid(),
                        (long)now.tv_sec, now.tv_nsec, rc);
  if (trace_fd < 0 || length < 0 || (size_t)length >= sizeof(line))
    return;
  const char *cursor = line;
  while (length > 0) {
    ssize_t written = write(trace_fd, cursor, (size_t)length);
    if (written < 0 && errno == EINTR)
      continue;
    if (written <= 0) {
      perror("MPI finalize trace: write");
      return;
    }
    cursor += written;
    length -= (int)written;
  }
}

/* Termination signals are recorded, then re-delivered with the previous action.
 * This shows whether a rank still inside MPI_Finalize was killed by the launcher.
 * snprintf is not formally async-signal-safe; acceptable for this diagnostic. */
static const int traced_signals[] = {SIGTERM, SIGINT, SIGHUP, SIGQUIT, SIGPIPE};
#define TRACED_SIGNAL_COUNT (sizeof(traced_signals) / sizeof(traced_signals[0]))
static struct sigaction previous_actions[TRACED_SIGNAL_COUNT];

static void trace_signal(int sig)
{
  record("signal", sig);
  for (size_t i = 0; i < TRACED_SIGNAL_COUNT; ++i) {
    if (traced_signals[i] == sig) {
      sigaction(sig, &previous_actions[i], NULL);
      break;
    }
  }
  raise(sig);
}

static void install_signal_handlers(void)
{
  struct sigaction action;
  memset(&action, 0, sizeof(action));
  action.sa_handler = trace_signal;
  sigemptyset(&action.sa_mask);
  action.sa_flags = SA_RESETHAND;
  for (size_t i = 0; i < TRACED_SIGNAL_COUNT; ++i)
    if (sigaction(traced_signals[i], &action, &previous_actions[i]) != 0)
      perror("MPI finalize trace: sigaction");
}

static void trace_atexit(void)
{
  record("atexit", 0);
  if (trace_fd >= 0)
    close(trace_fd);
  trace_fd = -1;
}

static void initialized(int rc)
{
  const char *directory = getenv("DUNE_MPI_FINALIZE_LOG_DIR");
  if (rc != MPI_SUCCESS || directory == NULL)
    return;
  PMPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &world_size);
  char path[4096];
  int length = snprintf(path, sizeof(path), "%s/rank-%d-pid-%ld.log",
                        directory, world_rank, (long)getpid());
  if (length < 0 || (size_t)length >= sizeof(path)) {
    fputs("MPI finalize trace: log path too long\n", stderr);
    return;
  }
  trace_fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (trace_fd < 0) {
    perror("MPI finalize trace: open");
    return;
  }
  if (atexit(trace_atexit) != 0)
    fputs("MPI finalize trace: atexit registration failed\n", stderr);
  record("init_return", rc);
  /* After PMPI_Init, so that handlers installed by MPI itself are chained. */
  install_signal_handlers();
}

int MPI_Init(int *argc, char ***argv)
{
  int rc = PMPI_Init(argc, argv);
  initialized(rc);
  return rc;
}

int MPI_Init_thread(int *argc, char ***argv, int required, int *provided)
{
  int rc = PMPI_Init_thread(argc, argv, required, provided);
  initialized(rc);
  return rc;
}

int MPI_Finalize(void)
{
  record("finalize_enter", 0);
  int rc = PMPI_Finalize();
  record("finalize_return", rc);
  return rc;
}

int MPI_Abort(MPI_Comm comm, int errorcode)
{
  record("abort_enter", errorcode);
  return PMPI_Abort(comm, errorcode);
}
