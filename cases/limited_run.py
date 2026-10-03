"""Run a serial/threaded workflow within at most 32 Linux CPU affinity slots.

Usage: python cases/limited_run.py --cores 8 --threads 1 -- python cases/run_sram.py ...
The command inherits the affinity and numerical-library thread limits. This is
not a cluster scheduler: MPI launchers must also be configured to honor the
allocation, and unrelated jobs are outside this process tree's accounting.
"""

import argparse
import json
import os
import sys


THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def allocation(available, cores, threads):
    if not 1 <= cores <= 32:
        raise ValueError("cores must be between 1 and 32")
    cpus = sorted(available)[:cores]
    if len(cpus) != cores:
        raise ValueError(f"Requested {cores} CPUs, but only {len(cpus)} are available")
    if not 1 <= threads <= cores:
        raise ValueError("threads must be between 1 and cores")
    return cpus, {name: str(threads) for name in THREAD_VARIABLES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", type=int, default=32)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("A command is required after --")
    try:
        cpus, limits = allocation(os.sched_getaffinity(0), args.cores, args.threads)
    except ValueError as error:
        parser.error(str(error))
    os.sched_setaffinity(0, cpus)
    environment = dict(os.environ, **limits, OMP_DYNAMIC="FALSE", OMP_MAX_ACTIVE_LEVELS="1")
    print(json.dumps({"cpu_affinity": cpus, "threads_per_library": args.threads}),
          file=sys.stderr, flush=True)
    os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()
