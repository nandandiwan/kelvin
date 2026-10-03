import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from cases.limited_run import allocation, THREAD_VARIABLES


@pytest.mark.parametrize("cores,threads", [(33, 1), (0, 1), (4, 5), (2, 0)])
def test_reject_oversubscription(cores, threads):
    with pytest.raises(ValueError):
        allocation(range(64), cores, threads)


def test_child_inherits_affinity_and_thread_limits():
    launcher = Path(__file__).resolve().parents[1] / "cases/limited_run.py"
    result = subprocess.run(
        [sys.executable, str(launcher), "--cores", "1", "--threads", "1", "--",
         sys.executable, "-c", "import json,os; print(json.dumps([list(os.sched_getaffinity(0)), "
         f"[os.environ[k] for k in {THREAD_VARIABLES!r}]]))"],
        check=True, text=True, capture_output=True,
    )
    cpus, threads = json.loads(result.stdout)
    assert cpus == [min(os.sched_getaffinity(0))]
    assert threads == ["1"] * len(THREAD_VARIABLES)
