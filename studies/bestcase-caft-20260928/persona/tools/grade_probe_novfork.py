"""Run step2/tools/grade_probe.py unchanged, with CPython's vfork-based process launch turned off.

29 Sept: grading the transfer journals failed again and again with "OSError: [Errno 14] Bad address" when launching
bwrap. The same answers graded one at a time never failed (300 of 300). The failures come from the thread pool
launching many sandboxes at once, where vfork can race with other threads. A failure raises and never mislabels an
answer (CODE_EVAL_SANDBOX_REQUIRED=1), so only speed is at stake. Plain fork/exec avoids the race.
Usage: same arguments as grade_probe.py.
"""
import runpy
import subprocess
import sys
from pathlib import Path

if not hasattr(subprocess, '_USE_VFORK'):
    raise SystemExit('this Python has no subprocess._USE_VFORK switch')
subprocess._USE_VFORK = False
TOOL = Path(__file__).resolve().parents[2] / 'step2' / 'tools' / 'grade_probe.py'
sys.argv = [str(TOOL)] + sys.argv[1:]
runpy.run_path(str(TOOL), run_name='__main__')
