"""Successor entry with original limits/source checks and rejection diagnostics."""
import argparse
import json
from pathlib import Path
import sys

from infra.gpu03.task_specific_discovery import resource_guard as guard
from infra.gpu03.task_specific_discovery.guard_failure_diagnostic import collect
from infra.gpu03.task_specific_discovery.phase_entry import check_limits, digest
from infra.gpu03.task_specific_discovery.successor_dispatch import ROOT, validate_successor_manifest


def rejection_callback(runtime):
    def record(mapping, snapshot, owned, table, cgroup):
        value = collect(mapping, snapshot, owned, table, owner_uid=guard.UID, exact_cgroup=cgroup)
        guard.journal(Path(runtime) / "gpu_rejections.jsonl", value)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True); p.add_argument("--sha256", required=True)
    args = p.parse_args()
    m = guard.load_manifest(args.manifest, args.sha256)
    validate_successor_manifest(m, str(ROOT))
    rows = Path("/proc/self/cgroup").read_text().splitlines()
    unified = [x.split(":", 2)[2] for x in rows if x.startswith("0::")]
    if len(unified) != 1 or not unified[0].endswith("/" + m["unit"] + ".service"):
        raise ValueError("Not inside exact independent successor service cgroup")
    cgroup = Path("/sys/fs/cgroup") / unified[0].lstrip("/")
    values = {k: (cgroup / k).read_text().strip() for k in
              ("memory.max", "memory.current", "memory.swap.max", "pids.max", "cpu.max")}
    check_limits(values, m["kernel_limits"])
    for filename, binding in m["bound_files"].items():
        f = Path(filename)
        if not f.is_file() or f.is_symlink() or f.stat().st_size != binding["size_bytes"] or digest(f) != binding["sha256"]:
            raise ValueError("Bound file differs: " + filename)
    guard.write_json(Path(m["entry_receipt"]), {
        "status": "source_inputs_and_kernel_limits_verified", "manifest_sha256": args.sha256,
        "cgroup": str(cgroup), "kernel_limits": values, "bound_files": len(m["bound_files"]),
        "diagnostics": "bounded_metadata_only_guard_rejection_unchanged"})
    result = guard.supervise(args.manifest, args.sha256, gpu_rejection_callback=rejection_callback(m["runtime"]))
    return 0 if result["status"] == "workers_exited_zero" else 1


if __name__ == "__main__":
    sys.exit(main())
