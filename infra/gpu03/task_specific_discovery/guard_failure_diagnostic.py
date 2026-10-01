"""Prospective metadata-only GPU guard diagnostics; never an ownership authority.

The prospective successor entry calls collect only after the unchanged guard
check rejects. Original staged campaigns are unaffected. The guard's optional
callback seam re-raises its exact original exception if collection/persistence
fails. check_devices_recording remains available for independent inert tests.
No command lines, environments, executable links, signals, or model work.
"""
from __future__ import annotations

import os
from pathlib import Path
import time

SCHEMA = "gpu03_unverified_gpu_process_diagnostic_v1"
MAX_PIDS = 32
MAX_STAT_BYTES = 16384
MAX_CGROUP_BYTES = 16384
MAX_SECONDS = 1.0
ROW_FIELDS = ("pid", "uid", "start", "ppid", "pgid", "sid", "state", "rss")


def _read_at(directory_fd, name, limit):
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory_fd)
    try:
        data = os.read(fd, limit + 1)
        if len(data) > limit:
            raise ValueError("proc metadata exceeds bound")
        return data.decode("utf-8", errors="strict")
    finally:
        os.close(fd)


def _stat(directory_fd, expected_pid):
    data = _read_at(directory_fd, "stat", MAX_STAT_BYTES)
    # The parenthesized comm field may contain spaces/parens. Never return it.
    if int(data.split(" ", 1)[0]) != expected_pid:
        raise ValueError("proc PID mismatch")
    fields = data[data.rfind(")") + 2:].split()
    return {"pid": expected_pid, "uid": os.fstat(directory_fd).st_uid,
            "state": fields[0], "ppid": int(fields[1]), "pgid": int(fields[2]),
            "sid": int(fields[3]), "start": int(fields[19])}


def read_pid_metadata(pid, *, proc_root=Path("/proc")):
    """Read only this PID using a pinned directory fd; bracket for PID/UID reuse.

    Absence, permission errors, parse failure, or changed identity remain explicit
    uncertainty. They never make a failed GPU ownership check pass.
    """
    if type(pid) is not int or pid <= 0:
        raise ValueError("invalid diagnostic PID")
    fd = os.open(Path(proc_root) / str(pid), os.O_RDONLY | os.O_DIRECTORY |
                 os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = _stat(fd, pid)
        raw_cgroup = _read_at(fd, "cgroup", MAX_CGROUP_BYTES)
        rows = [line[3:] for line in raw_cgroup.splitlines() if line.startswith("0::")]
        if len(rows) != 1 or not rows[0].startswith("/") or ".." in Path(rows[0]).parts:
            raise ValueError("noncanonical unified cgroup")
        cgroup = rows[0]
        if any(ord(char) < 32 or ord(char) == 127 for char in cgroup):
            raise ValueError("invalid cgroup control character")
        after = _stat(fd, pid)
        stable = (before["pid"], before["uid"], before["start"]) == (
            after["pid"], after["uid"], after["start"])
        return {"before": before, "after": after, "identity_stable": stable,
                "unified_cgroup": cgroup, "cgroup_membership_is_observation_only": True}
    finally:
        os.close(fd)


def collect(mapping, snapshot, owned, table, *, owner_uid, exact_cgroup=None,
            reader=read_pid_metadata, monotonic=time.monotonic):
    """Bounded capture after rejection; no re-query, ownership update or retry."""
    started = monotonic()
    rejected = {}
    for index in mapping:
        for pid in snapshot.get(index, {}).get("pids", []):
            if pid not in owned or owned[pid].get("uid") != owner_uid:
                rejected.setdefault(pid, []).append(index)
    selected = sorted(rejected)[:MAX_PIDS]
    out = {"schema": SCHEMA, "ownership_decision": "rejected_unchanged",
           "diagnostic_only": True, "owner_uid": owner_uid,
           "captured_at_epoch": time.time(), "observations_are_non_atomic": True,
           "rejected_pid_count": len(rejected), "truncated_pids": len(rejected) > MAX_PIDS,
           "selected_gpu_uuids": dict(mapping), "decision_gpu_snapshot": {}, "processes": []}
    for index in mapping:
        device = snapshot.get(index, {})
        out["decision_gpu_snapshot"][index] = {
            key: device[key] for key in ("uuid", "total_mib", "free_mib", "utilization") if key in device}
        out["decision_gpu_snapshot"][index].update(
            pids=device.get("pids", [])[:MAX_PIDS],
            pid_count=len(device.get("pids", [])),
            pids_truncated=len(device.get("pids", [])) > MAX_PIDS)
    for pid in selected:
        entry = {"pid": pid, "gpu_indices": rejected[pid],
                 "earlier_table": {k: table[pid][k] for k in ROW_FIELDS if k in table.get(pid, {})},
                 "was_in_owned_map": pid in owned}
        out["processes"].append(entry)
        if monotonic() - started >= MAX_SECONDS:
            entry["read_status"] = "diagnostic_time_budget_exhausted"
            continue
        try:
            entry["proc"] = reader(pid)
            entry["read_status"] = "observed"
            entry["observed_exact_service_cgroup"] = (
                exact_cgroup is not None and
                entry["proc"]["unified_cgroup"] == exact_cgroup)
        except (FileNotFoundError, ProcessLookupError):
            entry["read_status"] = "exited_or_unobservable"
        except Exception as exc:
            entry["read_status"] = "metadata_error"
            entry["error_type"] = type(exc).__name__
    out["elapsed_seconds"] = max(0.0, monotonic() - started)
    return out


def check_devices_recording(guard, mapping, snapshot, owned, table, record, *,
                            idle=False, exact_cgroup=None, reader=read_pid_metadata):
    """Diagnostics cannot soften, replace, or bypass the guard's decision."""
    try:
        return guard.check_devices(mapping, snapshot, owned, idle=idle)
    except guard.GuardError as original:
        try:
            diagnostic = collect(mapping, snapshot, owned, table, owner_uid=guard.UID,
                                 exact_cgroup=exact_cgroup, reader=reader)
            record(diagnostic)
        except Exception as exc:
            # Attach only an error type for the caller's final result. Preserve
            # the original exception object and original cleanup control flow.
            original.diagnostic_error_type = type(exc).__name__
        raise
