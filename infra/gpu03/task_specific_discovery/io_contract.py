"""Durable request boundary for NEW, manifest-bound discovery workers.

This module performs no model/scorer/AST work. Keep one context open per worker::

    with RequestJournal(path, run_token=token, input_sha256=config_sha,
                        request_hashes={rid: hash_value(request)}) as journal:
        # Only never_started IDs may be sent to a model or scorer.
        journal.start(rid)
        primitive = external_call()  # caller-owned operation; exactly once
        journal.record_primitive(rid, primitive)  # before ANY optional metadata
        journal.record_final(rid, derive_metadata(journal.primitive(rid)))

On re-entry, pending()['primitive_ready'] may finish metadata from primitive()
without the external call. A started ID with no durable primitive fails closed.
An unclosed start must NEVER be removed/relabelled/replayed by this module.
The tiny return-to-fsync crash window is intrinsically ambiguous; this contract
refuses replay rather than pretending to provide transactional external calls.

All identities are caller-supplied SHA256 bindings. Bind configuration, source,
runtime and input identities into input_sha256. Durable files use exclusive
creation; existing identical payloads are verified and reused, never overwritten.
A complete orphan payload left before its journal append is adopted on re-entry
only after validating the entire store. Truncated files/journals fail closed.
The advisory lock must remain held across the external call and metadata work.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat


class ContractError(ValueError):
    """Malformed, conflicting, or unauthenticated durable evidence."""


class ReplayRefused(ContractError):
    """An external call was already started and may not be called again."""


class AmbiguousRequest(ReplayRefused):
    def __init__(self, request_ids):
        self.request_ids = tuple(sorted(request_ids))
        super().__init__("Started requests lack durable primitives; no replay: "
                         + ", ".join(self.request_ids))


def _require(condition, message):
    if not condition:
        raise ContractError(message)


def canonical(value):
    def native_json(item, depth=0):
        _require(depth <= 128, "JSON nesting exceeds bounded evidence depth")
        if item is None or type(item) in (str, int, float, bool):
            return
        if type(item) is list:
            for element in item:
                native_json(element, depth + 1)
            return
        if type(item) is dict:
            _require(all(type(key) is str for key in item), "JSON object keys must be strings")
            for element in item.values():
                native_json(element, depth + 1)
            return
        raise ContractError("Evidence requires native JSON types without coercion")

    try:
        native_json(value)
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise ContractError("Value is not finite canonical JSON") from exc


def hash_value(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _decode(data):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def bad_constant(_):
        raise ContractError("Nonfinite JSON value")

    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=pairs,
                          parse_constant=bad_constant)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ContractError("Malformed JSON evidence") from exc


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _file_fd(path, flags, mode=0o600):
    fd = os.open(path, flags | os.O_NOFOLLOW, mode)
    info = os.fstat(fd)
    if not (stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
            and info.st_nlink == 1):
        os.close(fd)
        raise ContractError("Unsafe evidence file owner/type/link count: " + str(path))
    return fd


def _write_all(fd, data):
    offset = 0
    while offset < len(data):
        count = os.write(fd, data[offset:])
        _require(count > 0, "Short evidence write")
        offset += count


class RequestJournal:
    """Single-writer locked journal; construct inside a ``with`` statement.

    ``request_hashes`` freezes the complete worker shard, including unused IDs.
    ``max_payload_bytes`` bounds EACH primitive/final envelope, not all records.
    Payloads are loaded one at a time; journals retain compact metadata only.
    Failure during entry never grants permission to run any external request.
    """

    STAGES = ("started", "primitive_returned", "final_recorded")
    MAX_EVENT_BYTES = 4096

    def __init__(self, root, *, run_token, input_sha256, request_hashes,
                 max_payload_bytes=8 * 1024 * 1024):
        self.root = Path(root).absolute()
        _require(isinstance(run_token, str) and
                 re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_token),
                 "Invalid run token")
        _require(_digest(input_sha256), "Invalid input SHA256")
        _require(isinstance(request_hashes, dict) and request_hashes,
                 "Complete nonempty request-hash mapping required")
        _require(all(isinstance(k, str) and 0 < len(k.encode("utf-8")) <= 512
                     and "\x00" not in k and _digest(v)
                     for k, v in request_hashes.items()), "Invalid request binding")
        _require(type(max_payload_bytes) is int and 1 <= max_payload_bytes <= 64 * 1024 * 1024,
                 "Invalid payload byte cap")
        self.manifest = {"schema": 1, "run_token": run_token,
                         "input_sha256": input_sha256,
                         "request_hashes": dict(request_hashes),
                         "max_payload_bytes": max_payload_bytes}
        self.limit = max_payload_bytes
        self._lock = None
        self._failed = False
        self._events = {stage: {} for stage in self.STAGES}

    def __enter__(self):
        _require(self._lock is None, "Journal context already open")
        self._failed = False
        self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
        info = self.root.lstat()
        _require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid()
                 and not info.st_mode & 0o022, "Unsafe journal directory")
        try:
            self._lock = _file_fd(self.root / "writer.lock", os.O_RDWR | os.O_CREAT)
            try:
                fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ContractError("Journal already has an active writer") from exc
            manifest_path = self.root / "manifest.json"
            if not manifest_path.exists():
                _require(set(p.name for p in self.root.iterdir()) == {"writer.lock"},
                         "Incomplete initialization; preserve store for explicit inspection")
                for name in ("primitives", "finals"):
                    (self.root / name).mkdir(mode=0o700)
                for stage in self.STAGES:
                    fd = _file_fd(self.root / (stage + ".jsonl"), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                self._create_file(manifest_path, canonical(self.manifest) + b"\n", 64 * 1024 * 1024)
                _sync_dir(self.root)
            actual = self._read_json_file(manifest_path, 64 * 1024 * 1024)
            _require(actual == self.manifest, "Manifest/request/input binding mismatch")
            self._load_and_reconcile()
            return self
        except BaseException:
            self.close()
            raise

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self._lock is not None:
            os.close(self._lock)
            self._lock = None

    def _open(self):
        _require(self._lock is not None, "Journal requires its live context lock")
        _require(not self._failed, "Prior persistence failure; close and inspect before continuing")
        info = (self.root / "writer.lock").lstat()
        held = os.fstat(self._lock)
        _require((info.st_dev, info.st_ino) == (held.st_dev, held.st_ino),
                 "Journal lock identity changed")

    def _base(self, request_id):
        _require(request_id in self.manifest["request_hashes"], "Unknown request ID")
        return {"schema": 1, "run_token": self.manifest["run_token"],
                "input_sha256": self.manifest["input_sha256"], "request_id": request_id,
                "request_sha256": self.manifest["request_hashes"][request_id]}

    def _path(self, request_id, kind):
        return self.root / ("primitives" if kind == "primitive" else "finals") / (_sha(request_id.encode("utf-8")) + ".json")

    def _create_file(self, path, data, limit):
        _require(len(data) <= limit, "Evidence payload exceeds byte cap")
        try:
            fd = _file_fd(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            try:
                _write_all(fd, data)
                os.fchmod(fd, 0o400)
                os.fsync(fd)
            finally:
                os.close(fd)
            _sync_dir(path.parent)
        except BaseException:
            self._failed = True
            raise

    def _read_bytes(self, path, limit, *, immutable=True):
        fd = _file_fd(path, os.O_RDONLY)
        try:
            before = os.fstat(fd)
            _require(before.st_size <= limit, "Evidence exceeds byte cap")
            _require(not immutable or not before.st_mode & 0o222, "Immutable evidence is writable")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                data = stream.read(limit + 1)
            after = os.fstat(fd)
            _require((before.st_size, before.st_mtime_ns, before.st_ino) ==
                     (after.st_size, after.st_mtime_ns, after.st_ino), "Evidence changed during read")
            _require(len(data) == before.st_size and len(data) <= limit, "Evidence read length mismatch")
            return data
        finally:
            os.close(fd)

    def _read_json_file(self, path, limit):
        data = self._read_bytes(path, limit)
        _require(data.endswith(b"\n"), "Truncated evidence file")
        value = _decode(data)
        _require(canonical(value) + b"\n" == data, "Noncanonical evidence file")
        return value

    def _append_event(self, stage, value):
        data = canonical(value) + b"\n"
        _require(len(data) <= self.MAX_EVENT_BYTES, "Journal event exceeds byte cap")
        try:
            fd = _file_fd(self.root / (stage + ".jsonl"), os.O_WRONLY | os.O_APPEND)
            try:
                _write_all(fd, data)
                os.fsync(fd)
            finally:
                os.close(fd)
        except BaseException:
            self._failed = True
            raise
        self._events[stage][value["request_id"]] = value

    def _read_events(self, stage):
        found = {}
        path = self.root / (stage + ".jsonl")
        fd = _file_fd(path, os.O_RDONLY)
        try:
            with os.fdopen(fd, "rb", closefd=False) as stream:
                while True:
                    line = stream.readline(self.MAX_EVENT_BYTES + 1)
                    if not line:
                        break
                    _require(line.endswith(b"\n") and len(line) <= self.MAX_EVENT_BYTES,
                             "Truncated or oversized journal record")
                    row = _decode(line)
                    _require(isinstance(row, dict), "Journal record must be an object")
                    rid = row.get("request_id")
                    _require(isinstance(rid, str), "Journal request ID missing")
                    expected = {**self._base(rid), "stage": stage}
                    if stage != "started":
                        _require(_digest(row.get("file_sha256")), "Payload digest missing")
                        expected["file_sha256"] = row["file_sha256"]
                    _require(row == expected and canonical(row) + b"\n" == line,
                             "Journal identity/schema mismatch")
                    _require(rid not in found, "Duplicate journal request ID")
                    found[rid] = row
                    _require(len(found) <= len(self.manifest["request_hashes"]), "Journal request bound exceeded")
        finally:
            os.close(fd)
        return found

    def _payload(self, request_id, kind, expected_sha=None):
        path = self._path(request_id, kind)
        data = self._read_bytes(path, self.limit)
        _require(data.endswith(b"\n"), "Truncated payload")
        value = _decode(data)
        _require(isinstance(value, dict) and "payload" in value, "Invalid payload envelope")
        expected = {**self._base(request_id), "kind": kind, "payload": value["payload"]}
        if kind == "final":
            _require(request_id in self._events["primitive_returned"], "Final lacks primitive event")
            expected["primitive_file_sha256"] = self._events["primitive_returned"][request_id]["file_sha256"]
        _require(value == expected and canonical(value) + b"\n" == data, "Payload identity/schema mismatch")
        digest = _sha(data)
        _require(expected_sha is None or digest == expected_sha, "Payload SHA256 mismatch")
        return value["payload"], digest

    def _load_and_reconcile(self):
        allowed = {"writer.lock", "manifest.json", "primitives", "finals"} | {s + ".jsonl" for s in self.STAGES}
        _require({p.name for p in self.root.iterdir()} == allowed, "Unexpected or missing journal store member")
        self._events = {stage: self._read_events(stage) for stage in self.STAGES}
        starts = set(self._events["started"])
        _require(set(self._events["primitive_returned"]) <= starts,
                 "Primitive journal without start")
        _require(set(self._events["final_recorded"]) <= set(self._events["primitive_returned"]),
                 "Final journal without primitive")
        names = {_sha(rid.encode("utf-8")) + ".json": rid for rid in self.manifest["request_hashes"]}
        _require(len(names) == len(self.manifest["request_hashes"]), "Request filename collision")
        present = {}
        for kind, directory in (("primitive", "primitives"), ("final", "finals")):
            folder = self.root / directory
            info = folder.lstat()
            _require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid()
                     and not info.st_mode & 0o022, "Unsafe payload directory")
            paths = list(folder.iterdir())
            _require(all(p.name in names for p in paths), "Unknown payload filename")
            present[kind] = {names[p.name] for p in paths}
        _require(present["primitive"] <= starts and present["final"] <= present["primitive"],
                 "Payload without required predecessor")
        _require(set(self._events["primitive_returned"]) <= present["primitive"] and
                 set(self._events["final_recorded"]) <= present["final"], "Journal references missing payload")
        ambiguous = starts - present["primitive"]
        # Validate every artifact before adding any recoverable orphan event.
        orphans = []
        for rid in sorted(present["primitive"]):
            event = self._events["primitive_returned"].get(rid)
            _, digest = self._payload(rid, "primitive", event["file_sha256"] if event else None)
            if event is None:
                event = {**self._base(rid), "stage": "primitive_returned", "file_sha256": digest}
                self._events["primitive_returned"][rid] = event
                orphans.append(("primitive_returned", event))
        for rid in sorted(present["final"]):
            event = self._events["final_recorded"].get(rid)
            _, digest = self._payload(rid, "final", event["file_sha256"] if event else None)
            if event is None:
                orphans.append(("final_recorded", {**self._base(rid), "stage": "final_recorded", "file_sha256": digest}))
        if ambiguous:
            raise AmbiguousRequest(ambiguous)
        for stage, event in orphans:
            self._append_event(stage, event)

    def pending(self):
        self._open()
        starts = set(self._events["started"])
        primitives = set(self._events["primitive_returned"])
        finals = set(self._events["final_recorded"])
        return {"never_started": sorted(set(self.manifest["request_hashes"]) - starts),
                "primitive_ready": sorted(primitives - finals),
                "final_recorded": sorted(finals),
                "started_without_primitive": sorted(starts - primitives)}

    def start(self, request_id):
        """Commit intent before the external call; repeat starts always refuse."""
        self._open()
        base = self._base(request_id)
        if request_id in self._events["started"]:
            raise ReplayRefused("Request already started; no external replay: " + request_id)
        self._append_event("started", {**base, "stage": "started"})

    def _record(self, request_id, payload, kind):
        self._open()
        envelope = {**self._base(request_id), "kind": kind, "payload": payload}
        _require(request_id in self._events["started"], "Request was not started")
        if kind == "final":
            _require(request_id in self._events["primitive_returned"], "Final requires durable primitive")
            self.primitive(request_id)  # verify it still matches before deriving final evidence
            envelope["primitive_file_sha256"] = self._events["primitive_returned"][request_id]["file_sha256"]
        data = canonical(envelope) + b"\n"
        _require(len(data) <= self.limit, "Evidence payload exceeds byte cap")
        stage = "primitive_returned" if kind == "primitive" else "final_recorded"
        event = {**self._base(request_id), "stage": stage, "file_sha256": _sha(data)}
        path = self._path(request_id, kind)
        if path.exists() or path.is_symlink():
            existing = self._read_bytes(path, self.limit)
            _require(existing == data, "Conflicting immutable payload")
        else:
            _require(request_id not in self._events[stage], "Recorded payload disappeared")
            self._create_file(path, data, self.limit)
        if request_id in self._events[stage]:
            _require(self._events[stage][request_id] == event, "Conflicting journal identity")
            return False
        self._append_event(stage, event)
        return True

    def record_primitive(self, request_id, payload):
        """Persist the raw returned result before classification/optional diagnostics."""
        return self._record(request_id, payload, "primitive")

    def record_final(self, request_id, payload):
        """Persist derived metadata; cannot mutate or replace the primitive."""
        return self._record(request_id, payload, "final")

    def primitive(self, request_id):
        self._open()
        _require(request_id in self._events["primitive_returned"], "No durable primitive for request")
        return self._payload(request_id, "primitive", self._events["primitive_returned"][request_id]["file_sha256"])[0]

    def final(self, request_id):
        self._open()
        _require(request_id in self._events["final_recorded"], "No durable final for request")
        return self._payload(request_id, "final", self._events["final_recorded"][request_id]["file_sha256"])[0]
