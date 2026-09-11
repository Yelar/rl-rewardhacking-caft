#!/usr/bin/env python3

"""Spawn the teardown watchdog in an independent local process session."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def spawn(command: list[str], log_file: Path) -> int:
    if not command:
        raise ValueError("watchdog command is empty")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        log_file,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o600,
    )
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=descriptor,
            stderr=subprocess.STDOUT,
            close_fds=True,
            start_new_session=True,
        )
    finally:
        os.close(descriptor)
    return process.pid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    print(spawn(command, args.log_file), flush=True)


if __name__ == "__main__":
    main()
