#!/usr/bin/env python3

"""Run one external command with a hard timeout and bounded retries."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--backoff", type=float, default=2.0)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.timeout <= 0 or args.attempts <= 0:
        parser.error("a command, positive timeout, and positive attempt count are required")

    for attempt in range(1, args.attempts + 1):
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=not args.stream,
                text=True,
                timeout=args.timeout,
            )
        except subprocess.TimeoutExpired as error:
            if error.stdout:
                sys.stdout.write(error.stdout if isinstance(error.stdout, str) else error.stdout.decode())
            if error.stderr:
                sys.stderr.write(error.stderr if isinstance(error.stderr, str) else error.stderr.decode())
            sys.stderr.write(
                f"Command timed out after {args.timeout:g}s (attempt {attempt}/{args.attempts}): "
                f"{command[0]}\n"
            )
            return_code = 124
        else:
            if completed.stdout:
                sys.stdout.write(completed.stdout)
            if completed.stderr:
                sys.stderr.write(completed.stderr)
            return_code = completed.returncode
            if return_code == 0:
                return
            sys.stderr.write(
                f"Command failed with exit {return_code} (attempt {attempt}/{args.attempts}): "
                f"{command[0]}\n"
            )
        if attempt < args.attempts:
            time.sleep(args.backoff * attempt)
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
