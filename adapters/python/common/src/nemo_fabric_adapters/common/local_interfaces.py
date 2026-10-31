# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for interfaces that an adapter serves on loopback."""

from __future__ import annotations

import os
import signal
import socket
import tempfile
from pathlib import Path


def port_available(port: int) -> bool:
    """Return whether a loopback TCP port can be bound now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
        try:
            candidate.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def port_listener_in_session(port: int, session_id: int) -> bool:
    """Return whether a process in one session listens on a TCP port.

    Reads Linux procfs, so a health response can be tied to the interface's own
    process rather than to another process that already held the port.
    """
    sockets: set[str] = set()
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            rows = Path(table).read_text(encoding="ascii").splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            fields = row.split()
            # State 0A is LISTEN; the local address ends in the hexadecimal port.
            if fields[3] == "0A" and int(fields[1].rsplit(":", 1)[1], 16) == port:
                sockets.add(f"socket:[{fields[9]}]")
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            if os.getsid(int(process.name)) != session_id:
                continue
            for descriptor in (process / "fd").iterdir():
                if os.readlink(descriptor) in sockets:
                    return True
        except OSError:
            continue
    return False


def signal_process_group(pid: int, signum: signal.Signals) -> None:
    """Signal a child's process group, or only the child where that's denied.

    OpenShell's sandbox denies signals to a process group with EPERM.
    """
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:
        pass
    except PermissionError:
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            pass


def write_private_file(path: Path, payload: str) -> None:
    """Atomically write an owner-only file in an owner-only directory."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
    )
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(payload)
        os.replace(temporary_name, path)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
