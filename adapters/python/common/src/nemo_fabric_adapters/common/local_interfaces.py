# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for native interfaces that an adapter serves on loopback.

Interfaces are enabled under ``harness.settings.interfaces.<name>``, whose
shared fields the adapter contract's ``interface-settings`` schema defines.
"""

from __future__ import annotations

import os
import signal
import socket
import sys
import tempfile
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Literal

from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.common.lifecycle import LifecycleError

LOOPBACK = "127.0.0.1"
_SETTINGS = "harness.settings.interfaces"


@dataclass(frozen=True)
class InterfaceSpec:
    """One interface an adapter supports under ``harness.settings.interfaces``.

    An ``http`` interface requires an exact ``port`` from 1 to ``max_port``; a
    ``terminal`` interface takes none. ``fields`` names adapter-specific fields,
    which the adapter validates itself.
    """

    kind: Literal["http", "terminal"] = "http"
    fields: frozenset[str] = frozenset()
    max_port: int = 65535


def interface_settings(
    settings: Mapping[str, Any],
    supported: Mapping[str, InterfaceSpec],
    *,
    adapter: str,
    code: str,
) -> dict[str, dict[str, Any]]:
    """Validate ``harness.settings.interfaces`` and return each present interface.

    Each returned interface carries its normalized ``enabled`` flag, which
    defaults to true, so the adapter can still validate a disabled interface's
    own fields. Errors name the setting's path in ``metadata["field"]``.
    """

    def invalid(field: str, message: str) -> LifecycleError:
        return LifecycleError(code, message, metadata={"field": field})

    interfaces = settings.get("interfaces", {})
    if not isinstance(interfaces, dict):
        raise invalid(_SETTINGS, f"{adapter} interfaces must be an object")
    unsupported = sorted(set(interfaces) - set(supported))
    if unsupported:
        raise invalid(
            f"{_SETTINGS}.{unsupported[0]}",
            f"{adapter} doesn't support the {unsupported[0]!r} interface",
        )
    present: dict[str, dict[str, Any]] = {}
    for name, spec in supported.items():
        if name not in interfaces:
            continue
        field = f"{_SETTINGS}.{name}"
        interface = interfaces[name]
        if not isinstance(interface, dict):
            raise invalid(field, f"{adapter} {name} interface must be an object")
        allowed = {"enabled", *spec.fields} | (
            {"port"} if spec.kind == "http" else set()
        )
        unknown = sorted(set(interface) - allowed)
        if unknown:
            raise invalid(
                f"{field}.{unknown[0]}",
                f"{adapter} {name} interface doesn't accept {unknown[0]!r}",
            )
        if spec.kind == "http":
            port = interface.get("port")
            if (
                isinstance(port, bool)
                or not isinstance(port, int)
                or not 1 <= port <= spec.max_port
            ):
                raise invalid(
                    f"{field}.port",
                    f"{adapter} {name} port must be an integer from 1 to {spec.max_port}",
                )
        enabled = interface.get("enabled", True)
        if not isinstance(enabled, bool):
            raise invalid(
                f"{field}.enabled", f"{adapter} {name} enabled must be a boolean"
            )
        present[name] = {**interface, "enabled": enabled}
    return present


def runtime_directory(
    context: RuntimeContext, base_dir: str | Path, adapter: str
) -> Path:
    """Return ``<artifact root>/.fabric/<adapter>/runtimes/<runtime-id>``.

    The artifact root is the runtime context's artifact directory, resolved
    against ``base_dir``, or exactly ``<base_dir>/artifacts`` when it's unset.
    Interface tokens, launchers and markers live in its ``interfaces`` directory.
    """

    root = context.artifacts.root
    base = Path(base_dir).resolve()
    artifact_root = (base / str(root)).resolve() if root else base / "artifacts"
    return artifact_root / ".fabric" / adapter / "runtimes" / context.runtime_id


def http_interface(port: int, credential_file: Path | None = None) -> dict[str, Any]:
    """Return a runtime handle's entry for an HTTP interface on loopback.

    ``credential_file`` names the file that holds its credential, never the
    credential itself.
    """

    entry: dict[str, Any] = {"kind": "http", "address": LOOPBACK, "port": port}
    if credential_file is not None:
        entry["credential_file"] = str(credential_file)
    return entry


def port_available(port: int) -> bool:
    """Return whether nothing listens on a loopback TCP port.

    Closed connections in TIME_WAIT don't hold the port, as they don't for a
    server that reuses its address.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
        candidate.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            candidate.bind((LOOPBACK, port))
        except (OSError, OverflowError):
            return False
    return True


def _port_unavailable(
    name: str, port: int, *, code: str, message: str
) -> LifecycleError:
    return LifecycleError(
        code, message, metadata={"field": f"{_SETTINGS}.{name}.port", "port": port}
    )


def require_ports_free(
    name: str, ports: Iterable[int], *, adapter: str, code: str
) -> None:
    """Fail before launch when something already listens on an interface's port."""

    for port in ports:
        if not port_available(port):
            raise _port_unavailable(
                name,
                port,
                code=code,
                message=f"{adapter} {name} port {port} is already in use",
            )


def require_port_owner(
    name: str, port: int, session_id: int, *, adapter: str, code: str
) -> None:
    """Fail when a healthy port isn't held by the interface's own process session.

    Only Linux procfs can tell, so other platforms rely on the check before launch.
    """

    if sys.platform == "linux" and not port_listener_in_session(port, session_id):
        raise _port_unavailable(
            name,
            port,
            code=code,
            message=f"{adapter} {name} port {port} answered from another process",
        )


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
