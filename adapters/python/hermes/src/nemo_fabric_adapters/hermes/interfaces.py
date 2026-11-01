# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve Hermes's API server and dashboard as children of a managed runtime."""

from __future__ import annotations

import asyncio
import math
import os
import secrets
import signal
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemo_fabric_adapters.common import lifecycle
from nemo_fabric_adapters.common import local_interfaces

HEALTH_PATHS = {"api": "/health", "dashboard": "/api/status"}
DEFAULT_STARTUP_TIMEOUT_SECONDS = 60.0
# Fabric gives the adapter host 90 s to start; leave time for cleanup.
MAX_STARTUP_TIMEOUT_SECONDS = 75.0
# Fabric gives the adapter host 10 s to stop, including this wait.
STOP_GRACE_SECONDS = 5.0
# Hermes's gateway waits this long for in-flight API runs and cron jobs on
# SIGTERM, so it exits within STOP_GRACE_SECONDS.
GATEWAY_DRAIN_TIMEOUT_SECONDS = 3


def _invalid(field: str, message: str) -> lifecycle.LifecycleError:
    return lifecycle.LifecycleError(
        "hermes_invalid_configuration",
        message,
        metadata={"field": f"harness.settings.{field}"},
    )


def interface_ports(settings: dict[str, Any]) -> dict[str, int]:
    """Return the enabled interfaces' loopback ports, keyed by interface name."""

    interfaces = settings.get("interfaces", {})
    if not isinstance(interfaces, dict):
        raise _invalid("interfaces", "Hermes interfaces must be an object")
    unsupported = sorted(set(interfaces) - set(HEALTH_PATHS))
    if unsupported:
        raise _invalid(
            f"interfaces.{unsupported[0]}",
            f"Hermes doesn't support the {unsupported[0]!r} interface",
        )
    ports: dict[str, int] = {}
    for name in HEALTH_PATHS:
        if name not in interfaces:
            continue
        field = f"interfaces.{name}"
        interface = interfaces[name]
        if not isinstance(interface, dict):
            raise _invalid(field, f"Hermes {name} interface must be an object")
        unknown = sorted(set(interface) - {"enabled", "port"})
        if unknown:
            raise _invalid(
                f"{field}.{unknown[0]}",
                f"Hermes {name} interface doesn't accept {unknown[0]!r}",
            )
        port = interface.get("port")
        if (
            isinstance(port, bool)
            or not isinstance(port, int)
            or not 1 <= port <= 65535
        ):
            raise _invalid(
                f"{field}.port",
                f"Hermes {name} port must be an integer from 1 to 65535",
            )
        enabled = interface.get("enabled", True)
        if not isinstance(enabled, bool):
            raise _invalid(
                f"{field}.enabled", f"Hermes {name} enabled must be a boolean"
            )
        if enabled:
            ports[name] = port
    return ports


def startup_timeout(settings: dict[str, Any]) -> float:
    """Return the deadline for every enabled interface to pass its health check."""

    value = settings.get("startup_timeout_seconds", DEFAULT_STARTUP_TIMEOUT_SECONDS)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 < value <= MAX_STARTUP_TIMEOUT_SECONDS
    ):
        raise _invalid(
            "startup_timeout_seconds",
            "Hermes startup_timeout_seconds must be a number greater than 0 "
            f"and at most {MAX_STARTUP_TIMEOUT_SECONDS:g}",
        )
    return float(value)


@dataclass
class _Child:
    name: str
    port: int
    process: asyncio.subprocess.Process


class InterfaceChildren:
    """Start, track and stop one runtime's interface processes."""

    def __init__(self) -> None:
        self._children: list[_Child] = []
        self._token_path: Path | None = None

    async def start(
        self,
        ports: dict[str, int],
        *,
        hermes_home: Path,
        token_path: Path,
        timeout: float,
        environment: dict[str, str],
    ) -> None:
        """Start the enabled interfaces and write the API token once all pass health."""

        if os.name == "nt":
            raise lifecycle.LifecycleError(
                "hermes_interfaces_unsupported_platform",
                "Hermes interfaces require a POSIX host",
                metadata={"field": "harness.settings.interfaces"},
            )
        if len(set(ports.values())) < len(ports):
            raise lifecycle.LifecycleError(
                "hermes_interface_ports_equal",
                "Hermes api and dashboard ports must differ",
                metadata={"field": "harness.settings.interfaces.dashboard.port"},
            )
        for name, port in ports.items():
            if not local_interfaces.port_available(port):
                raise _port_unavailable(
                    name, port, f"Hermes {name} port {port} is already in use"
                )
        web_dist = _web_dist() if "dashboard" in ports else None
        token = secrets.token_urlsafe(32)
        log_dir = hermes_home / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        deadline = asyncio.get_running_loop().time() + timeout
        try:
            for name, port in ports.items():
                arguments, child_environment = _child_command(
                    name, port, token=token, web_dist=web_dist, environment=environment
                )
                with open(log_dir / f"fabric-interface-{name}.log", "ab") as log:
                    process = await asyncio.create_subprocess_exec(
                        sys.executable,
                        "-m",
                        "hermes_cli.main",
                        *arguments,
                        env=child_environment,
                        stdin=asyncio.subprocess.DEVNULL,
                        stdout=log,
                        stderr=asyncio.subprocess.STDOUT,
                        start_new_session=True,
                    )
                self._children.append(_Child(name, port, process))
            waits = [
                asyncio.create_task(_wait_healthy(child, deadline))
                for child in self._children
            ]
            try:
                await asyncio.gather(*waits)
            finally:
                for wait in waits:
                    wait.cancel()
            if sys.platform == "linux":
                for child in self._children:
                    if not local_interfaces.port_listener_in_session(
                        child.port, child.process.pid
                    ):
                        raise _port_unavailable(
                            child.name,
                            child.port,
                            f"Hermes {child.name} port {child.port} answered from another process",
                        )
            if "api" in ports:
                local_interfaces.write_private_file(token_path, token)
                self._token_path = token_path
        except BaseException:
            await self.stop()
            raise

    async def stop(self) -> None:
        """Remove the API token, then stop every interface process group."""

        token_path, self._token_path = self._token_path, None
        if token_path is not None:
            token_path.unlink(missing_ok=True)
        children, self._children = self._children, []
        running = [child for child in children if child.process.returncode is None]
        for child in running:
            local_interfaces.signal_process_group(child.process.pid, signal.SIGTERM)
        if running:
            waits = [asyncio.ensure_future(child.process.wait()) for child in running]
            _, pending = await asyncio.wait(waits, timeout=STOP_GRACE_SECONDS)
            for child in running:
                local_interfaces.signal_process_group(child.process.pid, signal.SIGKILL)
            if pending:
                await asyncio.wait(pending)


def _port_unavailable(name: str, port: int, message: str) -> lifecycle.LifecycleError:
    return lifecycle.LifecycleError(
        "hermes_interface_port_unavailable",
        message,
        metadata={"field": f"harness.settings.interfaces.{name}.port", "port": port},
    )


def _web_dist() -> Path:
    configured = os.environ.get("HERMES_WEB_DIST")
    if configured:
        dist = Path(configured)
    else:
        import hermes_cli

        dist = Path(hermes_cli.__file__).parent / "web_dist"
    if not (dist / "index.html").is_file():
        raise lifecycle.LifecycleError(
            "hermes_dashboard_frontend_missing",
            f"Hermes's dashboard needs a built frontend at {dist}; set "
            "HERMES_WEB_DIST to one",
            metadata={"field": "harness.settings.interfaces.dashboard"},
        )
    return dist


async def _wait_healthy(child: _Child, deadline: float) -> None:
    loop = asyncio.get_running_loop()
    while loop.time() < deadline:
        if child.process.returncode is not None:
            raise lifecycle.LifecycleError(
                "hermes_interface_start_failed",
                f"Hermes {child.name} exited before passing its health check",
                metadata={"interface": child.name},
            )
        if await _http_status(child.port, HEALTH_PATHS[child.name]) == 200:
            return
        await asyncio.sleep(0.1)
    raise lifecycle.LifecycleError(
        "hermes_interface_start_timeout",
        f"Hermes {child.name} didn't pass its health check before the startup timeout",
        retryable=True,
        metadata={"interface": child.name},
    )


def _child_command(
    name: str,
    port: int,
    *,
    token: str,
    web_dist: Path | None,
    environment: dict[str, str],
) -> tuple[list[str], dict[str, str]]:
    """Return one interface's Hermes CLI arguments and environment."""
    child_environment = {
        key: value
        for key, value in environment.items()
        if not key.startswith("API_SERVER_")
    }
    if name == "api":
        child_environment.update(
            API_SERVER_KEY=token,
            API_SERVER_HOST="127.0.0.1",
            API_SERVER_PORT=str(port),
        )
        return ["gateway", "run"], child_environment
    child_environment["HERMES_WEB_DIST"] = str(web_dist)
    # --isolated keeps one dashboard per runtime instead of attaching to another
    # runtime's dashboard on this host.
    return [
        "dashboard",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--no-open",
        "--skip-build",
        "--isolated",
    ], child_environment


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    """Return a redirect as its own status instead of following it."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


async def _http_status(port: int, path: str) -> int | None:
    """Return a loopback GET's status, or None when nothing answers."""

    def get() -> int | None:
        # No proxies: the sandbox's proxy can't reach this process's loopback.
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirects()
        )
        try:
            with opener.open(f"http://127.0.0.1:{port}{path}", timeout=1.0) as response:
                return response.status
        except urllib.error.HTTPError as error:
            return error.code
        except OSError:
            return None

    return await asyncio.to_thread(get)
