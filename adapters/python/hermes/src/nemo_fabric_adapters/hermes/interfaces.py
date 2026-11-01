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


PORT_UNAVAILABLE = "hermes_interface_port_unavailable"


def _invalid(field: str, message: str) -> lifecycle.LifecycleError:
    return lifecycle.LifecycleError(
        "hermes_invalid_configuration",
        message,
        metadata={"field": f"harness.settings.{field}"},
    )


def interface_ports(settings: dict[str, Any]) -> dict[str, int]:
    """Return the enabled interfaces' loopback ports, keyed by interface name."""

    interfaces = local_interfaces.interface_settings(
        settings,
        {name: local_interfaces.InterfaceSpec() for name in HEALTH_PATHS},
        adapter="Hermes",
        code="hermes_invalid_configuration",
    )
    return {
        name: interface["port"]
        for name, interface in interfaces.items()
        if interface["enabled"]
    }


def handle_interfaces(ports: dict[str, int], token_path: Path) -> dict[str, Any]:
    """Return the runtime handle's entries for the enabled interfaces.

    Only the API server has a credential; the dashboard has no sign-in.
    """

    return {
        name: local_interfaces.http_interface(
            port, token_path if name == "api" else None
        )
        for name, port in ports.items()
    }


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
            local_interfaces.require_ports_free(
                name, [port], adapter="Hermes", code=PORT_UNAVAILABLE
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
            for child in self._children:
                local_interfaces.require_port_owner(
                    child.name,
                    child.port,
                    child.process.pid,
                    adapter="Hermes",
                    code=PORT_UNAVAILABLE,
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
