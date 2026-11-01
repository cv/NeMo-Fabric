# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import signal
import socket
import sys
from pathlib import Path
from typing import Any

import pytest

from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.common import lifecycle
from nemo_fabric_adapters.common import local_interfaces

SUPPORTED = {
    "api": local_interfaces.InterfaceSpec(),
    "dashboard": local_interfaces.InterfaceSpec(
        fields=frozenset({"public_origins"}), max_port=65_425
    ),
    "terminal": local_interfaces.InterfaceSpec(kind="terminal"),
}
INTERFACES = "harness.settings.interfaces"


def test_port_available_reports_a_bound_port():
    with socket.create_server(("127.0.0.1", 0)) as server:
        port = server.getsockname()[1]
        bound = local_interfaces.port_available(port)

    assert not bound
    assert local_interfaces.port_available(port)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux keeps TIME_WAIT sockets")
def test_port_available_ignores_closed_connections_in_time_wait():
    with socket.create_server(("127.0.0.1", 0)) as server:
        port = server.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port)) as client:
            accepted, _ = server.accept()
            # The side that closes first keeps the port in TIME_WAIT.
            accepted.close()
            client.recv(1)

    assert local_interfaces.port_available(port)


@pytest.mark.skipif(sys.platform != "linux", reason="Reads Linux procfs")
def test_port_listener_in_session_finds_the_listening_session():
    with socket.create_server(("127.0.0.1", 0)) as server:
        port = server.getsockname()[1]
        listening = local_interfaces.port_listener_in_session(port, os.getsid(0))

    assert listening
    assert not local_interfaces.port_listener_in_session(port, os.getsid(0))


@pytest.mark.skipif(os.name == "nt", reason="Requires POSIX file modes")
def test_write_private_file_replaces_an_owner_only_file(tmp_path: Path):
    path = tmp_path / "interfaces" / "api.token"

    local_interfaces.write_private_file(path, "first")
    local_interfaces.write_private_file(path, "second")

    assert path.read_text(encoding="utf-8") == "second"
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert [entry.name for entry in path.parent.iterdir()] == ["api.token"]


@pytest.mark.skipif(os.name == "nt", reason="Requires POSIX process groups")
async def test_signal_process_group_signals_the_child_when_groups_are_denied(
    monkeypatch: pytest.MonkeyPatch,
):
    def deny_group_signals(_pgid: int, _signum: int) -> None:
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(local_interfaces.os, "killpg", deny_group_signals)
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-c", "import time; time.sleep(60)", start_new_session=True
    )

    local_interfaces.signal_process_group(process.pid, signal.SIGTERM)

    assert await asyncio.wait_for(process.wait(), timeout=10) == -signal.SIGTERM


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        ({}, {}),
        ({"interfaces": {}}, {}),
        (
            {"interfaces": {"api": {"port": 8642}}},
            {"api": {"port": 8642, "enabled": True}},
        ),
        (
            {
                "interfaces": {
                    "dashboard": {
                        "port": 18_789,
                        "enabled": False,
                        "public_origins": ["https://example.test"],
                    },
                    "terminal": {},
                }
            },
            {
                "dashboard": {
                    "port": 18_789,
                    "enabled": False,
                    "public_origins": ["https://example.test"],
                },
                "terminal": {"enabled": True},
            },
        ),
    ],
)
def test_interface_settings_reads_present_interfaces(
    settings: dict[str, Any], expected: dict[str, Any]
):
    parsed = local_interfaces.interface_settings(
        settings, SUPPORTED, adapter="Example", code="example_invalid_configuration"
    )

    assert parsed == expected


@pytest.mark.parametrize(
    ("interfaces", "field"),
    [
        pytest.param([], INTERFACES, id="not-an-object"),
        pytest.param({"chat": {}}, f"{INTERFACES}.chat", id="unsupported"),
        pytest.param({"api": []}, f"{INTERFACES}.api", id="interface-not-object"),
        pytest.param(
            {"api": {"port": 1, "token": "x"}}, f"{INTERFACES}.api.token", id="unknown"
        ),
        pytest.param({"api": {}}, f"{INTERFACES}.api.port", id="missing-port"),
        pytest.param({"api": {"port": 0}}, f"{INTERFACES}.api.port", id="port-zero"),
        pytest.param(
            {"api": {"port": True}}, f"{INTERFACES}.api.port", id="boolean-port"
        ),
        pytest.param(
            {"dashboard": {"port": 65_426}},
            f"{INTERFACES}.dashboard.port",
            id="port-above-adapter-maximum",
        ),
        pytest.param(
            {"api": {"port": 1, "enabled": "yes"}},
            f"{INTERFACES}.api.enabled",
            id="enabled-not-boolean",
        ),
        pytest.param(
            {"terminal": {"port": 1}}, f"{INTERFACES}.terminal.port", id="terminal-port"
        ),
    ],
)
def test_interface_settings_rejects_invalid_settings(interfaces: object, field: str):
    with pytest.raises(lifecycle.LifecycleError) as caught:
        local_interfaces.interface_settings(
            {"interfaces": interfaces},
            SUPPORTED,
            adapter="Example",
            code="example_invalid_configuration",
        )

    assert caught.value.code == "example_invalid_configuration"
    assert caught.value.metadata == {"field": field}


@pytest.mark.parametrize(
    ("root", "expected"),
    [(None, "base/artifacts"), ("out", "base/out"), ("/sandbox/artifacts", None)],
)
def test_runtime_directory_is_under_the_artifact_root(
    tmp_path: Path, root: str | None, expected: str | None
):
    context = RuntimeContext.from_mapping(
        {
            "runtime_id": "runtime-1",
            "invocation_id": "start",
            "request_id": "start",
            "environment": {
                "environment_id": "environment-1",
                "provider": "local",
                "control_location": "external_control",
                "ownership": "caller_owned",
            },
            "artifacts": {} if root is None else {"root": root},
        }
    )
    artifact_root = Path(root) if expected is None else tmp_path / expected

    directory = local_interfaces.runtime_directory(context, tmp_path / "base", "hermes")

    assert directory == artifact_root / ".fabric/hermes/runtimes/runtime-1"


def test_require_ports_free_names_the_taken_port():
    with socket.create_server(("127.0.0.1", 0)) as server:
        port = server.getsockname()[1]
        with pytest.raises(lifecycle.LifecycleError) as caught:
            local_interfaces.require_ports_free(
                "dashboard", [port], adapter="Example", code="example_port_unavailable"
            )

    assert caught.value.code == "example_port_unavailable"
    assert caught.value.message == f"Example dashboard port {port} is already in use"
    assert caught.value.metadata == {
        "field": f"{INTERFACES}.dashboard.port",
        "port": port,
    }


@pytest.mark.skipif(sys.platform != "linux", reason="Reads Linux procfs")
def test_require_port_owner_rejects_a_port_held_by_another_session():
    with socket.create_server(("127.0.0.1", 0)) as server:
        port = server.getsockname()[1]
        local_interfaces.require_port_owner(
            "api", port, os.getsid(0), adapter="Example", code="example_unavailable"
        )
        with pytest.raises(lifecycle.LifecycleError) as caught:
            local_interfaces.require_port_owner(
                "api", port, -1, adapter="Example", code="example_unavailable"
            )

    assert caught.value.message == (
        f"Example api port {port} answered from another process"
    )


@pytest.mark.parametrize("credential_file", [None, Path("/sandbox/api.token")])
def test_http_interface_reports_a_loopback_port_and_credential_location(
    credential_file: Path | None,
):
    entry = local_interfaces.http_interface(8642, credential_file)

    assert entry == {
        "kind": "http",
        "address": "127.0.0.1",
        "port": 8642,
        **(
            {} if credential_file is None else {"credential_file": str(credential_file)}
        ),
    }
