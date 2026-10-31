# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import signal
import socket
import sys
from pathlib import Path

import pytest

from nemo_fabric_adapters.common import local_interfaces


def test_port_available_reports_a_bound_port():
    with socket.create_server(("127.0.0.1", 0)) as server:
        port = server.getsockname()[1]
        bound = local_interfaces.port_available(port)

    assert not bound
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
