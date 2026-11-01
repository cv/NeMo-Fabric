# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A consumer reaches each P0 interface through the runtime handle alone.

The consumer below starts a runtime through the runtime server, rediscovers it
with the read-only ``status`` command, and reaches every interface the handle
reports. It knows no adapter's ports or paths. Only signing in is
adapter-specific, as NVIDIA/NeMo-Fabric#565 allows until #566 declares
authentication schemes.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from nemo_fabric.runtime_server.state import RuntimeServer

ROOT = Path(__file__).resolve().parents[2]
DEEPAGENTS_CODE = ROOT / "adapters/typescript/deepagents-code"
ACP_FIXTURE = ROOT / "adapters/typescript/common/test/fixtures/acp-agent.mjs"

# Per-adapter sign-in: the request that proves an HTTP interface accepts its
# stored credential, or the arguments that open a terminal interface.
SIGN_IN: dict[tuple[str, str], Any] = {
    ("openclaw", "dashboard"): "/v1/models",
    ("hermes", "api"): "/v1/models",
    ("hermes", "dashboard"): "/",
    ("deepagents-code", "terminal"): ["--version"],
}


def _free_port() -> int:
    with socket.create_server(("127.0.0.1", 0)) as server:
        return server.getsockname()[1]


def _config(
    name: str, adapter_id: str, descriptor: Path, settings: dict[str, Any], **extra: Any
) -> dict[str, Any]:
    return {
        "schema_version": "fabric.agent/v1alpha1",
        "metadata": {"name": name},
        "harness": {"adapter_id": adapter_id, "settings": settings},
        "discovery": {"local_paths": [str(descriptor)]},
        **extra,
    }


def _openclaw(tmp_path: Path) -> dict[str, Any]:
    if importlib.util.find_spec("nemo_fabric_adapters.openclaw") is None:
        pytest.skip("The OpenClaw adapter is not installed")
    return _config(
        "openclaw-interfaces",
        "nvidia.fabric.openclaw",
        ROOT / "adapters/python/openclaw/openclaw.fabric-adapter.json",
        {
            "openclaw_command": str(ROOT / "tests/_utils/mock_openclaw.py"),
            "interfaces": {"dashboard": {"port": _free_port()}},
        },
        models={"default": {"provider": "test", "model": "fabric-echo"}},
        environment={
            "provider": "local",
            "env": {"FAKE_OPENCLAW_CAPTURE": str(tmp_path / "openclaw.json")},
        },
    )


def _hermes(tmp_path: Path) -> dict[str, Any]:
    if importlib.util.find_spec("run_agent") is None:
        pytest.skip("Hermes Agent is not installed; run with the Hermes venv")
    web_dist = tmp_path / "web_dist"
    web_dist.mkdir()
    (web_dist / "index.html").write_text("<html><body>dashboard</body></html>")
    # Run the adapter host in this interpreter, which has Hermes installed.
    os.environ["ADAPTER_PYTHON"] = sys.executable
    os.environ["FABRIC_TEST_MODEL_KEY"] = "fabric-test-model-key"
    return _config(
        "hermes-interfaces",
        "nvidia.fabric.hermes",
        ROOT / "adapters/python/hermes/hermes.fabric-adapter.json",
        {
            "interfaces": {
                "api": {"port": _free_port()},
                "dashboard": {"port": _free_port()},
            }
        },
        models={
            "default": {
                "provider": "openai",
                "model": "fabric-echo",
                "base_url": "http://127.0.0.1:9/v1",
                "api_key_env": "FABRIC_TEST_MODEL_KEY",
            }
        },
        environment={
            "provider": "local",
            "env": {"HERMES_WEB_DIST": str(web_dist)},
        },
    )


def _deepagents_code(tmp_path: Path) -> dict[str, Any]:
    node = shutil.which("node")
    if node is None or not (DEEPAGENTS_CODE / "dist/cli.js").is_file():
        pytest.skip("Build the TypeScript adapters with Node.js to run")
    # A dcode stand-in that serves the shared ACP fixture, or prints its
    # arguments as the terminal UI.
    dcode = tmp_path / "dcode"
    dcode.write_text(
        "#!/bin/sh\n"
        'if [ "$1" != "--acp" ]; then echo "tui $*"; exit 0; fi\n'
        f'exec "{node}" "{ACP_FIXTURE}"\n'
    )
    dcode.chmod(0o755)
    os.environ["FABRIC_TEST_MODEL_KEY"] = "fabric-test-model-key"
    return _config(
        "deepagents-code-interfaces",
        "nvidia.fabric.langchain.deepagents-code",
        DEEPAGENTS_CODE / "deepagents-code.fabric-adapter.json",
        {
            "approval_mode": "auto",
            "dcode_command": str(dcode),
            "interfaces": {"terminal": {}},
        },
        models={
            "default": {
                "provider": "openai",
                "model": "fabric-echo",
                "base_url": "http://127.0.0.1:9/v1",
                "api_key_env": "FABRIC_TEST_MODEL_KEY",
            }
        },
        environment={"provider": "local", "workspace": str(tmp_path)},
    )


def _reach(adapter: str, name: str, interface: dict[str, Any]) -> None:
    """Reach one interface from its handle entry and the stored credential."""

    sign_in = SIGN_IN[(adapter, name)]
    if interface["kind"] == "terminal":
        opened = subprocess.run(
            [interface["launcher"], *sign_in],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert opened.returncode == 0, opened.stderr
        return
    headers = {}
    if "credential_file" in interface:
        token = Path(interface["credential_file"]).read_text(encoding="utf-8")
        headers["Authorization"] = f"Bearer {token}"
    url = f"http://{interface['address']}:{interface['port']}{sign_in}"
    response = httpx.get(url, headers=headers, timeout=10, trust_env=False)
    assert response.status_code == 200, (name, response.status_code)


@pytest.mark.skipif(sys.platform == "win32", reason="Requires POSIX file permissions")
@pytest.mark.parametrize(
    ("adapter", "config", "expected"),
    [
        pytest.param("openclaw", _openclaw, {"dashboard": "http"}, id="openclaw"),
        pytest.param(
            "hermes", _hermes, {"api": "http", "dashboard": "http"}, id="hermes"
        ),
        pytest.param(
            "deepagents-code",
            _deepagents_code,
            {"terminal": "terminal"},
            id="deepagents-code",
        ),
    ],
)
async def test_consumer_reaches_each_interface_through_the_handle(
    tmp_path: Path, adapter: str, config: Any, expected: dict[str, str]
):
    server = RuntimeServer(directory=tmp_path)
    fabric_config = config(tmp_path)
    agent = fabric_config["metadata"]["name"]
    request = {"interface_version": 2, "agent": agent}
    started = await server.handle(
        {**request, "operation": "start", "config": fabric_config}
    )
    assert started["status"] == "succeeded", started["error"]
    runtime_id = started["result"]["runtime_id"]
    try:
        status = await server.handle(
            {**request, "operation": "status", "runtime_id": runtime_id}
        )
        interfaces = status["result"]["interfaces"]
        for name, interface in interfaces.items():
            _reach(adapter, name, interface)
        credentials = [
            Path(interface["credential_file"]).read_text(encoding="utf-8")
            for interface in interfaces.values()
            if "credential_file" in interface
        ]
    finally:
        stopped = await server.handle(
            {**request, "operation": "stop", "runtime_id": runtime_id}
        )

    assert stopped["status"] == "succeeded", stopped["error"]
    assert status["result"] == started["result"]
    assert {name: entry["kind"] for name, entry in interfaces.items()} == expected
    # The handle reports the P0 locations that consumers find by convention.
    directory = (
        tmp_path
        / "artifacts/.fabric"
        / adapter
        / "runtimes"
        / runtime_id
        / "interfaces"
    )
    for name, interface in interfaces.items():
        location = interface.get("credential_file", interface.get("launcher"))
        if location is not None:
            assert Path(location).parent == directory
            assert Path(location).name in (f"{name}.token", name)
    encoded = json.dumps([started, status])
    assert all(credential not in encoded for credential in credentials)
