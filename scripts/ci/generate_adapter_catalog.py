# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bundle canonical adapter metadata without importing adapters or harness SDKs."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any


CATALOG = Path("sdk/python/nemo-fabric-adapter-catalog")
BUNDLE = CATALOG / "src/nemo_fabric_adapter_catalog/catalog.json"
INTERFACE_SETTINGS = Path("schemas/adapter-contract/interface-settings.schema.json")


def check_interface_settings(root: Path, descriptor: dict[str, Any], source: Path) -> None:
    """Keep a descriptor's interface settings on the shared adapter-contract fragment.

    Each interface the descriptor declares is listed under
    `settings_schema.properties.interfaces`, which copies the fragment's `$defs`
    unchanged and closes every object with `additionalProperties: false`.
    """
    settings = descriptor.get("settings_schema") or {}
    interfaces = (settings.get("properties") or {}).get("interfaces")
    declared = descriptor.get("interfaces") or {}
    if interfaces is None and not declared:
        return

    def invalid(message: str) -> ValueError:
        return ValueError(f"Invalid interface settings in {source}: {message}")

    if interfaces is None or interfaces.get("additionalProperties") is not False:
        raise invalid("settings_schema interfaces must reject unknown interfaces")
    offered = interfaces.get("properties") or {}
    if sorted(offered) != sorted(declared):
        raise invalid("settings_schema interfaces must match the declared interfaces")
    fragment = json.loads((root / INTERFACE_SETTINGS).read_bytes())["$defs"]
    defs = settings.get("$defs") or {}
    if any(defs.get(name) != value for name, value in fragment.items()):
        raise invalid(f"settings_schema $defs must copy {INTERFACE_SETTINGS} unchanged")
    for name, interface in offered.items():
        fields = interface.get("properties") or {}
        http = declared[name].get("kind") == "http"
        references = {
            "enabled": "interface_enabled",
            **({"port": "interface_port"} if http else {}),
        }
        if (
            interface.get("additionalProperties") is not False
            or any(
                (fields.get(field) or {}).get("$ref") != f"#/$defs/{reference}"
                for field, reference in references.items()
            )
            or http != ("port" in (interface.get("required") or []))
            or (not http and "port" in fields)
        ):
            raise invalid(f"interface {name!r} must use the shared fields for its kind")


def catalog_data(root: Path) -> dict[str, Any]:
    """Collect leaf-owned descriptors with source metadata outside their contracts."""
    version = tomllib.loads(
        (root / CATALOG / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["version"]
    catalog: dict[str, Any] = {
        "format_version": 1,
        "catalog_version": version,
        "adapters": {},
        "targets": {},
    }
    for ecosystem, manifest_name in (
        ("python", "pyproject.toml"),
        ("typescript", "package.json"),
    ):
        for manifest in sorted(
            (root / "adapters" / ecosystem).glob(f"*/{manifest_name}")
        ):
            adapters = sorted(manifest.parent.glob("*.fabric-adapter.json"))
            if not adapters:
                continue
            if ecosystem == "python":
                package = tomllib.loads(manifest.read_text(encoding="utf-8"))["project"]
            else:
                package = json.loads(manifest.read_bytes())
            targets = sorted(
                (manifest.parent / "targets").rglob("*.fabric-target.json")
            )
            for kind, sources in (("adapters", adapters), ("targets", targets)):
                for source in sources:
                    descriptor = json.loads(source.read_bytes())
                    record_id = descriptor["adapter_id" if kind == "adapters" else "id"]
                    if not isinstance(record_id, str) or not record_id:
                        raise ValueError(
                            f"Invalid descriptor ID in {source}: {record_id!r}"
                        )
                    if kind == "adapters":
                        check_interface_settings(root, descriptor, source)
                    if record_id in catalog[kind]:
                        raise ValueError(
                            f"Duplicate {kind} ID {record_id!r} in {source}"
                        )
                    canonical = json.dumps(
                        descriptor,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ).encode("utf-8")
                    catalog[kind][record_id] = {
                        "descriptor": descriptor,
                        "source": source.relative_to(root).as_posix(),
                        "ecosystem": ecosystem,
                        "package": {
                            "name": package["name"],
                            "version": package["version"],
                        },
                        "sha256": hashlib.sha256(canonical).hexdigest(),
                    }
    if not catalog["adapters"]:
        raise ValueError("No bundled adapter descriptors found")
    for target_id, record in catalog["targets"].items():
        if record["descriptor"]["adapter_id"] not in catalog["adapters"]:
            raise ValueError(f"Target {target_id!r} references an unbundled adapter")
    return catalog


def generate_catalog(root: Path, *, check: bool = False) -> None:
    expected = (
        json.dumps(catalog_data(root), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    )
    output = root / BUNDLE
    # Git can check out JSON with CRLF on Windows; line endings are not metadata.
    actual = output.read_text(encoding="utf-8") if output.exists() else None
    if check:
        if actual != expected:
            raise ValueError("Adapter catalog is stale; run just adapter-catalog")
    elif actual != expected:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(expected, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Fail if the committed bundle is stale"
    )
    args = parser.parse_args()
    try:
        generate_catalog(Path(__file__).resolve().parents[2], check=args.check)
    except (ValueError, KeyError) as error:
        parser.exit(1, f"{error}\n")
