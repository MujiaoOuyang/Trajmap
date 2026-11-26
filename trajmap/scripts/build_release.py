#!/usr/bin/env python3
"""Create a source-only release archive from an explicit file allowlist."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def collect_payloads(root: Path, manifest: Path) -> dict[str, bytes]:
    root = root.resolve()
    entries = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest 'files' must be a nonempty list")
    payloads = {}
    for entry in entries:
        if not isinstance(entry, str) or not entry or "\\" in entry:
            raise ValueError(f"Invalid manifest entry: {entry!r}")
        path = PurePosixPath(entry)
        if path.is_absolute() or ".." in path.parts or str(path) != entry:
            raise ValueError(f"Not a canonical relative path: {entry}")
        if entry in payloads or entry == "SHA256SUMS":
            raise ValueError(f"Duplicate or reserved archive entry: {entry}")
        source = root / path
        if any((root.joinpath(*path.parts[:i])).is_symlink() for i in range(1, len(path.parts) + 1)):
            raise ValueError(f"Symlinks cannot be published: {entry}")
        if not source.is_file():
            raise FileNotFoundError(source)
        payloads[entry] = source.read_bytes()

    payloads["SHA256SUMS"] = "".join(
        f"{hashlib.sha256(payloads[name]).hexdigest()}  {name}\n"
        for name in sorted(payloads)
    ).encode("ascii")
    return payloads


def export_directory(root: Path, manifest: Path, output: Path) -> list[str]:
    """Write only allowlisted source files into a new GitHub upload directory."""
    payloads = collect_payloads(root, manifest)
    output.mkdir(parents=True, exist_ok=False)
    for name, content in sorted(payloads.items()):
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(content)
    return sorted(set(payloads) - {"SHA256SUMS"})


def build_release(root: Path, manifest: Path, output: Path) -> list[str]:
    payloads = collect_payloads(root, manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects previous release artifacts and any local input.
    with output.open("xb") as stream:
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, content in sorted(payloads.items()):
                info = zipfile.ZipInfo("trajmap/" + name, date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, content)
    return sorted(set(payloads) - {"SHA256SUMS"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("dist/trajmap.zip"))
    parser.add_argument("--directory", type=Path, help="Also export a new source-only directory for GitHub")
    args = parser.parse_args()
    manifest = args.manifest or args.root / "release_manifest.json"
    if args.directory is not None:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(args.output)
        export_directory(args.root, manifest, args.directory)
        print(f"Prepared GitHub upload directory: {args.directory}")
    files = build_release(args.root, manifest, args.output)
    print(f"Archived {len(files)} allowlisted files and SHA256SUMS: {args.output}")
    print("No upload performed. Review licensing and source limitations before publication.")


if __name__ == "__main__":
    main()
