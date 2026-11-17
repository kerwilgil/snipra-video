#!/usr/bin/env python3
"""Export the canonical Brand V2 rounded app icons and the Windows ICO."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import struct
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError as exc:  # pragma: no cover - exercised by CLI users without Pillow
    raise SystemExit("Pillow is required: install it in the tooling environment with `python -m pip install Pillow`.") from exc


SIZES = (16, 32, 48, 64, 128, 180, 192, 256, 512, 1024)
ICO_SIZES = (16, 32, 48, 64, 128, 256)
MASTER_SIZE = 1254
CORNER_RADIUS_RATIO = 0.20
SUPERSAMPLE = 4


def rounded_square_mask(size: int) -> Image.Image:
    """Return a high-quality alpha mask matching the existing Brand V2 rounded-square silhouette."""
    scale = SUPERSAMPLE
    high = Image.new("L", (size * scale, size * scale), 0)
    radius = round((size - 1) * CORNER_RADIUS_RATIO * scale)
    ImageDraw.Draw(high).rounded_rectangle(
        (0, 0, (size - 1) * scale, (size - 1) * scale),
        radius=radius,
        fill=255,
    )
    # BOX keeps coverage antialiased without Lanczos' negative-lobe alpha fringe.
    return high.resize((size, size), Image.Resampling.BOX)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    if not path.is_file():
        return entries
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"Invalid checksum entry in {path}: {line!r}") from exc
        if relative in entries:
            raise ValueError(f"Duplicate checksum entry in {path}: {relative}")
        entries[relative] = digest
    return entries


def corner_alpha(image: Image.Image) -> tuple[int, int, int, int]:
    alpha = image.getchannel("A")
    width, height = image.size
    return tuple(alpha.getpixel(point) for point in ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)))  # type: ignore[return-value]


def write_ico(root: Path, icons: dict[int, Image.Image]) -> None:
    """Write classic 32-bit DIB ICO frames, retaining the PNG alpha channel."""
    _, _, _, ico_path = paths(root)
    frames: list[tuple[int, bytes]] = []
    for size in ICO_SIZES:
        icon = icons[size]
        rgba = icon.convert("RGBA")
        bgra = rgba.tobytes("raw", "BGRA")
        row_bytes = size * 4
        xor = b"".join(bgra[row * row_bytes : (row + 1) * row_bytes] for row in range(size - 1, -1, -1))
        mask_row_bytes = ((size + 31) // 32) * 4
        and_mask = bytearray(mask_row_bytes * size)
        alpha = rgba.getchannel("A")
        for row in range(size):
            source_y = size - 1 - row
            for x in range(size):
                if alpha.getpixel((x, source_y)) < 128:
                    and_mask[row * mask_row_bytes + x // 8] |= 1 << (7 - (x % 8))
        image_data = struct.pack(
            "<IiiHHIIiiII",
            40,
            size,
            size * 2,
            1,
            32,
            0,
            len(xor) + len(and_mask),
            0,
            0,
            0,
            0,
        ) + xor + bytes(and_mask)
        frames.append((size, image_data))

    offset = 6 + 16 * len(frames)
    directory = bytearray(struct.pack("<HHH", 0, 1, len(frames)))
    for size, frame in frames:
        dim = 0 if size == 256 else size
        directory.extend(struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(frame), offset))
        offset += len(frame)
    ico_path.write_bytes(bytes(directory) + b"".join(frame for _, frame in frames))


def paths(root: Path) -> tuple[Path, Path, Path, Path]:
    pack = root / "frontend" / "image" / "snipra-brand-v2"
    return (
        pack / "app-icons" / "snipra-app-icon-master-1254.png",
        pack / "app-icons",
        pack / "SHA256SUMS.txt",
        root / "snipra.ico",
    )


def web_paths(root: Path) -> dict[str, tuple[Path, Path]]:
    pack_web = root / "frontend" / "image" / "snipra-brand-v2" / "web"
    public = root / "frontend" / "public"
    return {
        "favicon.ico": (pack_web / "favicon.ico", public / "favicon.ico"),
        "apple-touch-icon.png": (pack_web / "apple-touch-icon.png", public / "apple-touch-icon.png"),
        "icon-192.png": (pack_web / "icon-192.png", public / "icon-192.png"),
        "icon-512.png": (pack_web / "icon-512.png", public / "icon-512.png"),
    }


def verify(root: Path, master: Image.Image | None = None) -> None:
    master_path, icon_dir, sums_path, ico_path = paths(root)
    if master is None:
        if not master_path.is_file():
            raise ValueError(f"Brand V2 master is missing: {master_path}")
        master = Image.open(master_path)
    if master.mode != "RGBA":
        raise ValueError(f"Brand V2 master must be RGBA; got {master.mode}")
    if master.size != (MASTER_SIZE, MASTER_SIZE):
        raise ValueError(f"Brand V2 master must be {MASTER_SIZE}x{MASTER_SIZE}; got {master.size}")
    expected_master_mask = rounded_square_mask(MASTER_SIZE)
    if master.getchannel("A").tobytes() != expected_master_mask.tobytes():
        raise ValueError("Brand V2 master alpha does not match the canonical rounded-square mask")
    if any(corner_alpha(master)):
        raise ValueError("Brand V2 master has an opaque or partially opaque corner")

    required_manifest_paths: set[str] = set()
    for size in SIZES:
        path = icon_dir / f"snipra-app-icon-{size}.png"
        if not path.is_file():
            raise ValueError(f"Missing app icon export: {path}")
        with Image.open(path) as icon:
            if icon.mode != "RGBA" or icon.size != (size, size):
                raise ValueError(f"{path.name} must be RGBA {size}x{size}; got {icon.mode} {icon.size}")
            if any(corner_alpha(icon)):
                raise ValueError(f"{path.name} has a non-transparent corner: {corner_alpha(icon)}")
            if icon.getchannel("A").tobytes() != rounded_square_mask(size).tobytes():
                raise ValueError(f"{path.name} alpha does not match the rounded-square mask")
        required_manifest_paths.add(f"app-icons/{path.name}")

    for name, (pack_path, public_path) in web_paths(root).items():
        if not pack_path.is_file() or not public_path.is_file():
            raise ValueError(f"Missing web icon asset: {pack_path if not pack_path.is_file() else public_path}")
        if pack_path.read_bytes() != public_path.read_bytes():
            raise ValueError(f"Frontend public {name} differs from the Brand V2 web asset")
        if name == "favicon.ico":
            if pack_path.read_bytes() != ico_path.read_bytes():
                raise ValueError("Brand V2 web favicon differs from the canonical Windows ICO")
        else:
            source_size = {"apple-touch-icon.png": 180, "icon-192.png": 192, "icon-512.png": 512}[name]
            with Image.open(pack_path) as web_icon, Image.open(icon_dir / f"snipra-app-icon-{source_size}.png") as source_icon:
                if web_icon.mode != "RGBA" or web_icon.size != (source_size, source_size):
                    raise ValueError(f"{name} must be RGBA {source_size}x{source_size}")
                if web_icon.tobytes() != source_icon.tobytes() or any(corner_alpha(web_icon)):
                    raise ValueError(f"{name} lost its rounded alpha or differs from Brand V2")
        required_manifest_paths.add(f"web/{name}")

    required_manifest_paths.add(f"app-icons/{master_path.name}")
    pack_root = sums_path.parent.resolve()
    manifest = read_manifest(sums_path)
    for relative, digest in manifest.items():
        target = (pack_root / relative).resolve()
        if not target.is_relative_to(pack_root):
            raise ValueError(f"Checksum path escapes the Brand V2 pack: {relative}")
        if not target.is_file() or sha256(target) != digest:
            raise ValueError(f"SHA256SUMS.txt mismatch: {relative}")
    missing_entries = required_manifest_paths - manifest.keys()
    if missing_entries:
        raise ValueError(f"SHA256SUMS.txt is missing generated assets: {sorted(missing_entries)}")

    if not ico_path.is_file():
        raise ValueError(f"Windows ICO is missing: {ico_path}")
    with Image.open(ico_path) as ico:
        if set(ico.ico.sizes()) != {(size, size) for size in ICO_SIZES}:
            raise ValueError(f"ICO must contain exactly {ICO_SIZES}; got {sorted(ico.ico.sizes())}")
        for size in ICO_SIZES:
            frame = ico.ico.getimage((size, size))
            if frame.mode != "RGBA" or any(corner_alpha(frame)):
                raise ValueError(f"ICO {size}x{size} frame lost transparent alpha corners")
            with Image.open(icon_dir / f"snipra-app-icon-{size}.png") as expected:
                if frame.tobytes() != expected.tobytes():
                    raise ValueError(f"ICO {size}x{size} frame differs from its Brand V2 PNG source")


def export(root: Path) -> None:
    master_path, icon_dir, sums_path, ico_path = paths(root)
    if not master_path.is_file():
        raise ValueError(f"Brand V2 master is missing: {master_path}")
    with Image.open(master_path) as source:
        if source.mode != "RGBA":
            raise ValueError(f"Brand V2 master must be RGBA; got {source.mode}")
        if source.size != (MASTER_SIZE, MASTER_SIZE):
            raise ValueError(f"Brand V2 master must be {MASTER_SIZE}x{MASTER_SIZE}; got {source.size}")
        master = source.copy()

    expected_mask = rounded_square_mask(MASTER_SIZE)
    original_alpha = master.getchannel("A")
    opaque = original_alpha.getextrema() == (255, 255)
    already_exported = original_alpha.tobytes() == expected_mask.tobytes()
    if not (opaque or already_exported):
        raise ValueError("Brand V2 master has unexpected alpha; refusing to alter its artwork")

    # Preserve all RGB artwork from the supplied master; only set the exterior alpha.
    master.putalpha(expected_mask)
    master.save(master_path, format="PNG", optimize=True)

    icons: dict[int, Image.Image] = {}
    for size in SIZES:
        exported = master.resize((size, size), Image.Resampling.LANCZOS)
        # Draw the target-size mask separately to avoid Lanczos ringing at the four corners.
        exported.putalpha(rounded_square_mask(size))
        exported.save(icon_dir / f"snipra-app-icon-{size}.png", format="PNG", optimize=True)
        icons[size] = exported

    write_ico(root, icons)

    # Keep the packaged web favicon and touch/PWA icons byte-identical to the
    # canonical Brand V2 exports; these are the assets actually served by Vite.
    web_assets = web_paths(root)
    for name, (pack_path, public_path) in web_assets.items():
        pack_path.parent.mkdir(parents=True, exist_ok=True)
        public_path.parent.mkdir(parents=True, exist_ok=True)
        source = ico_path if name == "favicon.ico" else icon_dir / {
            "apple-touch-icon.png": "snipra-app-icon-180.png",
            "icon-192.png": "snipra-app-icon-192.png",
            "icon-512.png": "snipra-app-icon-512.png",
        }[name]
        shutil.copyfile(source, pack_path)
        shutil.copyfile(source, public_path)

    entries = read_manifest(sums_path)
    generated_paths = {
        f"app-icons/snipra-app-icon-{size}.png": icon_dir / f"snipra-app-icon-{size}.png"
        for size in SIZES
    }
    generated_paths[f"app-icons/{master_path.name}"] = master_path
    generated_paths.update({f"web/{name}": pair[0] for name, pair in web_paths(root).items()})
    for relative, path in generated_paths.items():
        entries[relative] = sha256(path)
    sums_path.write_text(
        "\n".join(f"{digest}  {relative}" for relative, digest in entries.items()) + "\n",
        encoding="utf-8",
    )
    verify(root, master)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="repository root (defaults to this script's repository)")
    parser.add_argument("--check", action="store_true", help="validate current PNG, ICO, and SHA256 outputs without writing files")
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.check:
            verify(root)
            print("Brand V2 PNG / ICO / SHA256 validation: PASS")
        else:
            export(root)
            print("Brand V2 app icons exported and validated: PASS")
    except (OSError, ValueError) as exc:
        print(f"app icon export failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
