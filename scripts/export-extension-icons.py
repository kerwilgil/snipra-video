"""Reexporta el isotipo transparente Brand V2 para la toolbar Chromium.

Ejecutar con backend/.venv/Scripts/python.exe scripts/export-extension-icons.py.
Remuestrea el isotipo extraído de Brand V2 sin deformarlo.
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "extension/icons"
QA = ROOT / "artifacts/qa/instagram-cookies"
SOURCE = OUTPUT / "source-symbol.png"


def export_icons():
    source = Image.open(SOURCE).convert("RGBA")
    symbol = source.crop(source.getchannel("A").getbbox())
    report = {}
    for size in (16, 48, 128):
        padding = max(1, round(size / 16))
        scaled = symbol.copy()
        scaled.thumbnail((size - padding * 2, size - padding * 2), Image.Resampling.LANCZOS)
        icon = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        icon.alpha_composite(scaled, ((size - scaled.width) // 2, (size - scaled.height) // 2))
        icon.save(OUTPUT / f"icon{size}.png", optimize=True)
        alpha = icon.getchannel("A")
        bbox = alpha.getbbox()
        assert alpha.getextrema()[0] == 0
        assert bbox[0] >= padding and bbox[2] <= size - padding
        report[str(size)] = {"size": [size, size], "alpha": alpha.getextrema(), "bbox": bbox}

    manifest = json.loads((ROOT / "extension/manifest.json").read_text(encoding="utf-8"))
    for field in (manifest["icons"], manifest["action"]["default_icon"]):
        for size in (16, 48, 128):
            assert field[str(size)] == f"icons/icon{size}.png"
    QA.mkdir(parents=True, exist_ok=True)
    (QA / "icon-export.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    preview = Image.new("RGB", (680, 330), "white")
    draw = ImageDraw.Draw(preview)
    for x, background, foreground in ((0, "#f4f5f7", "#202124"), (340, "#202124", "#ffffff")):
        draw.rectangle((x, 0, x + 339, 330), fill=background)
        draw.text((x + 18, 12), "Brand V2 / toolbar: 16, 48, 128 px", fill=foreground)
        for size, y in ((16, 48), (48, 94), (128, 173)):
            icon = Image.open(OUTPUT / f"icon{size}.png")
            preview.paste(icon, (x + 32, y), icon)
            draw.text((x + 180, y + 4), f"{size} px", fill=foreground)
            if size == 16:
                enlarged = icon.resize((64, 64), Image.Resampling.NEAREST)
                preview.paste(enlarged, (x + 240, y), enlarged)
    preview.save(QA / "extension-icons-light-dark.png")
    print(json.dumps(report))


if __name__ == "__main__":
    export_icons()
