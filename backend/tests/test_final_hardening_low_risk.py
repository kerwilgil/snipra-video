import logging
import os
import subprocess
from pathlib import Path

import pytest

from app.core.config import settings
from app.pipeline import detect
from app.providers import vision
from app.providers.base import ClipCandidate
from app.worker import runner


@pytest.mark.parametrize("consumer", ["detect", "shadow", "rerank"])
def test_provider_error_logs_redact_configured_sentinel(monkeypatch, caplog, tmp_path, fake_segments, consumer):
    sentinel = "snipra-hardening-fake-secret-unique"
    monkeypatch.setattr(settings, "openai_api_key", sentinel)
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    def fail(*args, **kwargs):
        raise RuntimeError(f"fixture SDK error {sentinel}")
    candidate = ClipCandidate(start=0, end=10, title="fixture", score=0.73)
    caplog.set_level(logging.WARNING)
    if consumer == "detect":
        class Detector:
            detect = staticmethod(fail)
        monkeypatch.setattr(detect, "get_detector", lambda *a: Detector())
        detect.detect_clips(fake_segments, provider="openai", model="", num_clips=1, formats=["16x9"])
    elif consumer == "shadow":
        monkeypatch.setattr(settings, "enable_structured_visual_shadow", True)
        monkeypatch.setattr(vision, "_score_structured_shadow", fail)
        assert vision._append_structured_shadow(candidate, [b"frame"], "openai", "fixture") is candidate
    else:
        monkeypatch.setattr(vision, "_extract_frames", lambda *a, **k: [b"frame"])
        monkeypatch.setattr(vision, "_score_openai", fail)
        assert vision.rerank_by_vision([candidate], "fixture", provider="openai", model="fixture") == [candidate]
    assert "fixture SDK error" in caplog.text
    assert sentinel not in caplog.text


def test_empty_lock_during_exclusive_creation_cannot_be_stolen(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "WORKER_LOCK_PATH", tmp_path / "worker.lock")
    original_open = os.open
    second = []
    def open_during_acquire(path, flags, *args, **kwargs):
        fd = original_open(path, flags, *args, **kwargs)
        if path == runner.WORKER_LOCK_PATH:
            second.append(runner.acquire_worker_lock())
        return fd
    monkeypatch.setattr(os, "open", open_during_acquire)
    assert runner.acquire_worker_lock()
    assert second == [False]
    runner.release_worker_lock()
    runner.WORKER_LOCK_PATH.write_text("")
    old = __import__("time").time() - runner.WORKER_LOCK_STALE_SECONDS - 1
    os.utime(runner.WORKER_LOCK_PATH, (old, old))
    assert runner.acquire_worker_lock()
    runner.release_worker_lock()


@pytest.mark.skipif(os.name != "nt", reason="Windows release copy function probe")
def test_release_copy_excludes_private_runtime_and_untracked_files(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    names = [".env", ".env.local", ".env.example", "cookies.txt", "private.mp4", "scratch.log",
             "runtime.db", "untracked.txt", "runtime.py", "icon.png", "logs/nested.txt",
             "node_modules/package/index.js", "storage/local.json", "artifacts/qa/probe.txt"]
    for name in names:
        file = root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("fixture")
    script = tmp_path / "copy.ps1"
    script.write_text(
        "param($Source, $Fixture, $Destination)\n$ErrorActionPreference='Stop'\n"
        "$rootPath=[IO.Path]::GetFullPath($Fixture).TrimEnd('\\')+'\\'\n"
        "$ReleaseFiles=New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::Ordinal)\n"
        "Get-ChildItem $Fixture -Recurse -Force -File | ForEach-Object { "
        "$name=$_.FullName.Substring($rootPath.Length).Replace('\\','/'); "
        "if ($name -ne 'untracked.txt') { [void]$ReleaseFiles.Add($name) } }\n"
        "$t=$null; $e=$null\n"
        "$ast=[Management.Automation.Language.Parser]::ParseFile($Source,[ref]$t,[ref]$e)\n"
        "if ($e) { throw $e[0] }\n"
        "$fn=$ast.Find({param($n) $n -is [Management.Automation.Language.FunctionDefinitionAst] "
        "-and $n.Name -eq 'Copy-ReleaseItem'}, $true)\n"
        ". ([scriptblock]::Create($fn.Extent.Text))\n"
        "Copy-ReleaseItem $Fixture $Destination\n", encoding="utf-8-sig",
    )
    source = Path(__file__).resolve().parents[2] / "scripts/package-snipra-release.ps1"
    output = tmp_path / "output"
    result = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), str(source),
                             str(root), str(output)], capture_output=True, text=True, timeout=20,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode == 0, result.stderr
    assert {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()} == {
        ".env.example", "runtime.py", "icon.png",
    }
