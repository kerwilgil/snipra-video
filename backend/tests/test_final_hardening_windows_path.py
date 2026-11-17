import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(os.name != "nt", reason="Windows Start-Process quoting probe")
def test_launcher_spawn_handles_spaces_parentheses_and_unicode(tmp_path):
    fixture = tmp_path / "Snipra 1.0 (edición)"
    fixture.mkdir()
    start = fixture / "start fixture.ps1"
    start.write_text("[IO.File]::WriteAllText((Join-Path $PSScriptRoot 'executed.txt'), 'ok')",
                     encoding="utf-8-sig")
    probe = tmp_path / "probe.ps1"
    # Execute the actual spawn statement from the launcher, selected via PS AST.
    probe.write_text(
        "param($Source, $StartScript, $Root)\n$ErrorActionPreference='Stop'\n"
        "$StdOutLog=Join-Path $Root 'out.log'\n$StdErrLog=Join-Path $Root 'err.log'\n"
        "$tokens=$null; $errors=$null\n"
        "$ast=[Management.Automation.Language.Parser]::ParseFile($Source,[ref]$tokens,[ref]$errors)\n"
        "if ($errors) { throw $errors[0] }\n"
        "$spawn=$ast.Find({param($n) $n -is [Management.Automation.Language.CommandAst] -and "
        "$n.GetCommandName() -eq 'Start-Process'}, $true)\n"
        "& ([scriptblock]::Create($spawn.Extent.Text))\n"
        "$deadline=(Get-Date).AddSeconds(10)\n"
        "while (-not (Test-Path (Join-Path $Root 'executed.txt'))) { "
        "if ((Get-Date) -gt $deadline) { throw 'Script never executed' }; "
        "Start-Sleep -Milliseconds 100 }\n", encoding="utf-8-sig",
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(probe),
         str(ROOT / "scripts/windows_launcher.ps1"), str(start), str(fixture)],
        capture_output=True, text=True, timeout=20, creationflags=subprocess.CREATE_NO_WINDOW,
    )
    assert result.returncode == 0, result.stderr
    assert (fixture / "executed.txt").read_text() == "ok"
