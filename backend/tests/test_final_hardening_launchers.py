import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows process ownership probe")


def powershell(script, *arguments):
    return subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
         str(script), *map(str, arguments)], capture_output=True, text=True, timeout=40,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def test_foreign_listener_survives_start_and_stop(tmp_path):
    server = subprocess.Popen(
        [sys.executable, "-u", "-c",
         "import socket,time; s=socket.socket(); s.bind(('127.0.0.1',0)); "
         "s.listen(); print(s.getsockname()[1],flush=True); time.sleep(120)"],
        stdout=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW,
    )
    try:
        port = int(server.stdout.readline())
        for service in ("BackendPort", "FrontendPort"):
            result = powershell(ROOT / "scripts/start-snipra.ps1", f"-{service}", port, "-NoOpen")
            assert result.returncode != 0
            assert "ocupado por otro proceso" in result.stderr
            assert server.poll() is None
        result = powershell(ROOT / "scripts/stop-snipra.ps1", "-BackendPort", port,
                            "-FrontendPort", port + 1)
        assert result.returncode == 0
        assert server.poll() is None
    finally:
        server.terminate()
        server.wait(timeout=10)


def test_free_port_and_stale_snipra_listener(tmp_path):
    probe = tmp_path / "restart.ps1"
    probe.write_text(
        "param($Root, [int]$Port)\n$ErrorActionPreference='Stop'\n"
        ". (Join-Path $Root 'scripts/process-ownership.ps1')\n"
        "Restart-SnipraPort $Port $Root 'backend'\n", encoding="utf-8",
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    assert powershell(probe, ROOT, port).returncode == 0
    env = {**os.environ, "STORAGE_DIR": str(tmp_path / "storage"),
           "DATABASE_URL": f"sqlite:///{(tmp_path / 'probe.db').as_posix()}"}
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port)],
        cwd=ROOT / "backend", env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    try:
        deadline = time.monotonic() + 20
        while True:
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    break
            assert server.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.1)
        result = powershell(probe, ROOT, port)
        assert result.returncode == 0, result.stderr
        server.wait(timeout=10)
    finally:
        if server.poll() is None:
            server.terminate()
            server.wait(timeout=10)


def test_frontend_npm_shim_path_is_owned_and_recoverable(tmp_path):
    root = tmp_path / "Snipra frontend (fixture)"
    script = root / "frontend/node_modules/vite/bin/vite.js"
    script.parent.mkdir(parents=True)
    shim = root / "frontend/node_modules/.bin"
    shim.mkdir()
    script.write_text(
        "const s=require('node:net').createServer(); "
        "s.listen(0,'127.0.0.1',()=>console.log(s.address().port))"
    )
    probe = tmp_path / "frontend.ps1"
    probe.write_text(
        "param($Helper, $Root, [int]$Port)\n$ErrorActionPreference='Stop'\n"
        ". $Helper\nRestart-SnipraPort $Port $Root 'frontend'\n", encoding="utf-8-sig",
    )
    server = subprocess.Popen(["node", str(shim / "../vite/bin/vite.js")],
                              stdout=subprocess.PIPE, text=True,
                              creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        port = int(server.stdout.readline())
        result = powershell(probe, ROOT / "scripts/process-ownership.ps1", root, port)
        assert result.returncode == 0, result.stderr
        server.wait(timeout=10)
    finally:
        if server.poll() is None:
            server.terminate()
            server.wait(timeout=10)
