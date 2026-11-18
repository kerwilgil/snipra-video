"""Pruebas de rechazo seguro del packaging macOS, sin Apple Developer ID ni Mac real.

Los binarios de Apple se sustituyen por dobles que simulan resultados. Esto verifica
la lógica del gate, NO constituye una prueba de firma/notarización física.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
NOTARIZE = ROOT / "scripts/macos/notarize.sh"
BUILD = ROOT / "scripts/macos/build-macos-app.sh"


class MacOSDistributionGatesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.bin_dir = self.work / "bin"
        self.bin_dir.mkdir()
        self.app = self.work / "Snipra.app"
        self.app.mkdir()

        self.stub("codesign", """#!/bin/sh
echo 'Authority=Developer ID Application: QA (TESTTEAM)' >&2
echo 'flags=0x10000(runtime)' >&2
""")
        self.stub("ditto", "#!/bin/sh\nexit 0\n")
        self.stub("xcrun", """#!/bin/sh
case "$1 $2" in
  "notarytool submit")
    if [ "${NOTARY_SUBMIT_FAIL:-0}" = "1" ]; then exit 10; fi
    echo '{"status":"Accepted","id":"test-submission"}'
    exit 0 ;;
  "stapler staple"|"stapler validate") exit 0 ;;
  *) exit 20 ;;
esac
""")
        self.stub("spctl", """#!/bin/sh
[ "${SPCTL_FAIL:-0}" != "1" ]
""")
        if os.name == "nt":
            self.stub("python3", f'#!/bin/sh\nexec "{Path(sys.executable).as_posix()}" "$@"\n')

    def stub(self, name: str, source: str) -> None:
        path = self.bin_dir / name
        path.write_text(source, encoding="utf-8", newline="\n")
        path.chmod(0o755)

    def run_script(self, path: Path, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update({"PATH": str(self.bin_dir) + os.pathsep + env["PATH"]})
        if extra_env:
            env.update(extra_env)
        arguments = ["bash", str(path)]
        if path == NOTARIZE:
            arguments.append(str(self.app))
        if os.name == "nt":
            # WindowsApps/bash can be WSL, which cannot open Windows paths.
            # Use Git Bash for these shell-only stub tests, never Apple tooling.
            bash = Path(os.environ.get("SNIPRA_TEST_BASH", ""))
            if not bash.is_file():
                bash = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
            if not bash.is_file():
                self.skipTest("Git Bash is required for shell gate tests on Windows")
            env["PATH"] = os.environ["PATH"]
            arguments = [str(bash), "-c",
                         'export PATH="$(cygpath -u "$1"):$PATH"; exec bash "$2" "${@:3}"',
                         "snipra-gate-test", self.bin_dir.as_posix(), path.as_posix()]
            if path == NOTARIZE:
                arguments.append(self.app.as_posix())
        return subprocess.run(arguments, capture_output=True, text=True, env=env, timeout=30)

    def test_notarization_accepted_and_gatekeeper_accepted(self) -> None:
        proc = self.run_script(NOTARIZE, {"SNIPRA_NOTARY_PROFILE": "TEST", "SPCTL_FAIL": "0"})
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("NOTARIZATION_ACCEPTED", proc.stdout)

    def test_gatekeeper_rejection_is_fatal(self) -> None:
        proc = self.run_script(NOTARIZE, {"SNIPRA_NOTARY_PROFILE": "TEST", "SPCTL_FAIL": "1"})
        self.assertEqual(proc.returncode, 6, proc.stdout + proc.stderr)
        self.assertIn("GATEKEEPER_REJECTED", proc.stderr)
        self.assertNotIn("NOTARIZATION_ACCEPTED", proc.stdout)

    def test_submit_failure_is_fatal(self) -> None:
        proc = self.run_script(NOTARIZE, {"SNIPRA_NOTARY_PROFILE": "TEST", "NOTARY_SUBMIT_FAIL": "1"})
        self.assertEqual(proc.returncode, 5, proc.stdout + proc.stderr)
        self.assertIn("NOTARIZATION_SUBMIT_FAILED", proc.stderr)

    def test_missing_profile_never_submits(self) -> None:
        proc = self.run_script(NOTARIZE, {"SNIPRA_NOTARY_PROFILE": ""})
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("NOTARIZATION_PENDING", proc.stdout)

    def test_distribution_mode_requires_developer_id(self) -> None:
        proc = self.run_script(BUILD, {"SNIPRA_MACOS_BUILD_MODE": "distribution",
                                       "SNIPRA_CODESIGN_IDENTITY": "-"})
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("requiere SNIPRA_CODESIGN_IDENTITY", proc.stderr)

    def test_distribution_mode_requires_notary_profile(self) -> None:
        proc = self.run_script(BUILD, {"SNIPRA_MACOS_BUILD_MODE": "distribution",
                                       "SNIPRA_CODESIGN_IDENTITY": "Developer ID Application: QA (TESTTEAM)",
                                       "SNIPRA_NOTARY_PROFILE": ""})
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("falta SNIPRA_NOTARY_PROFILE", proc.stderr)

    def test_unknown_build_mode_is_rejected(self) -> None:
        proc = self.run_script(BUILD, {"SNIPRA_MACOS_BUILD_MODE": "insecure"})
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("SNIPRA_MACOS_BUILD_MODE debe ser", proc.stderr)

    def test_public_adhoc_rejects_developer_id_identity(self) -> None:
        proc = self.run_script(BUILD, {"SNIPRA_MACOS_BUILD_MODE": "public-adhoc",
                                       "SNIPRA_CODESIGN_IDENTITY": "Developer ID Application: QA (TESTTEAM)"})
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("public-adhoc exige firma ad hoc", proc.stderr)

    def test_public_adhoc_requires_verified_bottles(self) -> None:
        proc = self.run_script(BUILD, {"SNIPRA_MACOS_BUILD_MODE": "public-adhoc",
                                       "SNIPRA_FFMPEG_SOURCE": "local"})
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("bottles verificados", proc.stderr)

    def test_public_adhoc_never_relaxes_official_distribution_mode(self) -> None:
        text = BUILD.read_text(encoding="utf-8")
        self.assertIn('NOTARY_STATUS="NOT_NOTARIZED"', text)
        self.assertIn('"$BUILD_MODE" == "distribution" && "$NOTARY_STATUS" != "ACCEPTED"', text)


if __name__ == "__main__":
    unittest.main()
