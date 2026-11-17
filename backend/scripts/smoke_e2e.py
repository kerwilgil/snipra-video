"""Smoke E2E del worker: genera un video, crea un job y corre el pipeline completo.

Uso:  uv run python scripts/smoke_e2e.py [ruta_video_opcional]

Si no se pasa video, genera uno sintético (sin voz → 0 clips, pero valida la cadena).
Con un video real con voz, deberías ver clips generados.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

# asegura que 'app' sea importable al ejecutar el script directamente
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# storage aislado en temp
_TMP = Path(tempfile.gettempdir()) / "mediashorts_smoke"
_TMP.mkdir(parents=True, exist_ok=True)
os.environ["STORAGE_DIR"] = str(_TMP)
os.environ["DATABASE_URL"] = f"sqlite:///{(_TMP / 'smoke.db').as_posix()}"

from sqlmodel import Session, select  # noqa: E402

from app.core.db import engine, init_db  # noqa: E402
from app.models import Clip, Job, JobStatus, SourceType  # noqa: E402
from app.pipeline.ingest import job_dir  # noqa: E402
from app.worker.runner import process_job  # noqa: E402


def make_sample(dest: Path) -> Path:
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30:duration=15",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=15",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(dest),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return dest


def main() -> None:
    init_db()
    src_arg = sys.argv[1] if len(sys.argv) > 1 else None

    with Session(engine) as s:
        job = Job(
            source_type=SourceType.upload,
            source_name="smoke.mp4",
            title="Smoke E2E",
            status=JobStatus.queued,
            num_clips=3,
            formats="9x16",
            provider="naive",
        )
        s.add(job)
        s.commit()
        s.refresh(job)
        jid = job.id

    dest = job_dir(jid) / "source.mp4"
    if src_arg:
        import shutil

        shutil.copy(src_arg, dest)
        print(f"[smoke] usando video real: {src_arg}")
    else:
        make_sample(dest)
        print("[smoke] usando video sintético (sin voz)")

    print(f"[smoke] procesando job {jid} …")
    process_job(jid)

    with Session(engine) as s:
        job = s.get(Job, jid)
        print(f"[smoke] status={job.status} stage={job.stage!r} progress={job.progress}")
        if job.error:
            print(f"[smoke] error={job.error}")
        clips = s.exec(select(Clip).where(Clip.job_id == jid)).all()
        print(f"[smoke] clips generados: {len(clips)}")

    assert job.status == JobStatus.done, "El job no terminó en estado 'done'"
    print("[smoke] OK")


if __name__ == "__main__":
    main()
