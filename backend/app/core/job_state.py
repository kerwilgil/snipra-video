"""Estado y logging compartidos entre la API y el worker."""

from sqlmodel import Session

from app.models import JobLog, JobStatus

ACTIVE_JOB_STATUSES = {
    JobStatus.queued,
    JobStatus.ingesting,
    JobStatus.transcribing,
    JobStatus.detecting,
    JobStatus.rendering,
}


def add_job_log(
    session: Session,
    job_id: str,
    message: str,
    level: str = "info",
) -> None:
    session.add(JobLog(job_id=job_id, level=level, message=message[:1000]))
