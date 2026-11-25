"""Persist Feishu jobs with task details that can be read directly in SQL."""

import pickle

from apscheduler.job import Job
from apscheduler.jobstores.base import ConflictingIdError
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.util import datetime_to_utc_timestamp
from sqlalchemy import Column, Engine, Text, bindparam, inspect, or_
from sqlalchemy.exc import IntegrityError

_TASK_FIELDS = ("user_id", "sender_name", "schedule_value", "prompt")


def _task_details(job: Job) -> dict[str, str]:
    return {field: job.kwargs[field] for field in _TASK_FIELDS}


class FeishuJobStore(SQLAlchemyJobStore):
    def __init__(self, engine: Engine) -> None:
        super().__init__(engine=engine, tablename="feishu_scheduled_jobs")
        for field in _TASK_FIELDS:
            self.jobs_t.append_column(Column(field, Text))

    def start(self, scheduler, alias) -> None:
        super().start(scheduler, alias)
        # TODO: Delete the column migration and backfill after existing deployments have upgraded.
        table_name = self.jobs_t.name
        with self.engine.begin() as connection:
            existing = {column["name"] for column in inspect(connection).get_columns(table_name)}
            for field in _TASK_FIELDS:
                if field not in existing:
                    connection.exec_driver_sql(f"ALTER TABLE {table_name} ADD COLUMN {field} TEXT")

        missing_details = or_(*(self.jobs_t.c[field].is_(None) for field in _TASK_FIELDS))
        jobs = self._get_jobs(missing_details)
        if jobs:
            update = self.jobs_t.update().where(self.jobs_t.c.id == bindparam("job_id"))
            with self.engine.begin() as connection:
                connection.execute(update, [{"job_id": job.id, **_task_details(job)} for job in jobs])

    def add_job(self, job: Job) -> None:
        # Insert the scheduler state and its readable details in one transaction.
        insert = self.jobs_t.insert().values(
            id=job.id,
            next_run_time=datetime_to_utc_timestamp(job.next_run_time),
            job_state=pickle.dumps(job.__getstate__(), self.pickle_protocol),
            **_task_details(job),
        )
        try:
            with self.engine.begin() as connection:
                connection.execute(insert)
        except IntegrityError as exc:
            raise ConflictingIdError(job.id) from exc
