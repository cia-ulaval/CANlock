from datetime import datetime
from uuid import UUID

from sqlmodel import select

from dataset_export.app.persistance.database import get_session
from dataset_export.app.persistance.models import CanMessage, LogFile, Session


class _Database:
    @staticmethod
    def get_session_id(start_time: datetime, end_time: datetime, vehicle_id : UUID) -> UUID:
        stmt = select(Session).where(
            Session.start_time == start_time,
            Session.end_time == end_time,
            Session.vehicle_id == vehicle_id
        )
        with get_session() as db:
            if session := db.exec(stmt).first():
               return session.id

        session = Session(
            vehicle_id=vehicle_id,
            start_time=start_time,
            end_time = end_time,
        )
        with get_session() as db:
            db.add(session)
            db.commit()
            return session.id

    @staticmethod
    def import_messages(messages: list[CanMessage], log_file: LogFile | None) -> None:
        with get_session() as db:
            if log_file:
                db.add(log_file)
            db.bulk_save_objects(messages)
            db.commit()

    @staticmethod
    def log_imported(log_file_hash: str) -> bool:
        stmt = select(LogFile).where(LogFile.hash == log_file_hash)
        with get_session() as db:
            return db.exec(stmt).first() is not None
