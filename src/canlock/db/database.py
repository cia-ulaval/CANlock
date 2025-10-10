from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel

from canlock.db._settings import _Settings

_settings = _Settings()
_sql_url = (
    f"postgresql://{_settings.POSTGRES_USER}:{_settings.POSTGRES_PASSWORD}@"
    f"{_settings.POSTGRES_HOSTNAME}:"
    f"{_settings.POSTGRES_PORT}/{_settings.POSTGRES_DB}"
)
_engine = create_engine(_sql_url)

def init_db() -> None:
    SQLModel.metadata.create_all(_engine)

@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations."""
    with Session(_engine) as session:
        yield session
