import datetime
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import BigInteger
from sqlmodel import Field, Relationship, SQLModel, UniqueConstraint
from sqlalchemy import Column


class VehicleBase(SQLModel):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    license_plate: str
    model: str
    make: str
    year: int


class Vehicle(VehicleBase, table=True):
    """Represents a Vehicle."""
    sessions: list["Session"] = Relationship(back_populates="vehicle")

    __table_args__ = (
        UniqueConstraint(
            "license_plate",
            "make",
            "year",
            name="uq_vehicle_fields",
        ),
    )


class LogFile(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    hash: str
    session_id: UUID | None = Field(foreign_key="session.id")
    session: Optional["Session"] = Relationship(back_populates="log_files")

class SessionBase(SQLModel):
    """Represents a Session."""

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    start_time: datetime.datetime
    end_time: datetime.datetime | None
    vehicle_id: UUID = Field(foreign_key="vehicle.id")


class Session(SessionBase, table=True):
    """Represents a Session."""

    vehicle: Optional["Vehicle"] = Relationship(back_populates="sessions")
    log_files: list["LogFile"] = Relationship(back_populates="session", cascade_delete=True)
    can_messages: list["CanMessage"] = Relationship(back_populates="session", cascade_delete=True)


class CanMessageBase(SQLModel):
    timestamp: datetime.datetime
    can_identifier: int | None

    length: int | None
    session_id: UUID | None = Field(foreign_key="session.id")


class CanMessage(CanMessageBase, table=True):
    """Represents a CAN Message."""

    id: int | None = Field(default=None, primary_key=True)
    payload: bytes | None
    session: Optional["Session"] = Relationship(back_populates="can_messages")
