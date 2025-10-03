from dataclasses import dataclass
from uuid import UUID

from sqlmodel import Session, select

from dataset_export.app.persistance.database import get_session
from dataset_export.app.persistance.models import Vehicle


class _RenaultT520:
    @dataclass
    class _RenaultTruck:
        model: str = "T520",
        license_plate: str = "Unknown",
        make: str = "Renault",
        year: int = 2021,

    @staticmethod
    def get_vehicle_id() -> UUID:
        with get_session() as session:
            db_vehicle = session.exec(
                select(Vehicle).where(
                    Vehicle.model == _RenaultT520._RenaultTruck.model,
                    Vehicle.make == _RenaultT520._RenaultTruck.make,
                    Vehicle.license_plate == _RenaultT520._RenaultTruck.license_plate,
                    Vehicle.year == _RenaultT520._RenaultTruck.year
                )
            ).first()
            return db_vehicle.id if db_vehicle else _RenaultT520._create_vehicle(session).id

    @staticmethod
    def _create_vehicle(session: Session) -> Vehicle:
        vehicle = Vehicle(
            model=_RenaultT520._RenaultTruck.model,
            license_plate=_RenaultT520._RenaultTruck.license_plate,
            make=_RenaultT520._RenaultTruck.make,
            year=_RenaultT520._RenaultTruck.year,
        )
        session.add(vehicle)
        session.commit()
        return vehicle

