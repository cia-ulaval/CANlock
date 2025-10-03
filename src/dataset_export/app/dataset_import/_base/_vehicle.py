from uuid import UUID


class _Vehicle:
    @staticmethod
    def get_vehicle_id() -> UUID:
        raise NotImplementedError
