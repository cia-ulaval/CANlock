from uuid import UUID

from tqdm import tqdm

from dataset_export.app.dataset_import._base._log_file import (
    _J1939LogFile,
    _J1939LogFilesBatch,
)
from dataset_export.app.dataset_import._shared._database import _Database
from dataset_export.app.logger.logger import Logger
from dataset_export.app.persistance.models import CanMessage, LogFile
from dataset_export.app.settings import AppSettings


class _J1939LogImporter:
    def __init__(self, vehicle_id: UUID, settings: AppSettings):
        self._vehicle_id = vehicle_id
        self._settings = settings

    def import_log(self, logs_batch: _J1939LogFilesBatch) -> None:
        logger = Logger("J1939 Log Importer")
        logger.setLevel(self._settings.log_level)

        log_session_id = _Database.get_session_id(logs_batch.start_time, logs_batch.end_time, self._vehicle_id)

        logs_batch = list(logs_batch)
        progress = tqdm(total=len(logs_batch), position=1, leave=False, desc="Log Files", bar_format="{desc}: |{bar}| [{n}/{total}]")
        for log_file in logs_batch:
            if _Database.log_imported(log_file.hash):
                logger.debug(f"Skipping J1939 log file {str(log_file)} as it was already imported")
                continue
            can_messages = self._convert_log_messages_to_model(log_file, log_session_id)
            _Database.import_messages(can_messages, LogFile(hash=log_file.hash, session_id=log_session_id))
            progress.update(1)


    @staticmethod
    def _convert_log_messages_to_model(log_file: _J1939LogFile, session_id: UUID) -> list[CanMessage]:
        return [
            CanMessage(
                timestamp=log_line.timestamp,
                can_identifier=log_line.can_id,
                session_id = session_id,
                length=log_line.dlc,
                payload=log_line.data
            ) for log_line in log_file
        ]
