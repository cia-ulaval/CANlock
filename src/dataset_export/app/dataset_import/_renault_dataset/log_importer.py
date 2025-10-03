from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator

from dataset_export.app.dataset_import._renault_dataset._log_file import (
    _RenaultLogFilesBatch,
    _Timestamp,
)
from dataset_export.app.dataset_import._renault_dataset._vehicle import _RenaultT520
from dataset_export.app.dataset_import._shared._vehicle_logger import _J1939LogImporter
from tqdm import tqdm

from dataset_export.app.logger.logger import Logger
from dataset_export.app.settings import AppSettings


class RenaultLogBatches:
    def __init__(self, folder: str):
        self._log_files = list(Path(folder).glob("*.csv"))

    def __iter__(self) -> Generator[_RenaultLogFilesBatch, None, None]:
        self._log_files.sort(key=lambda file: _Timestamp(file))
        current_batch = [self._log_files[0]]
        prev_timestamp = _Timestamp(self._log_files[0])

        for log_file in self._log_files[1:]:
            if _Timestamp(log_file) - prev_timestamp > timedelta(minutes=1):
                yield _RenaultLogFilesBatch(current_batch)
                current_batch = []
            current_batch.append(log_file)
            prev_timestamp = _Timestamp(log_file)
        if current_batch:
            _RenaultLogFilesBatch(current_batch)


class J1939LogsImporter:

    def __init__(self, settings: AppSettings):
        self._settings = settings
        self._j1939_logs_importer = _J1939LogImporter(_RenaultT520.get_vehicle_id(), settings)

    def import_folder(self, folder: str) -> None:
        logger = Logger("J1939 Log Importer")
        logger.setLevel(AppSettings().log_level)
        log_batches = list(RenaultLogBatches(folder))


        logger.info(f"Importing datasets from folder {folder} with {len(log_batches)} log batches")

        progress = tqdm(total=len(log_batches), position=0, desc="Log Batches", bar_format="{desc}: |{bar}| [{n}/{total}]")
        for log_batch in log_batches:
            self._j1939_logs_importer.import_log(log_batch)
            progress.update(1)

