from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from mmap import ACCESS_READ, mmap
from pathlib import Path
from typing import Generator

from dataset_export.app.dataset_import._base._log_file import (
    _J1939LogFile,
    _J1939LogFilesBatch,
    _J1939LogLine,
)
from tqdm import tqdm

from dataset_export.app.settings import AppSettings


class _Timestamp(datetime):
    def __new__(cls, file_name: Path):
        return datetime.strptime(file_name.stem[:14], "%Y%m%d%H%M%S")


class _RenaultLogFile(_J1939LogFile):
    class _PartsIndexes:
        CAN_ID: int = 1
        DLC: int = 2
        DATA: int = 3
        TIMESTAMP: int = 0

    @property
    def hash(self) -> str:
        file_hash = sha256()
        with self._log.open("rb") as f:
            with mmap(f.fileno(), 0, access=ACCESS_READ) as mm:
                file_hash.update(mm)
        return file_hash.hexdigest()

    @property
    def timestamp(self) -> datetime:
        return _Timestamp(self._log)

    def __iter__(self) -> Generator[_J1939LogLine, None, None]:
        with open(self._log, "rb") as f:
            line_count = sum(buf.count(b"\n") for buf in iter(lambda: f.read(1024 * 1024), b"")) - 1

        progress_bar = tqdm(total=line_count, position=2, leave=False, desc="Log Lines", bar_format="{desc}: |{bar}| [{n}/{total}]")
        with open(self._log, "r") as log_file:
            log_file.readline()
            for log_line in log_file:
                parts = log_line.strip().split(";")
                progress_bar.update(1)
                yield _J1939LogLine(
                    can_id=int(parts[self._PartsIndexes.CAN_ID], 16),
                    dlc=self._format_dlc(int(parts[self._PartsIndexes.DLC])),
                    data=bytes(int(x) for x in parts[self._PartsIndexes.DATA:]),
                    timestamp=parts[self._PartsIndexes.TIMESTAMP],
                )

    @staticmethod
    def _format_dlc(dlc: int) -> int:
        return 1 if dlc == 255 else dlc


class _RenaultLogFilesBatch(_J1939LogFilesBatch):
    @property
    def start_time(self) -> datetime:
        return _Timestamp(self._log_files[0])

    @property
    def end_time(self)-> datetime:
        return _Timestamp(self._log_files[-1])

    def __iter__(self) -> Generator[_RenaultLogFile, None, None]:
        for log_file in self._log_files:
            yield _RenaultLogFile(log_file)
