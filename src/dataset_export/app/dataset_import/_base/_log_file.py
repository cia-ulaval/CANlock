from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from mmap import ACCESS_READ, mmap
from pathlib import Path
from typing import Generator

from tqdm import tqdm

from dataset_export.app.settings import AppSettings


@dataclass
class _J1939LogLine:
    can_id: int
    dlc: int
    data: bytes
    timestamp: str

class _J1939LogFile:
    def __init__(self, file_path: Path):
        self._log = file_path

    @property
    def hash(self) -> str:
        raise NotImplementedError

    def __str__(self):
        return str(self._log)

    def __iter__(self) -> Generator[_J1939LogLine, None, None]:
        raise NotImplementedError


class _J1939LogFilesBatch:
    def __init__(self, j1939_log_files: list[Path]):
        self._log_files = j1939_log_files

    @property
    def start_time(self) -> datetime:
        raise NotImplementedError

    @property
    def end_time(self)-> datetime:
        raise NotImplementedError

    def __iter__(self) -> Generator[_J1939LogFile, None, None]:
        raise NotImplementedError
