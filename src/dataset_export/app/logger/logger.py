import logging
import sys

from dataset_export.app.logger.local import LocalFormatter
from dataset_export.app.logger.type import LogType


class Logger(logging.Logger):
    """Logger.

    Examples:
        >>> from app.shared.tools.logger.logger import Logger
        >>>
        >>>
        >>> logger = Logger(__name__)
        >>> logger.info("Logger")

    """

    def __init__(
        self,
        name: str,
        log_type: LogType = LogType.LOCAL,
    ) -> None:
        """Initialize local logger formatter.

        Args:
            name (str): Logger name
            log_type (LogType, optional): Local or something.
                                          Defaults to LogType.LOCAL.

        """
        super().__init__(name=name)

        if log_type == LogType.LOCAL:
            formatter = LocalFormatter()
            handler = logging.StreamHandler(stream=sys.stdout)

            handler.setFormatter(formatter)
            self.addHandler(handler)
            return
