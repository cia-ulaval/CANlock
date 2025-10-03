from dataset_export.app.dataset_import._renault_dataset.log_importer import (
    J1939LogsImporter,
)

from dataset_export.app.logger.logger import Logger
from dataset_export.app.persistance.database import init_db
from dataset_export.app.settings import AppSettings


def run() -> None:
    settings = AppSettings()

    logger = Logger("Dataset Import Logger")
    logger.setLevel(settings.log_level)

    logger.info("Starting Import")
    logger.info(f"Importing datasets from folders: {settings.folders}")

    init_db()
    log_importer = J1939LogsImporter(settings)
    for folder in settings.folders:
        log_importer.import_folder(folder)

if __name__ == "__main__":
    run()
