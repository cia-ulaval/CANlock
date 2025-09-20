import logging
import tarfile
import zipfile
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

def run(url: str, data_folder: Path) -> None:
    if data_folder.exists():
        logger.info("Clear previous zip archive")
        data_folder.unlink()
    
    logger.info("Downloading ...")
    try:
        reponse = requests.get(url, stream=True)
    except requests.exceptions.RequestException as e:
        raise e
    
    logger.info("Start saving files ...")
    with open(data_folder, 'wb') as f:
        for i, chunk in enumerate(reponse.iter_content(chunk_size=8192)):
            logger.debug("Writing chunk {i}")
            f.write(chunk)
    
    zip_fname = data_folder.name
    destination_folder = data_folder.parent.joinpath(zip_fname.replace(".zip", ""))
    logger.info("Unzipping downloaded archive")
    try:
        with zipfile.ZipFile(data_folder, mode="r") as zf:
            zf.extractall(destination_folder)
    except Exception as e:
        raise e
    
    logger.info(f"Archive has been decompressed into {str(destination_folder)}")
    
    logger.info("Removing original downloaded archive")
    data_folder.unlink()
    
    logger.info("Decompressing the 4 data parts")
    for i, p in enumerate(destination_folder.iterdir()):
        logger.info(f"Part {i + 1}")
        tar_fname = p.name
        destination_folder = p.parent.joinpath(tar_fname.replace(".tar.xz", ""))
        
        try:
            with tarfile.open(p, mode="r:xz") as tar:
                tar.extractall(path=destination_folder)
        except Exception as e:
            raise e

        logger.info(f"Remove part {i + 1} original archive folder")
        p.unlink()
