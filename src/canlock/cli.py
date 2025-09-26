import logging
from pathlib import Path

import click
from click import Path as ClickPath

from canlock.data.download_heavy_duty_truck_data import run as heavy_truck_data_run
from canlock.data.csv_transform_data import transform_all_csv_files_multiprocessing_with_progress

logging.basicConfig(
    level=logging.INFO,
    force=True,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler("cli.log", "a", "utf-8"), logging.StreamHandler()],
)
logging.info("Recording")


@click.command()
@click.option("-n", "--name", default="you", type=str, help="Your name")
def main(name: str) -> None:
    print(f"Hello {name}, from canlock!")


@click.command()
@click.option(
    "-u",
    "--url",
    type=str,
    default=None,
    required=False,
    help="Download url from etsin.fairdate.fi website",
)
@click.option(
    "-d",
    "--data-folder",
    type=ClickPath(dir_okay=True, file_okay=False, path_type=Path),
    default="data/heavy_truck_data.zip",
    required=False,
    help="Folder to save data in",
)
def download_heavy_truck_data(url: str, data_folder: Path) -> None:
    if url is None:
        raise ValueError(
            "Rends toi sur le site 'https://etsin.fairdata.fi/dataset/7586f24f-c91b-41df-92af-283524de8b3e/data', clique sur les 3 points à côté du bouton 'Download all', copie/colle le lien donné dans URL"
        )
    heavy_truck_data_run(url, data_folder)

@click.command()
@click.option(
    "-d",
    "--data-folder",
    type=ClickPath(dir_okay=True, file_okay=False, path_type=Path),
    default="data/heavy_truck_data/",
    required=False,
    help="Folder to fetch data",
)
@click.option(
    "-o",
    "--output-folder",
    type=ClickPath(dir_okay=True, file_okay=False, path_type=Path),
    default="data/heavy_truck_data_transformed/",
    required=False,
    help="Folder to save data in",
)
@click.option(
    "-p",
    "--num-processes",
    type=int,
    default=1,
    required=False,
    help="Number of processes",
)
@click.option(
    "-b",
    "--batch-size",
    type=int,
    default=100,
    required=False,
    help="Batch size",
)
def preprocess_heavy_truck_data(data_folder: Path, output_folder: Path, num_processes: int, batch_size: int) -> None:
    transform_all_csv_files_multiprocessing_with_progress(
        data_directory=data_folder, 
        output_directory=output_folder,
        num_processes=num_processes,
        batch_size=batch_size,
    )

if __name__ == "__main__":
    main()
