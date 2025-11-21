import logging
from pathlib import Path

import click
from click import Path as ClickPath

from canlock.data.download_heavy_duty_truck_data import run as heavy_truck_data_run

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
@click.option("--session-id", type=str, required=True, help="Session ID to analyze")
def analyze_session(session_id: str) -> None:
    from uuid import UUID

    from canlock.db.database import get_session
    from canlock.decoder import SessionDecoder

    try:
        sess_uuid = UUID(session_id)
    except ValueError:
        click.echo("Invalid UUID format.")
        return

    with get_session() as session:
        decoder = SessionDecoder(session)
        df = decoder.decode(sess_uuid)
        
        if df.empty:
            click.echo("No data decoded for this session.")
        else:
            click.echo(f"Decoded {len(df)} rows.")
            click.echo(df.head())



if __name__ == "__main__":
    main()

