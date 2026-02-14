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


@click.command()
@click.option("--model", type=click.Choice(['cnn-lstm', 'lstm-cnn']), required=True, help="Model architecture to train")
@click.option("--dataset-dir", type=ClickPath(exists=True, path_type=Path), default="data/attack_datasets", help="Dataset directory")
@click.option("--epochs", type=int, default=50, help="Number of epochs")
@click.option("--batch-size", type=int, default=128, help="Batch size")
@click.option("--lr", type=float, default=0.001, help="Learning rate")
@click.option("--window-size", type=int, default=50, help="Window size")
@click.option("--checkpoint-dir", type=ClickPath(path_type=Path), default="checkpoints", help="Checkpoint directory")
def train_ids(model: str, dataset_dir: Path, epochs: int, batch_size: int, lr: float, window_size: int, checkpoint_dir: Path) -> None:
    """Train a CAN IDS model."""
    import torch
    import pandas as pd
    from canlock.models import CNNLSTM, LSTMCNN
    from canlock.training import create_dataloaders, IDSTrainer
    
    click.echo(f"Training {model.upper()} model...")
    
    # Load datasets
    train_df = pd.read_parquet(dataset_dir / "train.parquet")
    val_df = pd.read_parquet(dataset_dir / "val.parquet")
    test_df = pd.read_parquet(dataset_dir / "test.parquet")
    
    # Create dataloaders
    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        train_df, val_df, test_df, 
        window_size=window_size, 
        batch_size=batch_size,
        scaler_path=checkpoint_dir / "scaler.pkl"
    )
    
    # Create model
    input_dim = 12  # CAN_ID, DLC, Priority, TimeDelta + 8 payload bytes
    if model == 'cnn-lstm':
        ids_model = CNNLSTM(input_dim=input_dim, window_size=window_size)
    else:
        ids_model = LSTMCNN(input_dim=input_dim, window_size=window_size)
    
    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = IDSTrainer(
        model=ids_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=lr,
        checkpoint_dir=checkpoint_dir
    )
    
    # Train
    history = trainer.train(num_epochs=epochs)
    
    # Test
    metrics = trainer.test(test_loader)
    
    click.echo("\nTraining complete!")


@click.command()
@click.option("--output", type=ClickPath(path_type=Path), default="data/attack_datasets/full_dataset.parquet", help="Output path")
@click.option("--num-samples", type=int, default=100000, help="Number of samples to generate")
@click.option("--use-real-ids/--no-real-ids", default=True, help="Use real CAN IDs from database")
def generate_attacks(output: Path, num_samples: int, use_real_ids: bool) -> None:
    """Generate synthetic attack dataset."""
    from canlock.attacks import AttackDatasetGenerator
    
    click.echo(f"Generating {num_samples} attack samples...")
    
    generator = AttackDatasetGenerator(seed=42)
    
    if use_real_ids:
        df = generator.generate_from_database(num_samples)
    else:
        df = generator.generate(num_samples)
    
    # Split and save
    output_dir = output.parent
    train_df, val_df, test_df = generator.split_dataset(df)
    
    generator.save_dataset(train_df, output_dir / "train.parquet")
    generator.save_dataset(val_df, output_dir / "val.parquet")
    generator.save_dataset(test_df, output_dir / "test.parquet")
    generator.save_dataset(df, output)
    
    click.echo(f"\n✓ Dataset saved to {output_dir}")



if __name__ == "__main__":
    main()

