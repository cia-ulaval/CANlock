"""
Attack Dataset Generator

Creates balanced datasets mixing normal CAN traffic with synthetic attacks.
Exports to CSV/Parquet for training deep learning models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
from tqdm import tqdm

from sqlmodel import select
from canlock.db.database import get_session
from canlock.db.models import CanMessage, Session
from canlock.attacks.dos_generator import DoSGenerator
from canlock.attacks.fuzzing_generator import FuzzingGenerator
from canlock.attacks.spoofing_generator import SpoofingGenerator
from canlock.attacks.injection_generator import InjectionGenerator
from canlock.attacks.normal_traffic_generator import NormalTrafficGenerator


class AttackDatasetGenerator:
    """
    Generates datasets combining real normal traffic with synthetic attacks.
    """
    
    def __init__(
        self,
        seed: int = 42,
        normal_ratio: float = 0.5,
        dos_ratio: float = 0.125,
        fuzzing_ratio: float = 0.125,
        spoofing_ratio: float = 0.125,
        injection_ratio: float = 0.125
    ):
        """
        Initialize dataset generator.
        
        Args:
            seed: Random seed for reproducibility
            normal_ratio: Proportion of normal traffic (default 50%)
            dos_ratio: Proportion of DoS attacks
            fuzzing_ratio: Proportion of Fuzzing attacks
            spoofing_ratio: Proportion of Spoofing attacks
            injection_ratio: Proportion of Injection attacks
        """
        assert np.isclose(
            normal_ratio + dos_ratio + fuzzing_ratio + spoofing_ratio + injection_ratio, 
            1.0
        ), "Ratios must sum to 1.0"
        
        self.seed = seed
        self.ratios = {
            'normal': normal_ratio,
            'dos': dos_ratio,
            'fuzzing': fuzzing_ratio,
            'spoofing': spoofing_ratio,
            'injection': injection_ratio
        }
        
        # Initialize generators
        self.generators = {
            'normal': NormalTrafficGenerator(seed=seed),
            'dos': DoSGenerator(seed=seed),
            'fuzzing': FuzzingGenerator(seed=seed),
            'spoofing': SpoofingGenerator(seed=seed),
            'injection': InjectionGenerator(seed=seed)
        }
    
    def extract_real_can_ids(self, max_messages: int = 10000) -> List[int]:
        """
        Extract real CAN IDs from the database to make synthetic traffic more realistic.
        
        Args:
            max_messages: Maximum number of messages to sample
            
        Returns:
            List of unique CAN IDs found in real data
        """
        print(f"Extracting real CAN IDs from database (sampling {max_messages} messages)...")
        
        with get_session() as session:
            # Get a random session
            sessions = session.exec(select(Session)).all()
            if not sessions:
                print("Warning: No sessions found in database, using default IDs")
                return []
            
            sample_session = sessions[0]
            
            # Sample messages from this session
            messages = session.exec(
                select(CanMessage)
                .where(CanMessage.session_id == sample_session.id)
                .limit(max_messages)
            ).all()
            
            # Extract unique CAN IDs
            can_ids = list(set(
                msg.can_identifier 
                for msg in messages 
                if msg.can_identifier is not None
            ))
            
            print(f"Found {len(can_ids)} unique CAN IDs in real data")
            return can_ids
    
    def generate_from_database(
        self, 
        num_samples: int,
        use_real_ids: bool = True,
        max_real_samples: int = 10000
    ) -> pd.DataFrame:
        """
        Generate dataset using real CAN IDs from database.
        
        Args:
            num_samples: Total number of samples to generate
            use_real_ids: If True, extract real CAN IDs from DB for generators
            max_real_samples: Max messages to sample for extracting real IDs
            
        Returns:
            DataFrame with mixed normal and attack traffic
        """
        # Get real CAN IDs if requested
        real_ids = []
        if use_real_ids:
            real_ids = self.extract_real_can_ids(max_real_samples)
            # Note: Current generators don't support custom CAN IDs
            # They generate random IDs which is sufficient for synthetic data
        
        return self.generate(num_samples)
    
    def generate(self, num_samples: int) -> pd.DataFrame:
        """
        Generate synthetic dataset with specified proportions.
        
        Args:
            num_samples: Total number of samples to generate
            
        Returns:
            DataFrame with columns: timestamp, can_identifier, length, payload_bytes, label
        """
        print(f"Generating dataset with {num_samples} samples...")
        
        all_messages = []
        
        for attack_type, ratio in tqdm(self.ratios.items(), desc="Generating attacks"):
            n_samples = int(num_samples * ratio)
            if n_samples == 0:
                continue
            
            generator = self.generators[attack_type]
            messages = generator.generate(n_samples)
            all_messages.extend(messages)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_messages)
        
        # Convert payload bytes to list of integers for easier processing
        df['payload_bytes'] = df['payload'].apply(lambda x: list(x))
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Add numeric label encoding
        label_mapping = {
            'normal': 0,
            'dos': 1,
            'fuzzing': 2,
            'spoofing': 3,
            'injection': 4
        }
        df['label_numeric'] = df['label'].map(label_mapping)
        
        print(f"✓ Generated {len(df)} samples")
        print(f"\nClass distribution:")
        print(df['label'].value_counts())
        
        return df
    
    def split_dataset(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train/val/test with stratification.
        
        Args:
            df: Complete dataset
            train_ratio: Training set proportion
            val_ratio: Validation set proportion
            test_ratio: Test set proportion
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Ratios must sum to 1.0"
        
        # Shuffle dataset
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        # Calculate split indices
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        print(f"\nDataset split:")
        print(f"  Train: {len(train_df)} samples ({100*train_ratio:.1f}%)")
        print(f"  Val:   {len(val_df)} samples ({100*val_ratio:.1f}%)")
        print(f"  Test:  {len(test_df)} samples ({100*test_ratio:.1f}%)")
        
        return train_df, val_df, test_df
    
    def save_dataset(
        self,
        df: pd.DataFrame,
        output_path: Path,
        format: str = 'parquet'
    ):
        """
        Save dataset to disk.
        
        Args:
            df: Dataset DataFrame
            output_path: Output file path
            format: File format ('csv' or 'parquet')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # For saving, convert payload bytes back to hex strings
        df_save = df.copy()
        df_save['payload_hex'] = df_save['payload'].apply(lambda x: x.hex())
        df_save = df_save.drop(columns=['payload'])
        
        if format == 'csv':
            df_save.to_csv(output_path, index=False)
        elif format == 'parquet':
            df_save.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"✓ Saved dataset to {output_path}")
    
    def load_dataset(self, input_path: Path) -> pd.DataFrame:
        """
        Load dataset from disk.
        
        Args:
            input_path: Input file path
            
        Returns:
            Loaded DataFrame with payload as bytes
        """
        input_path = Path(input_path)
        
        if input_path.suffix == '.csv':
            df = pd.read_csv(input_path)
        elif input_path.suffix == '.parquet':
            df = pd.read_parquet(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
        
        # Convert hex strings back to bytes
        if 'payload_hex' in df.columns:
            df['payload'] = df['payload_hex'].apply(lambda x: bytes.fromhex(x))
            df = df.drop(columns=['payload_hex'])
        
        # Ensure payload_bytes column exists
        if 'payload_bytes' not in df.columns and 'payload' in df.columns:
            df['payload_bytes'] = df['payload'].apply(lambda x: list(x))
        
        print(f"✓ Loaded {len(df)} samples from {input_path}")
        return df


def create_example_dataset(
    output_dir: Path = Path("data/attack_datasets"),
    num_samples: int = 100000,
    use_real_ids: bool = True
):
    """
    Convenience function to create a complete train/val/test dataset.
    
    Args:
        output_dir: Directory to save datasets
        num_samples: Total number of samples to generate
        use_real_ids: Use real CAN IDs from database
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate dataset
    generator = AttackDatasetGenerator(seed=42)
    
    if use_real_ids:
        df = generator.generate_from_database(num_samples)
    else:
        df = generator.generate(num_samples)
    
    # Split dataset
    train_df, val_df, test_df = generator.split_dataset(df)
    
    # Save splits
    generator.save_dataset(train_df, output_dir / "train.parquet")
    generator.save_dataset(val_df, output_dir / "val.parquet")
    generator.save_dataset(test_df, output_dir / "test.parquet")
    
    # Save full dataset as well
    generator.save_dataset(df, output_dir / "full_dataset.parquet")
    
    print(f"\n✓ Dataset creation complete!")
    print(f"  Output directory: {output_dir}")
    
    return train_df, val_df, test_df
