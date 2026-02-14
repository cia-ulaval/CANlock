"""
Utilitaires pour Datasets d'Attaques Basés sur SPN

Fournit des fonctions pour extraire les statistiques SPN depuis la base de données
et générer des datasets de séries temporelles pour l'entraînement de modèles IDS.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm

from sqlmodel import select, func
from canlock.db.database import get_session
from canlock.db.models import CanMessage, SpnDefinition, Session
from canlock.decoder import SessionDecoder


def extract_spn_statistics(
    spn_id: int,
    session_id: Optional[str] = None,
    max_samples: int = 10000
) -> Dict[str, Any]:
    """
    Extrait les statistiques d'un SPN depuis la base de données.
    
    Args:
        spn_id: ID du SPN à analyser
        session_id: UUID de la session (si None, utilise la première disponible)
        max_samples: Nombre maximum d'échantillons à extraire
        
    Returns:
        Dictionary contenant:
        - 'spn_id': ID du SPN
        - 'spn_name': Nom du SPN
        - 'min': Valeur minimale observée
        - 'max': Valeur maximale observée
        - 'mean': Moyenne
        - 'std': Écart-type
        - 'count': Nombre de valeurs
        - 'sample_values': Échantillon de valeurs (np.ndarray)
    """
    print(f"Extracting statistics for SPN {spn_id}...")
    
    with get_session() as db_session:
        # Get SPN definition
        spn_def = db_session.exec(
            select(SpnDefinition).where(SpnDefinition.spn == spn_id)
        ).first()
        
        if not spn_def:
            raise ValueError(f"SPN {spn_id} not found in database")
        
        # Get session
        if session_id:
            from uuid import UUID
            sess_uuid = UUID(session_id)
            session = db_session.exec(
                select(Session).where(Session.id == sess_uuid)
            ).first()
        else:
            session = db_session.exec(select(Session)).first()
        
        if not session:
            raise ValueError("No session found in database")
        
        print(f"  Using session: {session.id}")
        print(f"  SPN name: {spn_def.spn_name}")
        
        # Decode session to get SPN values
        decoder = SessionDecoder(db_session)
        df = decoder.decode(session.id)
        
        if df.empty:
            raise ValueError(f"No data decoded for session {session.id}")
        
        # Filter for specific SPN
        spn_col = f'spn_{spn_id}'
        if spn_col not in df.columns:
            raise ValueError(f"SPN {spn_id} not found in decoded data")
        
        # Extract values (remove NaN)
        values = df[spn_col].dropna().values
        
        if len(values) == 0:
            raise ValueError(f"No valid values found for SPN {spn_id}")
        
        # Sample if too many values
        if len(values) > max_samples:
            sample_indices = np.random.choice(len(values), max_samples, replace=False)
            sample_values = values[sample_indices]
        else:
            sample_values = values
        
        # Calculate statistics
        stats = {
            'spn_id': spn_id,
            'spn_name': spn_def.spn_name,
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'count': len(values),
            'sample_values': sample_values
        }
        
        print(f"  Statistics:")
        print(f"    Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
        print(f"    Mean: {stats['mean']:.2f}")
        print(f"    Std: {stats['std']:.2f}")
        print(f"    Count: {stats['count']}")
        
        return stats


def list_available_spns(
    session_id: Optional[str] = None,
    min_samples: int = 100
) -> pd.DataFrame:
    """
    Liste tous les SPNs disponibles dans une session avec leurs statistiques.
    
    Args:
        session_id: UUID de la session (si None, utilise la première)
        min_samples: Nombre minimum de samples requis pour inclure un SPN
        
    Returns:
        DataFrame avec colonnes: spn_id, spn_name, count, min, max, mean, std
    """
    print("Listing available SPNs...")
    
    with get_session() as db_session:
        # Get session
        if session_id:
            from uuid import UUID
            sess_uuid = UUID(session_id)
            session = db_session.exec(
                select(Session).where(Session.id == sess_uuid)
            ).first()
        else:
            session = db_session.exec(select(Session)).first()
        
        if not session:
            raise ValueError("No session found")
        
        # Decode session
        decoder = SessionDecoder(db_session)
        df = decoder.decode(session.id)
        
        if df.empty:
            raise ValueError("No data decoded")
        
        # Find SPN columns
        spn_cols = [col for col in df.columns if col.startswith('spn_')]
        
        spn_list = []
        for col in tqdm(spn_cols, desc="Analyzing SPNs"):
            spn_id = int(col.split('_')[1])
            values = df[col].dropna()
            
            if len(values) >= min_samples:
                # Get SPN name
                spn_def = db_session.exec(
                    select(SpnDefinition).where(SpnDefinition.spn == spn_id)
                ).first()
                
                spn_list.append({
                    'spn_id': spn_id,
                    'spn_name': spn_def.spn_name if spn_def else f"SPN_{spn_id}",
                    'count': len(values),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'mean': float(values.mean()),
                    'std': float(values.std())
                })
        
        result_df = pd.DataFrame(spn_list)
        result_df = result_df.sort_values('count', ascending=False)
        
        print(f"\nFound {len(result_df)} SPNs with at least {min_samples} samples")
        
        return result_df


def generate_spn_attack_dataset(
    spn_id: int,
    num_samples_per_class: int = 1000,
    normal_range: Optional[tuple] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Génère un dataset d'attaques pour un SPN spécifique.
    
    Args:
        spn_id: ID du SPN
        num_samples_per_class: Nombre de samples par type d'attaque
        normal_range: Plage normale (min, max), si None extrait de la DB
        seed: Random seed
        
    Returns:
        DataFrame avec colonnes: value, label, label_numeric
    """
    from .dos_generator import DoSGenerator
    from .fuzzing_generator import FuzzingGenerator
    from .spoofing_generator import SpoofingGenerator
    from .injection_generator import InjectionGenerator
    from .normal_traffic_generator import NormalTrafficGenerator
    
    print(f"\nGenerating attack dataset for SPN {spn_id}...")
    
    # Extract normal range if not provided
    if normal_range is None:
        stats = extract_spn_statistics(spn_id)
        normal_range = (stats['min'], stats['max'])
        print(f"  Using range from DB: [{normal_range[0]:.2f}, {normal_range[1]:.2f}]")
    
    # Create generators
    generators = {
        'normal': NormalTrafficGenerator(seed=seed),
        'dos': DoSGenerator(seed=seed),
        'fuzzing': FuzzingGenerator(seed=seed),
        'spoofing': SpoofingGenerator(seed=seed),
        'injection': InjectionGenerator(seed=seed)
    }
    
    # Label mapping
    label_map = {
        'normal': 0,
        'dos': 1,
        'fuzzing': 2,
        'spoofing': 3,
        'injection': 4
    }
    
    # Generate values for each attack type
    all_data = []
    
    for attack_type, generator in tqdm(generators.items(), desc="Generating attacks"):
        values = generator.generate_spn_values(
            spn_id=spn_id,
            num_samples=num_samples_per_class,
            normal_range=normal_range
        )
        
        for val in values:
            all_data.append({
                'spn_id': spn_id,
                'value': val,
                'label': attack_type,
                'label_numeric': label_map[attack_type]
            })
    
    df = pd.DataFrame(all_data)
    
    print(f"\n✓ Generated {len(df)} samples")
    print("\nClass distribution:")
    print(df['label'].value_counts())
    
    return df
