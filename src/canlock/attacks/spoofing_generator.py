"""
Générateur d'Attaques par Usurpation (Spoofing)

Génère des valeurs plausibles mais fausses dans la plage normale.
Plus subtil que le fuzzing - les valeurs semblent légitimes mais suivent des patterns anormaux.
"""

import numpy as np
from typing import Tuple
from .attack_generator import AttackGenerator


class SpoofingGenerator(AttackGenerator):
    """
    Générateur d'attaques par spoofing (usurpation).
    
    Génère des valeurs qui semblent plausibles (dans la plage normale)
    mais qui sont incorrectes. Plus subtil que le fuzzing, cet type d'attaque
    est plus difficile à détecter car les valeurs paraissent légitimes.
    """
    
    def __init__(self, seed: int = 42, offset_factor: float = 0.3):
        """
        Initialise le générateur Spoofing.
        
        Args:
            seed: Graine aléatoire
            offset_factor: Facteur de décalage par rapport aux valeurs attendues (0-1)
        """
        super().__init__(seed)
        self.offset_factor = offset_factor
    
    def generate_spn_values(
        self,
        spn_id: int,
        num_samples: int,
        normal_range: Tuple[float, float],
        **kwargs
    ) -> np.ndarray:
        """
        Génère des valeurs de spoofing pour un SPN.
        
        Stratégie:
        - Générer des valeurs dans la plage normale
        - Suivre une distribution différente de celle attendue (ex: si normale est Gaussienne,
          utiliser uniforme ou bimodale)
        - Ajouter un biais systématique ou décalage
        - Créer des patterns irréalistes (constant, tendance linéaire, etc.)
        
        Args:
            spn_id: Identifiant SPN
            num_samples: Nombre de valeurs à générer
            normal_range: Plage de valeurs normale (min, max)
            **kwargs: Paramètres optionnels
                - pattern: 'constant', 'linear', 'bimodal', 'shifted' (défaut: aléatoire)
                - bias: Décalage constant à ajouter
        
        Returns:
            Tableau de valeurs d'attaque par spoofing
        """
        min_val, max_val = self.validate_normal_range(normal_range)
        range_width = max_val - min_val
        
        # Choisir le pattern
        pattern = kwargs.get('pattern', np.random.choice([
            'constant', 'linear', 'bimodal', 'shifted'
        ]))
        
        if pattern == 'constant':
            # Valeur constante (suspect pour la plupart des capteurs)
            target_val = kwargs.get('bias', min_val + range_width * 0.5)
            values = np.full(num_samples, target_val)
            # Ajouter un petit bruit pour rendre moins évident
            values += np.random.normal(0, range_width * 0.01, num_samples)
            
        elif pattern == 'linear':
            # Tendance linéaire (irréaliste pour la plupart des capteurs)
            start = min_val + range_width * 0.3
            end = max_val - range_width * 0.3
            values = np.linspace(start, end, num_samples)
            # Add small noise
            values += np.random.normal(0, range_width * 0.02, num_samples)
            
        elif pattern == 'bimodal':
            # Distribution bimodale (deux pics)
            peak1 = min_val + range_width * 0.3
            peak2 = max_val - range_width * 0.3
            # Moitié des échantillons autour de chaque pic
            half = num_samples // 2
            values = np.concatenate([
                np.random.normal(peak1, range_width * 0.05, half),
                np.random.normal(peak2, range_width * 0.05, num_samples - half)
            ])
            np.random.shuffle(values)
            
        else:  # 'shifted'
            # Décaler toutes les valeurs par décalage
            bias = kwargs.get('bias', range_width * self.offset_factor)
            center = (min_val + max_val) / 2 + bias
            values = np.random.normal(center, range_width * 0.15, num_samples)
        
        # Écrêter à la plage pour garder plausible
        values = np.clip(values, min_val, max_val)
        
        return values
    
    def get_attack_type(self) -> str:
        """Retourne l'identifiant du type d'attaque."""
        return 'spoofing'
    
    def get_metadata(self) -> dict:
        """Retourne les métadonnées du générateur."""
        metadata = super().get_metadata()
        metadata['offset_factor'] = self.offset_factor
        return metadata
