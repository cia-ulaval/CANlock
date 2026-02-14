"""
Générateur d'Attaques par Fuzzing

Génère des valeurs complètement aléatoires, hors plage, pour tester la robustesse du système.
Simule un attaquant sondant les vulnérabilités en envoyant des données malformées.
"""

import numpy as np
from typing import Tuple
from .attack_generator import AttackGenerator


class FuzzingGenerator(AttackGenerator):
    """
    Générateur d'attaques par fuzzing.
    
    Génère des valeurs complètement aléatoires, souvent en dehors de la
    plage normale, pour tester les limites du système et identifier des
    vulnérabilités potentielles.
    """
    
    def __init__(self, seed: int = 42, out_of_range_probability: float = 0.7):
        """
        Initialise le générateur Fuzzing.
        
        Args:
            seed: Graine aléatoire
            out_of_range_probability: Probabilité de générer des valeurs hors plage (0-1)
        """
        super().__init__(seed)
        self.out_of_range_probability = out_of_range_probability
    
    def generate_spn_values(
        self,
        spn_id: int,
        num_samples: int,
        normal_range: Tuple[float, float],
        **kwargs
    ) -> np.ndarray:
        """
        Génère des valeurs de fuzzing pour un SPN.
        
        Stratégie:
        - Générer des valeurs hors de la plage normale avec haute probabilité
        - Inclure des valeurs extrêmes (très hautes, très basses, négatives)
        - Générer occasionnellement des valeurs valides pour éviter les vérifications simples de plage
        
        Args:
            spn_id: Identifiant SPN
            num_samples: Nombre de valeurs à générer
            normal_range: Plage de valeurs normale (min, max)
            **kwargs: Paramètres optionnels
                - extreme_multiplier: Distance hors de la plage à parcourir (défaut: 2.0)
        
        Returns:
            Tableau de valeurs d'attaque par fuzzing
        """
        min_val, max_val = self.validate_normal_range(normal_range)
        extreme_multiplier = kwargs.get('extreme_multiplier', 2.0)
        
        values = np.zeros(num_samples)
        range_width = max_val - min_val
        
        for i in range(num_samples):
            if np.random.random() < self.out_of_range_probability:
                # Générer une valeur hors plage
                if np.random.random() < 0.5:
                    # Au-dessus du max
                    values[i] = max_val + np.random.uniform(0, range_width * extreme_multiplier)
                else:
                    # En dessous du min (possiblement négatif)
                    values[i] = min_val - np.random.uniform(0, range_width * extreme_multiplier)
            else:
                # Générer occasionnellement une valeur dans la plage
                values[i] = np.random.uniform(min_val, max_val)
        
        return values
    
    def get_attack_type(self) -> str:
        """Retourne l'identifiant du type d'attaque."""
        return 'fuzzing'
    
    def get_metadata(self) -> dict:
        """Retourne les métadonnées du générateur."""
        metadata = super().get_metadata()
        metadata['out_of_range_probability'] = self.out_of_range_probability
        return metadata
