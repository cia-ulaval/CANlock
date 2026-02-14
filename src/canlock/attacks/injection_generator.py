"""
Générateur d'Attaques par Injection

Injecte des pics/anomalies soudains dans le signal.
Simule l'injection de commandes malveillantes ou la manipulation de capteurs.
"""

import numpy as np
from typing import Tuple
from .attack_generator import AttackGenerator


class InjectionGenerator(AttackGenerator):
    """
    Générateur d'attaques par injection.
    
    Injecte des pics ou anomalies soudaines dans le signal, simulant
    l'injection de commandes malveillantes ou la manipulation de capteurs.
    """
    
    def __init__(self, seed: int = 42, injection_probability: float = 0.2):
        """
        Initialise le générateur Injection.
        
        Args:
            seed: Graine aléatoire
            injection_probability: Probabilité d'injecter un pic à chaque échantillon (0-1)
        """
        super().__init__(seed)
        self.injection_probability = injection_probability
    
    def generate_spn_values(
        self,
        spn_id: int,
        num_samples: int,
        normal_range: Tuple[float, float],
        **kwargs
    ) -> np.ndarray:
        """
        Génère des valeurs d'injection pour un SPN.
        
        Stratégie:
        - Commencer avec des valeurs de base d'apparence normale
        - Injecter aléatoirement des pics (sauts soudains vers des valeurs extrêmes)
        - Les pics peuvent être positifs ou négatifs
        - La durée du pic peut varier (échantillon unique ou courte rafale)
        
        Args:
            spn_id: Identifiant SPN
            num_samples: Nombre de valeurs à générer
            normal_range: Plage de valeurs normale (min, max)
            **kwargs: Paramètres optionnels
                - spike_magnitude: Magnitude des pics relative à la plage (défaut: 1.5)
                - burst_length: Longueur des rafales de pics (défaut: 1-5 échantillons)
        
        Returns:
            Tableau de valeurs d'attaque par injection
        """
        min_val, max_val = self.validate_normal_range(normal_range)
        spike_magnitude = kwargs.get('spike_magnitude', 1.5)
        range_width = max_val - min_val
        
        # Générer des valeurs de base normales
        center = (min_val + max_val) / 2
        values = np.random.normal(center, range_width * 0.2, num_samples)
        values = np.clip(values, min_val, max_val)
        
        # Injecter des pics
        i = 0
        while i < num_samples:
            if np.random.random() < self.injection_probability:
                # Injecter un pic
                burst_length = np.random.randint(1, 6)  # 1-5 échantillons
                burst_end = min(i + burst_length, num_samples)
                
                # Direction du pic (positif ou négatif)
                if np.random.random() < 0.5:
                    # Pic positif
                    spike_val = max_val + range_width * spike_magnitude * np.random.uniform(0.5, 1.0)
                else:
                    # Pic négatif
                    spike_val = min_val - range_width * spike_magnitude * np.random.uniform(0.5, 1.0)
                
                # Appliquer le pic à la rafale
                for j in range(i, burst_end):
                    values[j] = spike_val + np.random.normal(0, range_width * 0.05)
                
                i = burst_end
            else:
                i += 1
        
        return values
    
    def get_attack_type(self) -> str:
        """Retourne l'identifiant du type d'attaque."""
        return 'injection'
    
    def get_metadata(self) -> dict:
        """Retourne les métadonnées du générateur."""
        metadata = super().get_metadata()
        metadata['injection_probability'] = self.injection_probability
        return metadata
