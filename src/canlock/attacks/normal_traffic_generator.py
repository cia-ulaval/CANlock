"""
Générateur de Trafic Normal

Génère des valeurs normales réalistes pour l'entraînement de base.
Utilise les propriétés statistiques des données réelles pour créer des signaux plausibles.
"""

import numpy as np
from typing import Tuple, Optional
from .attack_generator import AttackGenerator


class NormalTrafficGenerator(AttackGenerator):
    """
    Générateur de trafic normal.
    
    Génère des valeurs normales réalistes pour servir de baseline lors
    de l'entraînement. Utilise les propriétés statistiques des données
    réelles si disponibles.
    """
    
    def __init__(
        self,
        seed: int = 42,
        mean: Optional[float] = None,
        std: Optional[float] = None
    ):
        """
        Initialise le générateur de trafic normal.
        
        Args:
            seed: Graine aléatoire
            mean: Valeur moyenne (si None, utilise le centre de la plage)
            std: Écart-type (si None, utilise plage/6)
        """
        super().__init__(seed)
        self.mean = mean
        self.std = std
    
    def generate_spn_values(
        self,
        spn_id: int,
        num_samples: int,
        normal_range: Tuple[float, float],
        **kwargs
    ) -> np.ndarray:
        """
        Génère des valeurs normales pour un SPN.
        
        Stratégie:
        - Utiliser une distribution Gaussienne centrée dans la plage
        - Ajouter une corrélation temporelle (valeurs changent graduellement)
        - Écrêter pour rester dans la plage normale
        - Optionnellement ajouter des patterns cycliques pour capteurs comme RPM, vitesse
        
        Args:
            spn_id: Identifiant SPN
            num_samples: Nombre de valeurs à générer
            normal_range: Plage de valeurs normale (min, max)
            **kwargs: Paramètres optionnels
                - add_trend: Ajouter une dérive lente au fil du temps (défaut: False)
                - add_cycles: Ajouter des patterns cycliques (défaut: False)
                - autocorrelation: Corrélation entre valeurs consécutives (défaut: 0.8)
        
        Returns:
            Tableau de valeurs de trafic normal
        """
        min_val, max_val = self.validate_normal_range(normal_range)
        range_width = max_val - min_val
        
        # Déterminer mean et std
        mean = self.mean if self.mean is not None else (min_val + max_val) / 2
        std = self.std if self.std is not None else range_width / 6
        
        # Paramètres
        autocorr = kwargs.get('autocorrelation', 0.8)
        add_trend = kwargs.get('add_trend', False)
        add_cycles = kwargs.get('add_cycles', False)
        
        # Générer des valeurs de base avec autocorrélation
        values = np.zeros(num_samples)
        values[0] = np.random.normal(mean, std)
        
        for i in range(1, num_samples):
            # Marche aléatoire autocorrélée
            innovation = np.random.normal(0, std * np.sqrt(1 - autocorr**2))
            values[i] = autocorr * values[i-1] + (1 - autocorr) * mean + innovation
        
        # Ajouter une tendance si demandé
        if add_trend:
            trend = np.linspace(0, range_width * 0.1, num_samples)
            if np.random.random() < 0.5:
                trend = -trend  # Tendance descendante
            values += trend
        
        # Ajouter un pattern cyclique si demandé
        if add_cycles:
            cycle_period = kwargs.get('cycle_period', num_samples // 4)
            cycle_amplitude = kwargs.get('cycle_amplitude', std * 2)
            t = np.arange(num_samples)
            cycle = cycle_amplitude * np.sin(2 * np.pi * t / cycle_period)
            values += cycle
        
        # Écrêter à la plage
        values = np.clip(values, min_val, max_val)
        
        return values
    
    def get_attack_type(self) -> str:
        """Retourne l'identifiant du type d'attaque."""
        return 'normal'
    
    def get_metadata(self) -> dict:
        """Retourne les métadonnées du générateur."""
        metadata = super().get_metadata()
        if self.mean is not None:
            metadata['mean'] = self.mean
        if self.std is not None:
            metadata['std'] = self.std
        return metadata
