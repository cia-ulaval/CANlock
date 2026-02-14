"""
Générateur d'Attaques par Déni de Service (DoS)

Génère des valeurs à haute fréquence pour saturer le signal du bus CAN.
Simule une attaque par flooding où un attaquant envoie de nombreux messages rapidement
pour empêcher le traitement des messages légitimes.
"""

import numpy as np
from typing import Tuple
from .attack_generator import AttackGenerator


class DoSGenerator(AttackGenerator):
    """
    Générateur d'attaques par déni de service (DoS).
    
    Simule une attaque par saturation du bus CAN en générant des valeurs
    à très haute fréquence. Les valeurs peuvent être répétitives ou
    aléatoires, l'objectif étant de surcharger le système.
    """
    
    def __init__(self, seed: int = 42, repetition_probability: float = 0.8):
        """
        Initialise le générateur DoS.
        
        Args:
            seed: Graine aléatoire
            repetition_probability: Probabilité de répéter la même valeur (0-1)
                Une haute répétition est typique des attaques DoS
        """
        super().__init__(seed)
        self.repetition_probability = repetition_probability
    
    def generate_spn_values(
        self,
        spn_id: int,
        num_samples: int,
        normal_range: Tuple[float, float],
        **kwargs
    ) -> np.ndarray:
        """
        Génère des valeurs DoS pour un SPN.
        
        Stratégie:
        - Sélectionner 1-3 "valeurs d'attaque" (aléatoires ou spécifiques comme min/max)
        - Répéter ces valeurs avec haute probabilité
        - Injecter occasionnellement des valeurs aléatoires pour rendre la détection plus difficile
        
        Args:
            spn_id: Identifiant SPN
            num_samples: Nombre de valeurs à générer
            normal_range: Plage de valeurs normale (min, max)
            **kwargs: Paramètres optionnels
                - attack_values: Valeurs spécifiques à utiliser pour le DoS (défaut: aléatoire)
        
        Returns:
            Tableau de valeurs d'attaque DoS
        """
        min_val, max_val = self.validate_normal_range(normal_range)
        
        # Obtenir ou générer les valeurs d'attaque
        attack_values = kwargs.get('attack_values', None)
        if attack_values is None:
            # Choisir 1-3 valeurs d'attaque (souvent des valeurs limites)
            num_attack_vals = np.random.randint(1, 4)
            attack_values = np.random.choice(
                [min_val, max_val, (min_val + max_val) / 2],
                size=num_attack_vals,
                replace=False
            )
        
        # Générer les valeurs
        values = np.zeros(num_samples)
        
        for i in range(num_samples):
            if np.random.random() < self.repetition_probability:
                # Répéter une des valeurs d'attaque
                values[i] = np.random.choice(attack_values)
            else:
                # Valeur aléatoire dans la plage (pour ajouter du bruit)
                values[i] = np.random.uniform(min_val, max_val)
        
        return values
    
    def get_attack_type(self) -> str:
        """Retourne l'identifiant du type d'attaque."""
        return 'dos'
    
    def get_metadata(self) -> dict:
        """Retourne les métadonnées du générateur."""
        metadata = super().get_metadata()
        metadata['repetition_probability'] = self.repetition_probability
        return metadata
