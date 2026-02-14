"""
Classe Abstraite de Base pour les Générateurs d'Attaques

Définit l'interface que tous les générateurs d'attaques doivent implémenter.
Chaque générateur se concentre sur la création de valeurs synthétiques pour un SPN spécifique.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List
import numpy as np
from datetime import datetime, timedelta


class AttackGenerator(ABC):
    """
    Classe abstraite de base pour tous les générateurs d'attaques.
    
    Chaque générateur doit implémenter les méthodes abstraites pour
    générer des valeurs synthétiques d'attaque pour un SPN spécifique.
    
    L'approche par SPN permet de cibler un seul signal à la fois,
    ce qui est plus réaliste pour la détection d'intrusions.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialise le générateur d'attaque.
        
        Args:
            seed: Graine aléatoire pour la reproductibilité
        """
        self.seed = seed
        np.random.seed(seed)
    
    @abstractmethod
    def generate_spn_values(
        self,
        spn_id: int,
        num_samples: int,
        normal_range: Tuple[float, float],
        **kwargs
    ) -> np.ndarray:
        """
        Génère des valeurs synthétiques pour un SPN spécifique.
        
        Cette méthode doit être implémentée par toutes les classes filles
        pour définir le comportement spécifique de l'attaque.
        
        Args:
            spn_id: ID du SPN cible (ex: 190 pour Engine Speed)
            num_samples: Nombre de valeurs à générer
            normal_range: Plage normale (min, max) du SPN dans les données réelles
            **kwargs: Paramètres additionnels spécifiques à l'attaque
            
        Returns:
            Tableau numpy de valeurs générées
            
        Raises:
            NotImplementedError: Si la méthode n'est pas implémentée
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement generate_spn_values()"
        )
    
    @abstractmethod
    def get_attack_type(self) -> str:
        """
        Retourne le type d'attaque.
        
        Returns:
            String identifier for the attack type (e.g., 'dos', 'fuzzing', 'spoofing')
            
        Raises:
            NotImplementedError: Si la méthode n'est pas implémentée
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_attack_type()"
        )
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Retourne les métadonnées du générateur.
        
        Cette méthode peut être surchargée pour fournir des informations
        supplémentaires spécifiques à chaque type d'attaque.
        
        Returns:
            Dictionnaire avec les métadonnées
        """
        return {
            'generator_class': self.__class__.__name__,
            'attack_type': self.get_attack_type(),
            'seed': self.seed
        }
    
    def validate_normal_range(
        self,
        normal_range: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Valide et retourne la plage normale.
        
        Args:
            normal_range: Tuple (min, max)
            
        Returns:
            Plage normale validée
            
        Raises:
            ValueError: Si la plage n'est pas valide
        """
        if len(normal_range) != 2:
            raise ValueError("normal_range must be a tuple of (min, max)")
        
        min_val, max_val = normal_range
        
        
        if min_val >= max_val:
            raise ValueError(f"min ({min_val}) must be less than max ({max_val})")
        
        return (float(min_val), float(max_val))
    
    def generate(self, num_samples: int) -> List[Dict[str, Any]]:
        """
        Génère des messages CAN synthétiques complets.
        
        Cette méthode crée des messages CAN avec des CAN IDs aléatoires et des payloads
        synthétiques basés sur le type d'attaque.
        
        Args:
            num_samples: Nombre de messages CAN à générer
            
        Returns:
            Liste de dictionnaires représentant des messages CAN
        """
        messages = []
        base_time = datetime.now()
        
        # IDs CAN communs (plage standard et étendue)
        common_can_ids = [0x0CF00400, 0x18FEF100, 0x18FEF200, 0x0CF00300, 
                         0x18FEEE00, 0x18FEF500, 0x18FEFC00, 0x18FEF600]
        
        for i in range(num_samples):
            # Générer l'ID CAN (biaisé vers les IDs communs)
            if np.random.random() < 0.7:
                can_id = np.random.choice(common_can_ids)
            else:
                can_id = np.random.randint(0, 0x1FFFFFFF)
            
            # Générer le payload (8 bytes)
            # Utiliser generate_spn_values pour créer des valeurs réalistes
            payload_values = self.generate_spn_values(
                spn_id=np.random.randint(0, 5000),
                num_samples=8,
                normal_range=(0, 255)
            )
            payload = bytes([int(v) % 256 for v in payload_values])
            
            # Timestamp avec intervalle variable
            timestamp = base_time + timedelta(milliseconds=i * np.random.uniform(1, 20))
            
            messages.append({
                'timestamp': timestamp,
                'can_identifier': can_id,
                'length': len(payload),
                'payload': payload,
                'label': self.get_attack_type()
            })
        
        return messages
    
    def __repr__(self) -> str:
        """Représentation textuelle du générateur."""
        return f"{self.__class__.__name__}(seed={self.seed})"

