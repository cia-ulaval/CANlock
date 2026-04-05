from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class AttackBase(ABC):
    """Abstract base class for attack algorithms.

    Subclasses must implement `apply(df, target)` where `target` is the identifier
    of the signal to attack (SPN, PGN or a custom filter). For now the
    implementation expects the caller to provide a DataFrame and a `target`
    that the subclass understands.
    """

    def __init__(self, name: str, signal_name: Optional[str] = None):
        """Initialize the base attack algorithm.
        
        Args:
            name: The descriptive name of the attack.
            signal_name: Optional name of the signal to attack.
        """
        self.name = name
        self.signal_name = signal_name

    @abstractmethod
    def apply(self, df: pd.DataFrame, target: Optional[int] = None) -> pd.DataFrame:
        """Apply attack to the DataFrame and return a new DataFrame.

        Args:
            df: input DataFrame with CAN messages
            target: identifier of the signal to attack (semantics left to subclass)
        Returns:
            df_attacked: DataFrame with modifications (attack annotations)
        """
        raise NotImplementedError()

    @abstractmethod
    def get_attack_name(self):
        """Return the name of the attack."""
        pass