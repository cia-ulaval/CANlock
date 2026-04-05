from typing import Optional

import pandas as pd

from canlock.attacks.attack_base import AttackBase


class DDoSAttack(AttackBase):
    """Denial of Service (DDoS) attack that floods the network with repeated message configurations."""
    
    def __init__(
        self, signal_name: str, repetitions: int = 1000, interval: float = 0.0001
    ) -> None:
        """Initialize the DDoS attack.
        
        Args:
            signal_name: Name of the targeted signal.
            repetitions: Number of times to repeat the message.
            interval: Time interval between repeated messages.
        """
        super().__init__("DDoS", signal_name)
        self.repetitions = repetitions
        self.interval = interval

    def apply(self, df: pd.DataFrame, target: Optional[int] = None) -> pd.DataFrame:
        """Apply the DDoS attack to loop repetitions of the last seen message.
        
        Args:
            df: The dataframe carrying CAN messages.
            target: Ignored by this attack type.
            
        Returns:
            The augmented dataframe.
        """
        if self.signal_name not in df.columns:
            raise ValueError("Signal not found in the dataset")

        df_copy = df.copy()

        last_time = df_copy.index.max()
        template_row = df_copy.iloc[-1].copy()

        attack_rows = []

        for i in range(self.repetitions):
            new_row = template_row.copy()
            new_time = last_time + (i + 1) * self.interval

            attack_rows.append((new_time, new_row))

        attack_df = pd.DataFrame(
            [row for _, row in attack_rows], index=[time for time, _ in attack_rows]
        )

        df_attacked = pd.concat([df_copy, attack_df])
        df_attacked.sort_index(inplace=True)

        return df_attacked

    def get_attack_name(self) -> str:
        """Get the name of the attack."""
        return self.name
