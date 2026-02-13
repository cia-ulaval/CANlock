import pandas as pd

class DDoSAttack:

    def __init__(self, signal_name, repetitions=1000, interval=0.0001):
        self.signal_name = signal_name
        self.repetitions = repetitions
        self.interval = interval  # intervalle temporel


    def inject(self, df):

        if self.signal_name not in df.columns:
            raise ValueError("Signal non trouvé dans le dataset")

        df_copy = df.copy()

        last_time = df_copy.index.max()
        template_row = df_copy.iloc[-1].copy()

        attack_rows = []

        for i in range(self.repetitions):
            new_row = template_row.copy()
            new_time = last_time + (i + 1) * self.interval

            attack_rows.append((new_time, new_row))

        attack_df = pd.DataFrame(
            [row for _, row in attack_rows],
            index=[time for time, _ in attack_rows]
        )

        df_attacked = pd.concat([df_copy, attack_df])
        df_attacked.sort_index(inplace=True)

        return df_attacked
