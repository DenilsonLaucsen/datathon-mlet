from typing import Dict

import pandas as pd

from .data_cleaning import processar_dataset


def consolidar_datasets(datasets: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    dfs_processados = []

    for ano, df in datasets.items():
        df_processado = processar_dataset(df, ano)
        dfs_processados.append(df_processado)

    df_final = pd.concat(dfs_processados, ignore_index=True)

    return df_final
