from pathlib import Path

import pandas as pd


def carregar_planilhas(caminho_arquivo: str) -> dict:
    caminho = Path(caminho_arquivo)

    return {
        2022: pd.read_excel(caminho, sheet_name="PEDE2022"),
        2023: pd.read_excel(caminho, sheet_name="PEDE2023"),
        2024: pd.read_excel(caminho, sheet_name="PEDE2024"),
    }
