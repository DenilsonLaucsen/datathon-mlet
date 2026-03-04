import pandas as pd

MAPA_COLUNAS = {
    "Ano nasc": "AnoNascimento",
    "Data de Nasc": "AnoNascimento",
    "Defas": "Defasagem",
    "Defasagem": "Defasagem",
    "Fase ideal": "FaseIdeal",
    "Fase Ideal": "FaseIdeal",
    "Idade 22": "Idade",
    "Idade": "Idade",
    "Inglês": "Ingles",
    "Ing": "Ingles",
    "Matem": "Matematica",
    "Mat": "Matematica",
    "Nome": "Nome",
    "Nome Anonimizado": "Nome",
    "Portug": "Portugues",
    "Por": "Portugues",
}

COLUNAS_NUMERICAS = [
    "Defasagem",
    "Idade",
    "Ingles",
    "Matematica",
    "Portugues",
]


def padronizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.rename(columns=MAPA_COLUNAS)
    return df


def tratar_ano_nascimento(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "AnoNascimento" not in df.columns:
        return df

    col = df["AnoNascimento"]

    if pd.api.types.is_datetime64_any_dtype(col):
        df["AnoNascimento"] = col.dt.year

    elif col.dtype == "object":
        df["AnoNascimento"] = pd.to_datetime(col, errors="coerce").dt.year

    df["AnoNascimento"] = df["AnoNascimento"].astype("Int64")

    return df


def converter_numericas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in COLUNAS_NUMERICAS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def organizar_inde(df: pd.DataFrame, ano: int) -> pd.DataFrame:
    df = df.copy()

    df["INDE_atual"] = None
    df["INDE_ano_anterior"] = None
    df["INDE_2_anos_atras"] = None

    colunas_possiveis = {
        2022: ["INDE 22"],
        2023: ["INDE 23", "INDE 2023"],
        2024: ["INDE 2024"],
    }

    for col in colunas_possiveis.get(ano, []):
        if col in df.columns:
            df["INDE_atual"] = df[col]
            break

    if ano >= 2023 and "INDE 22" in df.columns:
        df["INDE_ano_anterior"] = df["INDE 22"]

    if ano == 2024:
        if "INDE 23" in df.columns:
            df["INDE_ano_anterior"] = df["INDE 23"]
        if "INDE 22" in df.columns:
            df["INDE_2_anos_atras"] = df["INDE 22"]

    for col in ["INDE_atual", "INDE_ano_anterior", "INDE_2_anos_atras"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    colunas_antigas = [
        "INDE 22",
        "INDE 23",
        "INDE 2023",
        "INDE 2024",
    ]

    df = df.drop(columns=[c for c in colunas_antigas if c in df.columns])

    return df


def organizar_pedra(df: pd.DataFrame, ano: int) -> pd.DataFrame:
    df = df.copy()

    df["Pedra"] = None

    colunas_possiveis = {
        2022: ["Pedra 22"],
        2023: ["Pedra 23", "Pedra 2023"],
        2024: ["Pedra 2024"],
    }

    for col in colunas_possiveis.get(ano, []):
        if col in df.columns:
            df["Pedra"] = df[col]
            break

    colunas_pedra = [
        "Pedra 20",
        "Pedra 21",
        "Pedra 22",
        "Pedra 23",
        "Pedra 2023",
        "Pedra 2024",
    ]

    df = df.drop(columns=[c for c in colunas_pedra if c in df.columns])

    return df


def processar_dataset(df: pd.DataFrame, ano: int) -> pd.DataFrame:
    df = padronizar_colunas(df)
    df = tratar_ano_nascimento(df)
    df = converter_numericas(df)
    df = organizar_inde(df, ano)
    df = organizar_pedra(df, ano)
    df["AnoReferencia"] = ano

    return df
