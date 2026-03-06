import pandas as pd

from src.data.cleaning import (
    converter_numericas,
    organizar_inde,
    organizar_pedra,
    padronizar_colunas,
    processar_dataset,
    tratar_ano_nascimento,
)


class TestPadronizarColunas:
    """Testes para a função padronizar_colunas"""

    def test_padronizar_colunas_nao_modifica_original(self):
        """Testa se a função não modifica o DataFrame original"""
        # Arrange
        df_original = pd.DataFrame({"Ano nasc": [2000, 2001]})
        df = df_original.copy()

        # Act
        padronizar_colunas(df)

        # Assert
        assert "Ano nasc" in df.columns
        pd.testing.assert_frame_equal(df, df_original)

    def test_padronizar_colunas_renomeia_corretamente(self):
        """Testa se as colunas são renomeadas corretamente"""
        # Arrange
        df = pd.DataFrame({"Ano nasc": [2000, 2001]})

        # Act
        resultado = padronizar_colunas(df)

        # Assert
        assert "AnoNascimento" in resultado.columns
        assert "Ano nasc" not in resultado.columns

    def test_padronizar_colunas_multiplas_renomeacoes(self):
        """Testa renomeação de múltiplas colunas"""
        # Arrange
        df = pd.DataFrame(
            {
                "Ano nasc": [2000, 2001],
                "Defas": [1, -1],
                "Inglês": [7, 8],
            }
        )

        # Act
        resultado = padronizar_colunas(df)

        # Assert
        assert "AnoNascimento" in resultado.columns
        assert "Defasagem" in resultado.columns
        assert "Ingles" in resultado.columns

    def test_padronizar_colunas_preserva_outras_colunas(self):
        """Testa se colunas não mapeadas são preservadas"""
        # Arrange
        df = pd.DataFrame(
            {
                "Ano nasc": [2000, 2001],
                "ColunaNaoMapeada": [1, 2],
            }
        )

        # Act
        resultado = padronizar_colunas(df)

        # Assert
        assert "ColunaNaoMapeada" in resultado.columns


class TestTratarAnoNascimento:
    """Testes para a função tratar_ano_nascimento"""

    def test_tratar_ano_nascimento_sem_coluna(self):
        """Testa comportamento quando coluna AnoNascimento não existe"""
        # Arrange
        df = pd.DataFrame({"Outra": [1, 2]})

        # Act
        resultado = tratar_ano_nascimento(df)

        # Assert
        assert "AnoNascimento" not in resultado.columns
        assert len(resultado) == 2

    def test_tratar_ano_nascimento_datetime(self):
        """Testa conversão de valores datetime"""
        # Arrange
        df = pd.DataFrame(
            {"AnoNascimento": pd.to_datetime(["2000-01-01", "2001-05-15"])}
        )

        # Act
        resultado = tratar_ano_nascimento(df)

        # Assert
        assert resultado["AnoNascimento"].dtype == "Int64"
        assert list(resultado["AnoNascimento"]) == [2000, 2001]

    def test_tratar_ano_nascimento_string(self):
        """Testa conversão de valores string"""
        # Arrange
        df = pd.DataFrame({"AnoNascimento": ["2000-01-01", "2001-05-15"]})

        # Act
        resultado = tratar_ano_nascimento(df)

        # Assert
        assert resultado["AnoNascimento"].dtype == "Int64"
        assert list(resultado["AnoNascimento"]) == [2000, 2001]

    def test_tratar_ano_nascimento_com_valores_invalidos(self):
        """Testa conversão com valores inválidos (deve resultar em NaN)"""
        # Arrange
        df = pd.DataFrame({"AnoNascimento": ["2000-01-01", "invalido"]})

        # Act
        resultado = tratar_ano_nascimento(df)

        # Assert
        assert resultado["AnoNascimento"].isna().sum() == 1


class TestConverterNumericas:
    """Testes para a função converter_numericas"""

    def test_converter_numericas_converte_string(self):
        """Testa conversão de strings para números"""
        # Arrange
        df = pd.DataFrame(
            {
                "Defasagem": ["1", "2", "3"],
                "Idade": ["25", "30", "35"],
            }
        )

        # Act
        resultado = converter_numericas(df)

        # Assert
        assert resultado["Defasagem"].dtype in [float, "int64"]
        assert resultado["Idade"].dtype in [float, "int64"]
        assert list(resultado["Defasagem"]) == [1, 2, 3]

    def test_converter_numericas_preserva_numeros(self):
        """Testa que números já convertidos são preservados"""
        # Arrange
        df = pd.DataFrame(
            {
                "Defasagem": [1, 2, 3],
                "Idade": [25, 30, 35],
            }
        )

        # Act
        resultado = converter_numericas(df)

        # Assert
        assert all(resultado["Defasagem"] == [1, 2, 3])

    def test_converter_numericas_colunas_inexistentes(self):
        """Testa comportamento com colunas inexistentes"""
        # Arrange
        df = pd.DataFrame({"Outra": [1, 2, 3]})

        # Act
        resultado = converter_numericas(df)

        # Assert
        assert list(resultado.columns) == ["Outra"]

    def test_converter_numericas_valores_invalidos(self):
        """Testa conversão com valores inválidos (resulta em NaN)"""
        # Arrange
        df = pd.DataFrame(
            {
                "Defasagem": ["1", "invalido", "3"],
            }
        )

        # Act
        resultado = converter_numericas(df)

        # Assert
        assert resultado["Defasagem"].isna().sum() == 1


class TestOrganizarInde:
    """Testes para a função organizar_inde"""

    def test_organizar_inde_2022(self):
        """Testa organização de INDE em 2022"""
        # Arrange
        df = pd.DataFrame(
            {
                "INDE 22": [0.5, 0.7],
                "Outra": [1, 2],
            }
        )

        # Act
        resultado = organizar_inde(df, 2022)

        # Assert
        assert "INDE_atual" in resultado.columns
        assert "INDE 22" not in resultado.columns
        assert all(resultado["INDE_atual"] == [0.5, 0.7])

    def test_organizar_inde_2023_com_anterior(self):
        """Testa organização de INDE em 2023"""
        # Arrange
        df = pd.DataFrame(
            {
                "INDE 23": [0.7, 0.8],
                "INDE 22": [0.5, 0.6],
            }
        )

        # Act
        resultado = organizar_inde(df, 2023)

        # Assert
        assert all(resultado["INDE_atual"] == [0.7, 0.8])
        assert all(resultado["INDE_ano_anterior"] == [0.5, 0.6])

    def test_organizar_inde_2024_multiplos_anos(self):
        """Testa organização de INDE em 2024 com histórico"""
        # Arrange
        df = pd.DataFrame(
            {
                "INDE 2024": [0.8, 0.9],
                "INDE 23": [0.7, 0.8],
                "INDE 22": [0.5, 0.6],
            }
        )

        # Act
        resultado = organizar_inde(df, 2024)

        # Assert
        assert all(resultado["INDE_atual"] == [0.8, 0.9])
        assert all(resultado["INDE_ano_anterior"] == [0.7, 0.8])
        assert all(resultado["INDE_2_anos_atras"] == [0.5, 0.6])

    def test_organizar_inde_remove_colunas_antigas(self):
        """Testa se as colunas antigas são removidas"""
        # Arrange
        df = pd.DataFrame(
            {
                "INDE 22": [0.5],
                "INDE 23": [0.7],
                "INDE 2023": [0.7],
                "INDE 2024": [0.8],
            }
        )

        # Act
        resultado = organizar_inde(df, 2024)

        # Assert
        assert "INDE 22" not in resultado.columns
        assert "INDE 23" not in resultado.columns
        assert "INDE 2023" not in resultado.columns
        assert "INDE 2024" not in resultado.columns


class TestOrganizarPedra:
    """Testes para a função organizar_pedra"""

    def test_organizar_pedra_2022(self):
        """Testa organização de Pedra em 2022"""
        # Arrange
        df = pd.DataFrame(
            {
                "Pedra 22": [1, 2],
                "Outra": [3, 4],
            }
        )

        # Act
        resultado = organizar_pedra(df, 2022)

        # Assert
        assert "Pedra" in resultado.columns
        assert "Pedra 22" not in resultado.columns
        assert all(resultado["Pedra"] == [1, 2])

    def test_organizar_pedra_2023(self):
        """Testa organização de Pedra em 2023"""
        # Arrange
        df = pd.DataFrame(
            {
                "Pedra 23": [2, 3],
                "Outracol": [5, 6],
            }
        )

        # Act
        resultado = organizar_pedra(df, 2023)

        # Assert
        assert "Pedra" in resultado.columns
        assert "Pedra 23" not in resultado.columns
        assert all(resultado["Pedra"] == [2, 3])

    def test_organizar_pedra_2024(self):
        """Testa organização de Pedra em 2024"""
        # Arrange
        df = pd.DataFrame(
            {
                "Pedra 2024": [3, 4],
            }
        )

        # Act
        resultado = organizar_pedra(df, 2024)

        # Assert
        assert "Pedra" in resultado.columns
        assert "Pedra 2024" not in resultado.columns

    def test_organizar_pedra_remove_colunas_antigas(self):
        """Testa se todas as variações de Pedra são removidas"""
        # Arrange
        df = pd.DataFrame(
            {
                "Pedra 20": [1],
                "Pedra 21": [2],
                "Pedra 22": [3],
                "Pedra 23": [4],
                "Pedra 2023": [5],
                "Pedra 2024": [6],
            }
        )

        # Act
        resultado = organizar_pedra(df, 2024)

        # Assert
        for col in [
            "Pedra 20",
            "Pedra 21",
            "Pedra 22",
            "Pedra 23",
            "Pedra 2023",
            "Pedra 2024",
        ]:
            assert col not in resultado.columns


class TestProcessarDataset:
    """Testes para a função processar_dataset"""

    def test_processar_dataset_retorna_dataframe(self):
        """Testa se processar_dataset retorna um DataFrame"""
        # Arrange
        df = pd.DataFrame(
            {
                "Ano nasc": [2000],
                "INDE 22": [0.5],
                "Pedra 22": [1],
            }
        )

        # Act
        resultado = processar_dataset(df, 2022)

        # Assert
        assert isinstance(resultado, pd.DataFrame)

    def test_processar_dataset_adiciona_ano_referencia(self):
        """Testa se coluna AnoReferencia é adicionada"""
        # Arrange
        df = pd.DataFrame({"col": [1]})

        # Act
        resultado = processar_dataset(df, 2023)

        # Assert
        assert "AnoReferencia" in resultado.columns
        assert all(resultado["AnoReferencia"] == 2023)

    def test_processar_dataset_aplica_transformacoes(self):
        """Testa se todas as transformações são aplicadas"""
        # Arrange
        df = pd.DataFrame(
            {
                "Ano nasc": ["2000-01-01"],
                "Defas": ["1"],
                "INDE 22": [0.5],
                "Pedra 22": [1],
            }
        )

        # Act
        resultado = processar_dataset(df, 2022)

        # Assert
        # Padronizar: Defas -> Defasagem
        assert "Defasagem" in resultado.columns
        # Tratar ano nascimento: converter datetime
        assert "AnoNascimento" in resultado.columns
        # Ano referência adicionado
        assert "AnoReferencia" in resultado.columns
        # Organizar INDE
        assert "INDE_atual" in resultado.columns
        # Organizar Pedra
        assert "Pedra" in resultado.columns

    def test_processar_dataset_nao_modifica_original(self):
        """Testa se o DataFrame original não é modificado"""
        # Arrange
        df_original = pd.DataFrame(
            {
                "Ano nasc": [2000],
                "INDE 22": [0.5],
            }
        )
        df = df_original.copy()

        # Act
        processar_dataset(df, 2022)

        # Assert
        pd.testing.assert_frame_equal(df, df_original)

    def test_processar_dataset_ano_2024(self):
        """Testa processamento especializado para ano 2024"""
        # Arrange
        df = pd.DataFrame(
            {
                "Ano nasc": [2000],
                "INDE 2024": [0.8],
                "INDE 23": [0.7],
                "INDE 22": [0.5],
                "Pedra 2024": [1],
            }
        )

        # Act
        resultado = processar_dataset(df, 2024)

        # Assert
        assert all(resultado["INDE_atual"] == 0.8)
        assert all(resultado["INDE_ano_anterior"] == 0.7)
        assert all(resultado["INDE_2_anos_atras"] == 0.5)
