from unittest.mock import patch

import pandas as pd

from src.data_integration import consolidar_datasets


class TestConsolidarDatasets:
    """Testes para a função consolidar_datasets"""

    @patch("src.data_integration.processar_dataset")
    def test_consolidar_datasets_retorna_dataframe(self, mock_processar):
        """Testa se consolidar_datasets retorna um DataFrame"""
        # Arrange
        mock_processar.return_value = pd.DataFrame({"col1": [1, 2]})
        datasets = {
            2022: pd.DataFrame({"col1": [1, 2]}),
            2023: pd.DataFrame({"col1": [3, 4]}),
            2024: pd.DataFrame({"col1": [5, 6]}),
        }

        # Act
        resultado = consolidar_datasets(datasets)

        # Assert
        assert isinstance(resultado, pd.DataFrame)

    @patch("src.data_integration.processar_dataset")
    def test_consolidar_datasets_processa_todos_anos(self, mock_processar):
        """Testa se todos os datasets são processados"""
        # Arrange
        mock_processar.return_value = pd.DataFrame({"col1": [1, 2]})
        datasets = {
            2022: pd.DataFrame({"col1": [1, 2]}),
            2023: pd.DataFrame({"col1": [3, 4]}),
            2024: pd.DataFrame({"col1": [5, 6]}),
        }

        # Act
        consolidar_datasets(datasets)

        # Assert
        assert mock_processar.call_count == 3

    @patch("src.data_integration.processar_dataset")
    def test_consolidar_datasets_concatena_corretamente(self, mock_processar):
        """Testa se os datasets são concatenados corretamente"""
        # Arrange
        df_2022 = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        df_2023 = pd.DataFrame({"col1": [3, 4], "col2": ["c", "d"]})
        df_2024 = pd.DataFrame({"col1": [5, 6], "col2": ["e", "f"]})

        mock_processar.side_effect = [df_2022, df_2023, df_2024]
        datasets = {
            2022: pd.DataFrame(),
            2023: pd.DataFrame(),
            2024: pd.DataFrame(),
        }

        # Act
        resultado = consolidar_datasets(datasets)

        # Assert
        assert len(resultado) == 6
        assert list(resultado["col1"]) == [1, 2, 3, 4, 5, 6]

    @patch("src.data_integration.processar_dataset")
    def test_consolidar_datasets_preserve_colunas(self, mock_processar):
        """Testa se as colunas são preservadas após consolidação"""
        # Arrange
        df_processado = pd.DataFrame(
            {"col1": [1, 2], "col2": ["a", "b"], "col3": [10.5, 20.5]}
        )
        mock_processar.return_value = df_processado
        datasets = {
            2022: pd.DataFrame(),
            2023: pd.DataFrame(),
            2024: pd.DataFrame(),
        }

        # Act
        resultado = consolidar_datasets(datasets)

        # Assert
        assert "col1" in resultado.columns
        assert "col2" in resultado.columns
        assert "col3" in resultado.columns

    @patch("src.data_integration.processar_dataset")
    def test_consolidar_datasets_dataset_vazio(self, mock_processar):
        """Testa consolidação com apenas um dataset"""
        # Arrange
        df_vazio = pd.DataFrame({"col": []})
        mock_processar.return_value = df_vazio
        datasets = {2022: pd.DataFrame()}

        # Act
        resultado = consolidar_datasets(datasets)

        # Assert
        assert isinstance(resultado, pd.DataFrame)

    @patch("src.data_integration.processar_dataset")
    def test_consolidar_datasets_chama_processar_com_ano(self, mock_processar):
        """Testa se processar_dataset é chamado com os parâmetros corretos"""
        # Arrange
        mock_processar.return_value = pd.DataFrame({"col1": [1]})
        df_2022 = pd.DataFrame({"col1": [1, 2]})
        datasets = {2022: df_2022}

        # Act
        consolidar_datasets(datasets)

        # Assert
        mock_processar.assert_called_with(df_2022, 2022)
