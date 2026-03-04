from unittest.mock import patch

import pandas as pd

from src.data_loader import carregar_planilhas


class TestCarregarPlanilhas:
    """Testes para a função carregar_planilhas"""

    @patch("pandas.read_excel")
    def test_carregar_planilhas_retorna_dict(self, mock_read_excel):
        """Testa se carregar_planilhas retorna um dicionário"""
        # Arrange
        mock_df = pd.DataFrame({"col1": [1, 2, 3]})
        mock_read_excel.return_value = mock_df

        # Act
        resultado = carregar_planilhas("caminho/arquivo.xlsx")

        # Assert
        assert isinstance(resultado, dict)
        assert len(resultado) == 3

    @patch("pandas.read_excel")
    def test_carregar_planilhas_chaves_corretas(self, mock_read_excel):
        """Testa se as chaves do dicionário são os anos esperados"""
        # Arrange
        mock_df = pd.DataFrame({"col1": [1, 2, 3]})
        mock_read_excel.return_value = mock_df

        # Act
        resultado = carregar_planilhas("caminho/arquivo.xlsx")

        # Assert
        assert 2022 in resultado
        assert 2023 in resultado
        assert 2024 in resultado

    @patch("pandas.read_excel")
    def test_carregar_planilhas_values_sao_dataframes(self, mock_read_excel):
        """Testa se os valores do dicionário são DataFrames"""
        # Arrange
        mock_df = pd.DataFrame({"col1": [1, 2, 3]})
        mock_read_excel.return_value = mock_df

        # Act
        resultado = carregar_planilhas("caminho/arquivo.xlsx")

        # Assert
        for ano, df in resultado.items():
            assert isinstance(df, pd.DataFrame)

    @patch("pandas.read_excel")
    def test_carregar_planilhas_sheets_corretos(self, mock_read_excel):
        """Testa se as abas corretas são lidas"""
        # Arrange
        mock_df = pd.DataFrame({"col1": [1, 2, 3]})
        mock_read_excel.return_value = mock_df

        # Act
        carregar_planilhas("caminho/arquivo.xlsx")

        # Assert
        assert mock_read_excel.call_count == 3
        calls = mock_read_excel.call_args_list

        sheet_names = [call.kwargs.get("sheet_name") for call in calls]
        assert "PEDE2022" in sheet_names
        assert "PEDE2023" in sheet_names
        assert "PEDE2024" in sheet_names

    @patch("pandas.read_excel")
    def test_carregar_planilhas_usa_path(self, mock_read_excel):
        """Testa se o caminho é convertido para Path"""
        # Arrange
        mock_df = pd.DataFrame({"col1": [1, 2, 3]})
        mock_read_excel.return_value = mock_df
        caminho = "../data/raw/BASE DE DADOS PEDE 2024.xlsx"

        # Act
        carregar_planilhas(caminho)

        # Assert
        assert mock_read_excel.called
