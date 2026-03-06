import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.model_factory import get_model


class TestGetModel:
    """Testes para a função get_model"""

    def test_get_model_logreg(self):
        """Testa se get_model retorna LogisticRegression para 'logreg'"""
        # Arrange
        model_name = "logreg"

        # Act
        model = get_model(model_name)

        # Assert
        assert isinstance(model, LogisticRegression)

    def test_get_model_logreg_parametros(self):
        """Testa se LogisticRegression tem os parâmetros corretos"""
        # Arrange
        model_name = "logreg"

        # Act
        model = get_model(model_name)

        # Assert
        assert model.random_state == 42
        assert model.max_iter == 1000

    def test_get_model_rf(self):
        """Testa se get_model retorna RandomForestClassifier para 'rf'"""
        # Arrange
        model_name = "rf"

        # Act
        model = get_model(model_name)

        # Assert
        assert isinstance(model, RandomForestClassifier)

    def test_get_model_rf_parametros(self):
        """Testa se RandomForestClassifier tem os parâmetros corretos"""
        # Arrange
        model_name = "rf"

        # Act
        model = get_model(model_name)

        # Assert
        assert model.random_state == 42

    def test_get_model_invalido(self):
        """Testa se get_model lança ValueError para modelo não suportado"""
        # Arrange
        model_name = "modelo_inexistente"

        # Act & Assert
        with pytest.raises(ValueError, match="Modelo não suportado"):
            get_model(model_name)

    def test_get_model_case_sensitive(self):
        """Testa se get_model é sensível a maiúsculas/minúsculas"""
        # Arrange
        model_name = "LogReg"

        # Act & Assert
        with pytest.raises(ValueError, match="Modelo não suportado"):
            get_model(model_name)
