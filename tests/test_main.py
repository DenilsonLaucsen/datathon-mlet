from unittest.mock import MagicMock, patch

import pytest

from app.main import app


# Fixture para criar cliente de teste
@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    return TestClient(app)


class TestHealthEndpoint:
    """Testes para o endpoint /health"""

    def test_health_returns_ok(self, client):
        """Testa se health endpoint retorna status ok"""
        # Act
        response = client.get("/health")

        # Assert
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_health_response_type(self, client):
        """Testa se resposta do health é um dicionário"""
        # Act
        response = client.get("/health")
        data = response.json()

        # Assert
        assert isinstance(data, dict)
        assert "status" in data

    def test_health_status_value(self, client):
        """Testa se o valor de status é 'ok'"""
        # Act
        response = client.get("/health")
        data = response.json()

        # Assert
        assert data["status"] == "ok"


class TestPredictEndpoint:
    """Testes para o endpoint /predict"""

    @patch("app.predict.get_model")
    def test_predict_com_dados_validos(self, mock_get_model, client):
        """Testa predict endpoint com dados válidos"""
        # Arrange
        mock_model = MagicMock()
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = [[0.3, 0.7]]
        mock_get_model.return_value = mock_model

        request_data = {
            "Matematica": 8.0,
            "Portugues": 7.5,
            "Ingles": 6.0,
            "IDA": 7.5,
            "Cg": 100,
            "Cf": 50,
            "Ct": 20,
        }

        # Act
        response = client.post("/predict", json=request_data)

        # Assert
        assert response.status_code == 200

    @patch("app.predict.get_model")
    def test_predict_status_code(self, mock_get_model, client):
        """Testa se status code é 200"""
        # Arrange
        mock_model = MagicMock()
        mock_model.predict.return_value = [0]
        mock_model.predict_proba.return_value = [[0.8, 0.2]]
        mock_get_model.return_value = mock_model

        request_data = {
            "Matematica": 9.0,
            "Portugues": 8.5,
            "Ingles": 7.0,
            "IDA": 8.5,
            "Cg": 50,
            "Cf": 25,
            "Ct": 10,
        }

        # Act
        response = client.post("/predict", json=request_data)

        # Assert
        assert response.status_code == 200

    @patch("app.predict.get_model")
    def test_predict_response_formato(self, mock_get_model, client):
        """Testa se resposta segue formato PredictionResponse"""
        # Arrange
        mock_model = MagicMock()
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = [[0.4, 0.6]]
        mock_get_model.return_value = mock_model

        request_data = {
            "Matematica": 7.0,
            "Portugues": 6.5,
            "Ingles": 5.0,
            "IDA": 6.5,
            "Cg": 150,
            "Cf": 75,
            "Ct": 30,
        }

        # Act
        response = client.post("/predict", json=request_data)
        data = response.json()

        # Assert
        assert "prediction" in data
        assert "probability" in data

    @patch("app.predict.get_model")
    def test_predict_prediction_field_is_int(self, mock_get_model, client):
        """Testa se prediction é um inteiro"""
        # Arrange
        mock_model = MagicMock()
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = [[0.2, 0.8]]
        mock_get_model.return_value = mock_model

        request_data = {
            "Matematica": 8.0,
            "Portugues": 7.5,
            "Ingles": 6.0,
            "IDA": 7.5,
            "Cg": 100,
            "Cf": 50,
            "Ct": 20,
        }

        # Act
        response = client.post("/predict", json=request_data)
        data = response.json()

        # Assert
        assert isinstance(data["prediction"], int)
        assert data["prediction"] in [0, 1]

    @patch("app.predict.get_model")
    def test_predict_probability_field_is_float(self, mock_get_model, client):
        """Testa se probability é um float"""
        # Arrange
        mock_model = MagicMock()
        mock_model.predict.return_value = [0]
        mock_model.predict_proba.return_value = [[0.9, 0.1]]
        mock_get_model.return_value = mock_model

        request_data = {
            "Matematica": 9.0,
            "Portugues": 8.5,
            "Ingles": 7.0,
            "IDA": 8.5,
            "Cg": 50,
            "Cf": 25,
            "Ct": 10,
        }

        # Act
        response = client.post("/predict", json=request_data)
        data = response.json()

        # Assert
        assert isinstance(data["probability"], float)
        assert 0 <= data["probability"] <= 1

    @patch("app.predict.get_model")
    def test_predict_probability_range(self, mock_get_model, client):
        """Testa se probability está entre 0 e 1"""
        # Arrange
        mock_model = MagicMock()
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = [[0.3, 0.7]]
        mock_get_model.return_value = mock_model

        request_data = {
            "Matematica": 5.0,
            "Portugues": 4.5,
            "Ingles": 3.0,
            "IDA": 4.5,
            "Cg": 200,
            "Cf": 100,
            "Ct": 50,
        }

        # Act
        response = client.post("/predict", json=request_data)
        data = response.json()

        # Assert
        assert 0 <= data["probability"] <= 1

    def test_predict_missing_field(self, client):
        """Testa predict com campo obrigatório faltando"""
        # Arrange
        request_data = {
            "Matematica": 8.0,
            "Portugues": 7.5,
            "Ingles": 6.0,
            "IDA": 7.5,
            # Faltam: Cg, Cf, Ct
        }

        # Act
        response = client.post("/predict", json=request_data)

        # Assert
        assert response.status_code == 422

    def test_predict_invalid_type(self, client):
        """Testa predict com tipo de dado inválido"""
        # Arrange
        request_data = {
            "Matematica": "não é número",  # Inválido
            "Portugues": 7.5,
            "Ingles": 6.0,
            "IDA": 7.5,
            "Cg": 100,
            "Cf": 50,
            "Ct": 20,
        }

        # Act
        response = client.post("/predict", json=request_data)

        # Assert
        assert response.status_code == 422

    def test_predict_empty_request(self, client):
        """Testa predict com requisição vazia"""
        # Act
        response = client.post("/predict", json={})

        # Assert
        assert response.status_code == 422


class TestPredictLogic:
    """Testes para lógica de predição"""

    @patch("app.predict.get_model")
    def test_predict_chama_model_predict(self, mock_get_model, client):
        """Testa se predict chama model.predict com dados corretos"""
        # Arrange
        mock_model = MagicMock()
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = [[0.3, 0.7]]
        mock_get_model.return_value = mock_model

        request_data = {
            "Matematica": 8.0,
            "Portugues": 7.5,
            "Ingles": 6.0,
            "IDA": 7.5,
            "Cg": 100,
            "Cf": 50,
            "Ct": 20,
        }

        # Act
        response = client.post("/predict", json=request_data)

        # Assert
        assert response.status_code == 200
        assert mock_model.predict.called
        assert mock_model.predict_proba.called

    @patch("app.predict.get_model")
    def test_predict_retorna_probabilidade_correta(self, mock_get_model, client):
        """Testa se predict retorna a probabilidade correta"""
        # Arrange
        mock_model = MagicMock()
        probability_value = 0.75
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = [
            [1 - probability_value, probability_value]
        ]
        mock_get_model.return_value = mock_model

        request_data = {
            "Matematica": 8.0,
            "Portugues": 7.5,
            "Ingles": 6.0,
            "IDA": 7.5,
            "Cg": 100,
            "Cf": 50,
            "Ct": 20,
        }

        # Act
        response = client.post("/predict", json=request_data)
        data = response.json()

        # Assert
        assert data["probability"] == pytest.approx(probability_value, abs=0.01)

    @patch("app.predict.get_model")
    def test_predict_defasado(self, mock_get_model, client):
        """Testa predição de aluno defasado (prediction=1)"""
        # Arrange
        mock_model = MagicMock()
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = [[0.2, 0.8]]
        mock_get_model.return_value = mock_model

        request_data = {
            "Matematica": 3.0,
            "Portugues": 3.5,
            "Ingles": 2.0,
            "IDA": 3.5,
            "Cg": 250,
            "Cf": 150,
            "Ct": 80,
        }

        # Act
        response = client.post("/predict", json=request_data)
        data = response.json()

        # Assert
        assert data["prediction"] == 1
        assert data["probability"] > 0.5

    @patch("app.predict.get_model")
    def test_predict_nao_defasado(self, mock_get_model, client):
        """Testa predição de aluno não defasado (prediction=0)"""
        # Arrange
        mock_model = MagicMock()
        mock_model.predict.return_value = [0]
        mock_model.predict_proba.return_value = [[0.85, 0.15]]
        mock_get_model.return_value = mock_model

        request_data = {
            "Matematica": 9.0,
            "Portugues": 8.5,
            "Ingles": 8.0,
            "IDA": 8.5,
            "Cg": 20,
            "Cf": 10,
            "Ct": 5,
        }

        # Act
        response = client.post("/predict", json=request_data)
        data = response.json()

        # Assert
        assert data["prediction"] == 0
        assert data["probability"] < 0.5


class TestAppConfiguration:
    """Testes para configuração da aplicação"""

    def test_app_title(self):
        """Testa se a aplicação tem título correto"""
        # Assert
        assert app.title == "Student Defasagem Predictor"

    def test_app_version(self):
        """Testa se a aplicação tem versão correta"""
        # Assert
        assert app.version == "1.0.0"

    def test_health_endpoint_exists(self, client):
        """Testa se endpoint /health existe"""
        # Act
        response = client.get("/health")

        # Assert
        assert response.status_code == 200

    def test_predict_endpoint_accepts_post(self, client):
        """Testa se endpoint /predict aceita método POST"""
        # Arrange
        # Tentar POST sem dados válidos para verificar que endpoint existe
        response = client.post("/predict", json={})

        # Assert - Espera erro de validação, não 404
        assert response.status_code != 404
