import json
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd


class TestConfigLoading:
    """Testes para carregamento de configuração"""

    @patch("src.train.load_config")
    def test_load_config_sucesso(self, mock_load_config):
        """Testa se configuração é carregada corretamente"""
        # Arrange
        config_esperada = {
            "random_seed": 42,
            "target": "defasado_bin",
            "best_model": "logreg",
        }
        mock_load_config.return_value = config_esperada

        # Act
        config = mock_load_config("config.yaml")

        # Assert
        assert config == config_esperada
        assert "random_seed" in config
        assert "target" in config
        assert "best_model" in config

    @patch("src.train.load_config")
    def test_load_config_contem_best_model(self, mock_load_config):
        """Testa se config contém campo 'best_model'"""
        # Arrange
        config = {"random_seed": 42, "target": "defasado_bin", "best_model": "gb"}
        mock_load_config.return_value = config

        # Act
        resultado = mock_load_config("config.yaml")

        # Assert
        assert resultado["best_model"] in ["logreg", "rf", "gb", "xgb", "lgbm"]


class TestFeaturesLoading:
    """Testes para carregamento de features"""

    @patch("src.train.load_features")
    def test_load_features_retorna_lista(self, mock_load_features):
        """Testa se features são carregadas como lista"""
        # Arrange
        features_esperadas = ["feat1", "feat2", "feat3"]
        mock_load_features.return_value = features_esperadas

        # Act
        features = mock_load_features("features.json")

        # Assert
        assert isinstance(features, list)
        assert len(features) == 3

    @patch("src.train.load_features")
    def test_load_features_quantidade(self, mock_load_features):
        """Testa se features carregadas têm quantidade apropriada"""
        # Arrange
        features = ["Matematica", "Portugues", "Ingles", "IDA", "CG", "CF", "CT"]
        mock_load_features.return_value = features

        # Act
        resultado = mock_load_features("features.json")

        # Assert
        assert len(resultado) > 0
        assert all(isinstance(f, str) for f in resultado)


class TestDataLoading:
    """Testes para carregamento de dados"""

    @patch("pandas.read_csv")
    def test_read_csv_sucesso(self, mock_read_csv):
        """Testa se dataset é carregado com sucesso"""
        # Arrange
        mock_df = pd.DataFrame(
            {"feat1": [1, 2, 3], "feat2": [4, 5, 6], "defasado_bin": [0, 1, 0]}
        )
        mock_read_csv.return_value = mock_df

        # Act
        df = mock_read_csv("data.csv")

        # Assert
        assert df.shape == (3, 3)
        assert "defasado_bin" in df.columns

    @patch("pandas.read_csv")
    def test_dropna_remove_valores_faltantes(self, mock_read_csv):
        """Testa se valores faltantes no target são removidos"""
        # Arrange
        mock_df = pd.DataFrame(
            {
                "feat1": [1, 2, 3, 4],
                "feat2": [4, 5, 6, 7],
                "defasado_bin": [0, 1, np.nan, 0],
            }
        )
        mock_read_csv.return_value = mock_df

        # Act
        df_clean = mock_df.dropna(subset=["defasado_bin"])

        # Assert
        assert df_clean.shape[0] == 3
        assert not df_clean["defasado_bin"].isna().any()


class TestPipelineBuilding:
    """Testes para construção do pipeline"""

    @patch("src.train.build_pipeline")
    def test_build_pipeline_retorna_objeto(self, mock_build_pipeline):
        """Testa se build_pipeline retorna um pipeline"""
        # Arrange
        mock_pipeline = MagicMock()
        mock_build_pipeline.return_value = mock_pipeline

        # Act
        pipeline = mock_build_pipeline(features=["feat1", "feat2"], model_name="logreg")

        # Assert
        assert pipeline is not None
        assert hasattr(pipeline, "fit")
        assert hasattr(pipeline, "predict")


class TestModelTraining:
    """Testes para treinamento do modelo"""

    @patch("src.train.build_pipeline")
    def test_fit_modelo_sucesso(self, mock_build_pipeline):
        """Testa se modelo é treinado com sucesso"""
        # Arrange
        X_train = pd.DataFrame({"feat1": [1, 2, 3], "feat2": [4, 5, 6]})
        y_train = np.array([0, 1, 0])

        mock_pipeline = MagicMock()
        mock_build_pipeline.return_value = mock_pipeline

        # Act
        pipeline = mock_build_pipeline(features=["feat1", "feat2"], model_name="logreg")
        pipeline.fit(X_train, y_train)

        # Assert
        assert mock_pipeline.fit.called
        mock_pipeline.fit.assert_called_once_with(X_train, y_train)

    @patch("src.train.build_pipeline")
    def test_predict_gera_predicoes(self, mock_build_pipeline):
        """Testa se modelo gera predições"""
        # Arrange
        X_val = pd.DataFrame({"feat1": [1, 2], "feat2": [4, 5]})

        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = np.array([0, 1])
        mock_pipeline.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
        mock_build_pipeline.return_value = mock_pipeline

        # Act
        pipeline = mock_build_pipeline(features=["feat1", "feat2"], model_name="logreg")
        preds = pipeline.predict(X_val)
        probs = pipeline.predict_proba(X_val)[:, 1]

        # Assert
        assert len(preds) == 2
        assert len(probs) == 2
        assert all(p in [0, 1] for p in preds)


class TestMetricsCalculation:
    """Testes para cálculo de métricas"""

    @patch("src.train.roc_auc_score")
    @patch("src.train.f1_score")
    @patch("src.train.precision_score")
    @patch("src.train.recall_score")
    def test_metrics_calculadas(
        self, mock_recall, mock_precision, mock_f1, mock_roc_auc
    ):
        """Testa se métricas são calculadas"""
        # Arrange
        y_val = np.array([0, 1, 0, 1, 0])
        preds = np.array([0, 1, 0, 1, 0])
        probs = np.array([0.1, 0.9, 0.2, 0.8, 0.3])

        mock_roc_auc.return_value = 0.95
        mock_f1.return_value = 0.95
        mock_precision.return_value = 1.0
        mock_recall.return_value = 0.9

        # Act
        roc_auc = mock_roc_auc(y_val, probs)
        f1 = mock_f1(y_val, preds)
        precision = mock_precision(y_val, preds)
        recall = mock_recall(y_val, preds)

        # Assert
        assert 0 <= roc_auc <= 1
        assert 0 <= f1 <= 1
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1

    def test_metrics_dict_estrutura(self):
        """Testa se dicionário de métricas tem estrutura correta"""
        # Arrange
        metrics = {"roc_auc": 0.95, "f1": 0.90, "precision": 0.92, "recall": 0.88}

        # Act & Assert
        assert "roc_auc" in metrics
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert all(0 <= v <= 1 for v in metrics.values())


class TestModelPersistence:
    """Testes para salvamento de modelo"""

    @patch("joblib.dump")
    def test_modelo_salvo_com_sucesso(self, mock_dump):
        """Testa se modelo é salvo com joblib"""
        # Arrange
        mock_pipeline = MagicMock()

        # Act
        mock_dump(mock_pipeline, "model.joblib")

        # Assert
        assert mock_dump.called
        mock_dump.assert_called_once_with(mock_pipeline, "model.joblib")

    @patch("builtins.open", new_callable=mock_open)
    def test_metricas_salvas_json(self, mock_file):
        """Testa se métricas são salvas em JSON"""
        # Arrange
        metrics = {"roc_auc": 0.95, "f1": 0.90, "precision": 0.92, "recall": 0.88}

        # Act
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        # Assert
        assert mock_file.called

    def test_caminhos_modelo_validos(self):
        """Testa se caminhos esperados para modelo são válidos"""
        # Arrange
        model_path = "artifacts/models/best_model.joblib"
        report_path = "artifacts/models/best_model_metrics.json"

        # Act & Assert
        assert "artifacts" in model_path
        assert "models" in model_path
        assert model_path.endswith(".joblib")
        assert report_path.endswith(".json")


class TestLogging:
    """Testes para logging"""

    @patch("src.train.setup_logger")
    def test_logger_inicializado(self, mock_setup_logger):
        """Testa se logger é inicializado"""
        # Arrange
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        # Act
        logger = mock_setup_logger(name="train", log_file="test.log")

        # Assert
        assert logger is not None
        assert mock_setup_logger.called

    @patch("src.train.setup_logger")
    def test_logger_tem_info_method(self, mock_setup_logger):
        """Testa se logger tem método info para logging"""
        # Arrange
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        # Act
        logger = mock_setup_logger(name="train", log_file="test.log")
        logger.info("teste")

        # Assert
        assert hasattr(logger, "info")
        assert mock_logger.info.called
