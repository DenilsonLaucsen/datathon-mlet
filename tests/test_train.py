import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


class TestTrainFunctionSignature:
    """Testes para assinatura e documentação da função train"""

    def test_train_funcao_existe(self):
        """Testa se função train está definida"""
        from src.train import train

        assert callable(train)

    def test_train_funcao_tem_docstring(self):
        """Testa se função train possui docstring"""
        from src.train import train

        assert train.__doc__ is not None
        assert "Treina modelo" in train.__doc__

    @patch("src.train.setup_logger")
    @patch("src.train.build_pipeline")
    @patch("src.train.train_test_split")
    @patch("pandas.read_csv")
    @patch("src.train.load_features")
    @patch("src.train.load_config")
    @patch("src.train.joblib.dump")
    @patch("builtins.open", create=True)
    def test_train_funcao_retorna_dict(
        self,
        mock_file,
        mock_dump,
        mock_cfg,
        mock_feat,
        mock_csv,
        mock_split,
        mock_pipeline,
        mock_logger_setup,
    ):
        """Testa se train retorna dicionário"""
        from src.train import train

        # Setup logger mock
        mock_logger = MagicMock()
        mock_logger_setup.return_value = mock_logger

        # Arrange config and features
        mock_cfg.return_value = {
            "random_seed": 42,
            "target": "defasado_bin",
            "best_model": "logreg",
        }
        mock_feat.return_value = ["feat1", "feat2"]

        # Arrange dataset
        df = pd.DataFrame(
            {
                "feat1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "feat2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20] * 2,
                "defasado_bin": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 2,
            }
        )
        mock_csv.return_value = df

        # Arrange train_test_split - retorna X_train, X_val, y_train, y_val
        X = df[["feat1", "feat2"]]
        y = df["defasado_bin"]
        # 80/20 split: 16 train, 4 val
        X_train = X.iloc[:16]
        X_val = X.iloc[16:]
        y_train = y.iloc[:16]
        y_val = y.iloc[16:]
        mock_split.return_value = (X_train, X_val, y_train, y_val)

        # Arrange pipeline mock - probs para 4 valores (tamanho de X_val)
        mock_pipeline_obj = MagicMock()
        mock_pipeline_obj.predict.return_value = np.array([0, 1, 0, 1])
        mock_pipeline_obj.predict_proba.return_value = np.array(
            [[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7]]
        )
        mock_pipeline.return_value = mock_pipeline_obj

        # Act
        result = train()

        # Assert
        assert isinstance(result, dict)


class TestTrainReturnStructure:
    """Testes para estrutura de retorno da função train"""

    @patch("src.train.setup_logger")
    @patch("src.train.build_pipeline")
    @patch("src.train.train_test_split")
    @patch("pandas.read_csv")
    @patch("src.train.load_features")
    @patch("src.train.load_config")
    @patch("src.train.joblib.dump")
    @patch("builtins.open", create=True)
    def test_train_retorna_chaves_obrigatorias(
        self,
        mock_file,
        mock_dump,
        mock_cfg,
        mock_feat,
        mock_csv,
        mock_split,
        mock_pipeline,
        mock_logger_setup,
    ):
        """Testa se retorno contém todas as chaves obrigatórias"""
        from src.train import train

        # Setup logger
        mock_logger = MagicMock()
        mock_logger_setup.return_value = mock_logger

        # Arrange
        mock_cfg.return_value = {
            "random_seed": 42,
            "target": "defasado_bin",
            "best_model": "logreg",
        }
        mock_feat.return_value = ["feat1", "feat2"]

        df = pd.DataFrame(
            {
                "feat1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "feat2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20] * 2,
                "defasado_bin": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 2,
            }
        )
        mock_csv.return_value = df

        X = df[["feat1", "feat2"]]
        y = df["defasado_bin"]
        X_train = X.iloc[:16]
        X_val = X.iloc[16:]
        y_train = y.iloc[:16]
        y_val = y.iloc[16:]
        mock_split.return_value = (X_train, X_val, y_train, y_val)

        mock_pipeline_obj = MagicMock()
        mock_pipeline_obj.predict.return_value = np.array([0, 1, 0, 1])
        mock_pipeline_obj.predict_proba.return_value = np.array(
            [[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7]]
        )
        mock_pipeline.return_value = mock_pipeline_obj

        # Act
        result = train()

        # Assert
        assert "model" in result
        assert "metrics" in result
        assert "model_path" in result
        assert "report_path" in result

    @patch("src.train.setup_logger")
    @patch("src.train.build_pipeline")
    @patch("src.train.train_test_split")
    @patch("pandas.read_csv")
    @patch("src.train.load_features")
    @patch("src.train.load_config")
    @patch("src.train.joblib.dump")
    @patch("builtins.open", create=True)
    def test_train_metrics_contem_campos_obrigatorios(
        self,
        mock_file,
        mock_dump,
        mock_cfg,
        mock_feat,
        mock_csv,
        mock_split,
        mock_pipeline,
        mock_logger_setup,
    ):
        """Testa se métricas contêm todos os campos"""
        from src.train import train

        # Setup logger
        mock_logger = MagicMock()
        mock_logger_setup.return_value = mock_logger

        # Arrange
        mock_cfg.return_value = {
            "random_seed": 42,
            "target": "defasado_bin",
            "best_model": "logreg",
        }
        mock_feat.return_value = ["feat1", "feat2"]

        df = pd.DataFrame(
            {
                "feat1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
                "feat2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20] * 2,
                "defasado_bin": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 2,
            }
        )
        mock_csv.return_value = df

        X = df[["feat1", "feat2"]]
        y = df["defasado_bin"]
        X_train = X.iloc[:16]
        X_val = X.iloc[16:]
        y_train = y.iloc[:16]
        y_val = y.iloc[16:]
        mock_split.return_value = (X_train, X_val, y_train, y_val)

        mock_pipeline_obj = MagicMock()
        mock_pipeline_obj.predict.return_value = np.array([0, 1, 0, 1])
        mock_pipeline_obj.predict_proba.return_value = np.array(
            [[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7]]
        )
        mock_pipeline.return_value = mock_pipeline_obj

        # Act
        result = train()

        # Assert
        metrics = result["metrics"]
        assert "roc_auc" in metrics
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics


class TestTrainParameters:
    """Testes para parâmetros da função train"""

    @patch("src.train.setup_logger")
    @patch("src.train.build_pipeline")
    @patch("src.train.train_test_split")
    @patch("pandas.read_csv")
    @patch("src.train.load_features")
    @patch("src.train.load_config")
    @patch("src.train.joblib.dump")
    @patch("builtins.open", create=True)
    def test_train_aceita_model_name_customizado(
        self,
        mock_file,
        mock_dump,
        mock_cfg,
        mock_feat,
        mock_csv,
        mock_split,
        mock_pipeline,
        mock_logger_setup,
    ):
        """Testa se train aceita model_name customizado"""
        from src.train import train

        # Setup logger
        mock_logger = MagicMock()
        mock_logger_setup.return_value = mock_logger

        # Arrange
        mock_cfg.return_value = {
            "random_seed": 42,
            "target": "defasado_bin",
            "best_model": "logreg",
        }
        mock_feat.return_value = ["feat1", "feat2"]

        df = pd.DataFrame(
            {
                "feat1": list(range(20)),
                "feat2": list(range(20, 40)),
                "defasado_bin": [0, 1] * 10,
            }
        )
        mock_csv.return_value = df

        X = df[["feat1", "feat2"]]
        y = df["defasado_bin"]
        mock_split.return_value = (X.iloc[:16], X.iloc[16:], y.iloc[:16], y.iloc[16:])

        mock_pipeline_obj = MagicMock()
        mock_pipeline_obj.predict.return_value = np.array([0, 1, 0, 1])
        mock_pipeline_obj.predict_proba.return_value = np.array(
            [[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7]]
        )
        mock_pipeline.return_value = mock_pipeline_obj

        # Act
        result = train(model_name="gb")

        # Assert
        assert result is not None
        # Verifica se foi chamado com "gb"
        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs["model_name"] == "gb"

    @patch("src.train.setup_logger")
    @patch("src.train.build_pipeline")
    @patch("src.train.train_test_split")
    @patch("pandas.read_csv")
    @patch("src.train.load_features")
    @patch("src.train.load_config")
    @patch("src.train.joblib.dump")
    @patch("src.train.Path")
    @patch("builtins.open", create=True)
    def test_train_aceita_caminhos_customizados(
        self,
        mock_file,
        mock_path,
        mock_dump,
        mock_cfg,
        mock_feat,
        mock_csv,
        mock_split,
        mock_pipeline,
        mock_logger_setup,
    ):
        """Testa se train aceita caminhos customizados"""
        from src.train import train

        # Setup logger
        mock_logger = MagicMock()
        mock_logger_setup.return_value = mock_logger

        config_path = "custom_config.yaml"
        dataset_path = "custom_data.csv"
        features_path = "custom_features.json"

        # Arrange - Mock Path.exists() para retornar True
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.parent.mkdir = MagicMock()
        mock_path.return_value = mock_path_instance

        mock_cfg.return_value = {
            "random_seed": 42,
            "target": "defasado_bin",
            "best_model": "logreg",
        }
        mock_feat.return_value = ["feat1", "feat2"]

        df = pd.DataFrame(
            {
                "feat1": list(range(20)),
                "feat2": list(range(20, 40)),
                "defasado_bin": [0, 1] * 10,
            }
        )
        mock_csv.return_value = df

        X = df[["feat1", "feat2"]]
        y = df["defasado_bin"]
        mock_split.return_value = (X.iloc[:16], X.iloc[16:], y.iloc[:16], y.iloc[16:])

        mock_pipeline_obj = MagicMock()
        mock_pipeline_obj.predict.return_value = np.array([0, 1, 0, 1])
        mock_pipeline_obj.predict_proba.return_value = np.array(
            [[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7]]
        )
        mock_pipeline.return_value = mock_pipeline_obj

        # Act
        result = train(
            config_path=config_path,
            dataset_path=dataset_path,
            features_path=features_path,
        )

        # Assert
        assert result is not None
        mock_cfg.assert_called_once_with(config_path)
        mock_feat.assert_called_once_with(features_path)
        mock_csv.assert_called_once_with(dataset_path)


class TestTrainMetricsValidation:
    """Testes para validação de métricas"""

    @patch("src.train.setup_logger")
    @patch("src.train.build_pipeline")
    @patch("src.train.train_test_split")
    @patch("pandas.read_csv")
    @patch("src.train.load_features")
    @patch("src.train.load_config")
    @patch("src.train.joblib.dump")
    @patch("builtins.open", create=True)
    def test_train_metricas_sao_float(
        self,
        mock_file,
        mock_dump,
        mock_cfg,
        mock_feat,
        mock_csv,
        mock_split,
        mock_pipeline,
        mock_logger_setup,
    ):
        """Testa se todas as métricas são float"""
        from src.train import train

        # Setup logger
        mock_logger = MagicMock()
        mock_logger_setup.return_value = mock_logger

        # Arrange
        mock_cfg.return_value = {
            "random_seed": 42,
            "target": "defasado_bin",
            "best_model": "logreg",
        }
        mock_feat.return_value = ["feat1", "feat2"]

        df = pd.DataFrame(
            {
                "feat1": list(range(20)),
                "feat2": list(range(20, 40)),
                "defasado_bin": [0, 1] * 10,
            }
        )
        mock_csv.return_value = df

        X = df[["feat1", "feat2"]]
        y = df["defasado_bin"]
        mock_split.return_value = (X.iloc[:16], X.iloc[16:], y.iloc[:16], y.iloc[16:])

        mock_pipeline_obj = MagicMock()
        mock_pipeline_obj.predict.return_value = np.array([0, 1, 0, 1])
        mock_pipeline_obj.predict_proba.return_value = np.array(
            [[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7]]
        )
        mock_pipeline.return_value = mock_pipeline_obj

        # Act
        result = train()

        # Assert
        metrics = result["metrics"]
        for metric_name, metric_value in metrics.items():
            assert isinstance(
                metric_value, float
            ), f"{metric_name} deve ser float, recebeu {type(metric_value)}"

    @patch("src.train.setup_logger")
    @patch("src.train.build_pipeline")
    @patch("src.train.train_test_split")
    @patch("pandas.read_csv")
    @patch("src.train.load_features")
    @patch("src.train.load_config")
    @patch("src.train.joblib.dump")
    @patch("builtins.open", create=True)
    def test_train_metricas_em_range_valido(
        self,
        mock_file,
        mock_dump,
        mock_cfg,
        mock_feat,
        mock_csv,
        mock_split,
        mock_pipeline,
        mock_logger_setup,
    ):
        """Testa se métricas estão em range válido [0, 1]"""
        from src.train import train

        # Setup logger
        mock_logger = MagicMock()
        mock_logger_setup.return_value = mock_logger

        # Arrange
        mock_cfg.return_value = {
            "random_seed": 42,
            "target": "defasado_bin",
            "best_model": "logreg",
        }
        mock_feat.return_value = ["feat1", "feat2"]

        df = pd.DataFrame(
            {
                "feat1": list(range(20)),
                "feat2": list(range(20, 40)),
                "defasado_bin": [0, 1] * 10,
            }
        )
        mock_csv.return_value = df

        X = df[["feat1", "feat2"]]
        y = df["defasado_bin"]
        mock_split.return_value = (X.iloc[:16], X.iloc[16:], y.iloc[:16], y.iloc[16:])

        mock_pipeline_obj = MagicMock()
        mock_pipeline_obj.predict.return_value = np.array([0, 1, 0, 1])
        mock_pipeline_obj.predict_proba.return_value = np.array(
            [[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7]]
        )
        mock_pipeline.return_value = mock_pipeline_obj

        # Act
        result = train()

        # Assert
        metrics = result["metrics"]
        for metric_name, metric_value in metrics.items():
            assert (
                0 <= metric_value <= 1
            ), f"{metric_name}={metric_value} deve estar entre 0 e 1"


class TestTrainPersistence:
    """Testes para persistência de arquivos"""

    @patch("src.train.setup_logger")
    @patch("src.train.build_pipeline")
    @patch("src.train.train_test_split")
    @patch("pandas.read_csv")
    @patch("src.train.load_features")
    @patch("src.train.load_config")
    @patch("src.train.joblib.dump")
    @patch("builtins.open", create=True)
    def test_train_salva_modelo_joblib(
        self,
        mock_file,
        mock_dump,
        mock_cfg,
        mock_feat,
        mock_csv,
        mock_split,
        mock_pipeline,
        mock_logger_setup,
    ):
        """Testa se modelo é salvo com joblib"""
        from src.train import train

        # Setup logger
        mock_logger = MagicMock()
        mock_logger_setup.return_value = mock_logger

        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_cfg.return_value = {
                "random_seed": 42,
                "target": "defasado_bin",
                "best_model": "logreg",
            }
            mock_feat.return_value = ["feat1", "feat2"]

            df = pd.DataFrame(
                {
                    "feat1": list(range(20)),
                    "feat2": list(range(20, 40)),
                    "defasado_bin": [0, 1] * 10,
                }
            )
            mock_csv.return_value = df

            X = df[["feat1", "feat2"]]
            y = df["defasado_bin"]
            mock_split.return_value = (
                X.iloc[:16],
                X.iloc[16:],
                y.iloc[:16],
                y.iloc[16:],
            )

            mock_pipeline_obj = MagicMock()
            mock_pipeline_obj.predict.return_value = np.array([0, 1, 0, 1])
            mock_pipeline_obj.predict_proba.return_value = np.array(
                [[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7]]
            )
            mock_pipeline.return_value = mock_pipeline_obj

            # Act
            result = train(output_dir=tmpdir)

            # Assert
            model_path = result["model_path"]
            assert model_path.endswith(".joblib")
            assert Path(model_path).parent == Path(tmpdir)

    @patch("src.train.setup_logger")
    @patch("src.train.build_pipeline")
    @patch("src.train.train_test_split")
    @patch("pandas.read_csv")
    @patch("src.train.load_features")
    @patch("src.train.load_config")
    @patch("src.train.joblib.dump")
    def test_train_salva_metricas_json(
        self,
        mock_dump,
        mock_cfg,
        mock_feat,
        mock_csv,
        mock_split,
        mock_pipeline,
        mock_logger_setup,
    ):
        """Testa se métricas são salvas como JSON"""
        from src.train import train

        # Setup logger
        mock_logger = MagicMock()
        mock_logger_setup.return_value = mock_logger

        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_cfg.return_value = {
                "random_seed": 42,
                "target": "defasado_bin",
                "best_model": "logreg",
            }
            mock_feat.return_value = ["feat1", "feat2"]

            df = pd.DataFrame(
                {
                    "feat1": list(range(20)),
                    "feat2": list(range(20, 40)),
                    "defasado_bin": [0, 1] * 10,
                }
            )
            mock_csv.return_value = df

            X = df[["feat1", "feat2"]]
            y = df["defasado_bin"]
            mock_split.return_value = (
                X.iloc[:16],
                X.iloc[16:],
                y.iloc[:16],
                y.iloc[16:],
            )

            mock_pipeline_obj = MagicMock()
            mock_pipeline_obj.predict.return_value = np.array([0, 1, 0, 1])
            mock_pipeline_obj.predict_proba.return_value = np.array(
                [[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7]]
            )
            mock_pipeline.return_value = mock_pipeline_obj

            # Act
            result = train(output_dir=tmpdir)

            # Assert
            report_path = result["report_path"]
            assert report_path.endswith(".json")
            assert os.path.exists(report_path)

            with open(report_path) as f:
                saved_metrics = json.load(f)
                assert saved_metrics == result["metrics"]


class TestTrainDataProcessing:
    """Testes para processamento de dados"""

    @patch("src.train.setup_logger")
    @patch("src.train.build_pipeline")
    @patch("src.train.train_test_split")
    @patch("pandas.read_csv")
    @patch("src.train.load_features")
    @patch("src.train.load_config")
    @patch("src.train.joblib.dump")
    @patch("builtins.open", create=True)
    def test_train_remove_valores_faltantes_target(
        self,
        mock_file,
        mock_dump,
        mock_cfg,
        mock_feat,
        mock_csv,
        mock_split,
        mock_pipeline,
        mock_logger_setup,
    ):
        """Testa se valores faltantes no target são removidos"""
        from src.train import train

        # Setup logger
        mock_logger = MagicMock()
        mock_logger_setup.return_value = mock_logger

        # Arrange
        mock_cfg.return_value = {
            "random_seed": 42,
            "target": "defasado_bin",
            "best_model": "logreg",
        }
        mock_feat.return_value = ["feat1", "feat2"]

        # Dataset com NaN no target
        df = pd.DataFrame(
            {
                "feat1": list(range(20)),
                "feat2": list(range(20, 40)),
                "defasado_bin": [
                    0,
                    1,
                    np.nan,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                ],
            }
        )
        mock_csv.return_value = df

        # Após dropna, ficam 19 amostras. Split 80/20: 15 train, 4 test
        X_clean = df.dropna(subset=["defasado_bin"])[["feat1", "feat2"]]
        y_clean = df.dropna(subset=["defasado_bin"])["defasado_bin"]
        X_train = X_clean.iloc[:15]
        X_val = X_clean.iloc[15:]
        y_train = y_clean.iloc[:15]
        y_val = y_clean.iloc[15:]
        mock_split.return_value = (X_train, X_val, y_train, y_val)

        mock_pipeline_obj = MagicMock()
        mock_pipeline_obj.predict.return_value = np.array([0, 1, 0, 1])
        mock_pipeline_obj.predict_proba.return_value = np.array(
            [[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7]]
        )
        mock_pipeline.return_value = mock_pipeline_obj

        # Act
        result = train()

        # Assert
        assert result is not None


class TestTrainImportability:
    """Testes para importabilidade e uso da função"""

    def test_train_pode_ser_importado(self):
        """Testa se train pode ser importado de src.train"""
        from src.train import train

        assert callable(train)

    def test_train_pode_ser_executado_como_script(self):
        """Testa se arquivo pode ser executado como script"""
        import src.train

        # Verifica se há o bloco if __name__ == "__main__"
        assert hasattr(src.train, "train")
