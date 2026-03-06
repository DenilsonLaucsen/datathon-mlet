import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessing import build_preprocessor


class TestBuildPreprocessor:
    """Testes para a função build_preprocessor"""

    def test_build_preprocessor_retorna_column_transformer(self):
        """Testa se build_preprocessor retorna um ColumnTransformer"""
        # Arrange
        features = ["feature1", "feature2"]

        # Act
        preprocessor = build_preprocessor(features)

        # Assert
        assert isinstance(preprocessor, ColumnTransformer)

    def test_build_preprocessor_tem_numeric_transformer(self):
        """Testa se o preprocessor possui o transformer 'numeric'"""
        # Arrange
        features = ["feature1", "feature2"]
        X = pd.DataFrame({"feature1": [1.0, 3.0], "feature2": [2.0, 4.0]})

        # Act
        preprocessor = build_preprocessor(features)
        preprocessor.fit(X)

        # Assert
        assert "numeric" in preprocessor.named_transformers_

    def test_build_preprocessor_numeric_pipeline_correto(self):
        """Testa se o numeric transformer é uma Pipeline com imputer e scaler"""
        # Arrange
        features = ["feature1", "feature2"]
        X = pd.DataFrame({"feature1": [1.0, 3.0], "feature2": [2.0, 4.0]})

        # Act
        preprocessor = build_preprocessor(features)
        preprocessor.fit(X)
        numeric_pipeline = preprocessor.named_transformers_["numeric"]

        # Assert
        assert isinstance(numeric_pipeline, Pipeline)
        assert "imputer" in numeric_pipeline.named_steps
        assert "scaler" in numeric_pipeline.named_steps

    def test_build_preprocessor_imputer_strategy_median(self):
        """Testa se o imputer usa estratégia 'median'"""
        # Arrange
        features = ["feature1", "feature2"]
        X = pd.DataFrame({"feature1": [1.0, 3.0], "feature2": [2.0, 4.0]})

        # Act
        preprocessor = build_preprocessor(features)
        preprocessor.fit(X)
        numeric_pipeline = preprocessor.named_transformers_["numeric"]
        imputer = numeric_pipeline.named_steps["imputer"]

        # Assert
        assert isinstance(imputer, SimpleImputer)
        assert imputer.strategy == "median"

    def test_build_preprocessor_scaler_correto(self):
        """Testa se o scaler é StandardScaler"""
        # Arrange
        features = ["feature1", "feature2"]
        X = pd.DataFrame({"feature1": [1.0, 3.0], "feature2": [2.0, 4.0]})

        # Act
        preprocessor = build_preprocessor(features)
        preprocessor.fit(X)
        numeric_pipeline = preprocessor.named_transformers_["numeric"]
        scaler = numeric_pipeline.named_steps["scaler"]

        # Assert
        assert isinstance(scaler, StandardScaler)

    def test_build_preprocessor_pipeline_ordem(self):
        """Testa se os steps estão na ordem correta (imputer depois scaler)"""
        # Arrange
        features = ["feature1", "feature2"]
        X = pd.DataFrame({"feature1": [1.0, 3.0], "feature2": [2.0, 4.0]})

        # Act
        preprocessor = build_preprocessor(features)
        preprocessor.fit(X)
        numeric_pipeline = preprocessor.named_transformers_["numeric"]

        # Assert
        step_names = [name for name, _ in numeric_pipeline.steps]
        assert step_names[0] == "imputer"
        assert step_names[1] == "scaler"

    def test_build_preprocessor_com_multiplas_features(self):
        """Testa se build_preprocessor funciona com múltiplas features"""
        # Arrange
        features = ["feat1", "feat2", "feat3", "feat4", "feat5"]

        # Act
        preprocessor = build_preprocessor(features)

        # Assert
        assert isinstance(preprocessor, ColumnTransformer)
        # Verifica se as features estão configuradas
        assert preprocessor.transformers[0][2] == features

    def test_build_preprocessor_feature_unica(self):
        """Testa se build_preprocessor funciona com uma única feature"""
        # Arrange
        features = ["feature1"]

        # Act
        preprocessor = build_preprocessor(features)

        # Assert
        assert isinstance(preprocessor, ColumnTransformer)

    def test_build_preprocessor_processa_dados(self):
        """Testa se o preprocessor processa dados corretamente"""
        # Arrange
        features = ["feature1", "feature2"]
        preprocessor = build_preprocessor(features)
        X = pd.DataFrame({"feature1": [1.0, 3.0, 5.0], "feature2": [2.0, 4.0, 6.0]})

        # Act
        X_transformed = preprocessor.fit_transform(X)

        # Assert
        assert X_transformed.shape == (3, 2)
        assert isinstance(X_transformed, np.ndarray)

    def test_build_preprocessor_imputa_valores_faltantes(self):
        """Testa se o preprocessor imputa valores faltantes com mediana"""
        # Arrange
        features = ["feature1", "feature2"]
        preprocessor = build_preprocessor(features)
        X = pd.DataFrame({"feature1": [1.0, np.nan, 5.0], "feature2": [2.0, 4.0, 6.0]})

        # Act
        X_transformed = preprocessor.fit_transform(X)

        # Assert
        assert not np.isnan(X_transformed).any()

    def test_build_preprocessor_normaliza_dados(self):
        """Testa se o preprocessor normaliza os dados com StandardScaler"""
        # Arrange
        features = ["feature1", "feature2"]
        preprocessor = build_preprocessor(features)
        X = pd.DataFrame({"feature1": [1.0, 3.0, 5.0], "feature2": [2.0, 4.0, 6.0]})

        # Act
        X_transformed = preprocessor.fit_transform(X)

        # Assert
        # StandardScaler deve fazer com que a média seja próxima de 0
        assert np.abs(np.mean(X_transformed, axis=0)).max() < 1e-10
        # E o desvio padrão seja próximo de 1
        assert np.abs(np.std(X_transformed, axis=0) - 1.0).max() < 1e-10
