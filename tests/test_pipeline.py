import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.pipeline import build_pipeline


class TestBuildPipeline:
    """Testes para a função build_pipeline"""

    def test_build_pipeline_retorna_pipeline(self):
        """Testa se build_pipeline retorna um objeto Pipeline"""
        # Arrange
        features = ["feature1", "feature2"]
        model_name = "logreg"

        # Act
        pipeline = build_pipeline(features, model_name)

        # Assert
        assert isinstance(pipeline, Pipeline)

    def test_build_pipeline_tem_preprocessador(self):
        """Testa se o pipeline possui o step 'preprocessor'"""
        # Arrange
        features = ["feature1", "feature2"]
        model_name = "logreg"

        # Act
        pipeline = build_pipeline(features, model_name)

        # Assert
        assert "preprocessor" in pipeline.named_steps
        assert pipeline.named_steps["preprocessor"] is not None

    def test_build_pipeline_tem_modelo(self):
        """Testa se o pipeline possui o step 'model'"""
        # Arrange
        features = ["feature1", "feature2"]
        model_name = "logreg"

        # Act
        pipeline = build_pipeline(features, model_name)

        # Assert
        assert "model" in pipeline.named_steps
        assert pipeline.named_steps["model"] is not None

    def test_build_pipeline_com_logreg(self):
        """Testa se build_pipeline cria pipeline com LogisticRegression"""
        # Arrange
        features = ["feature1", "feature2"]
        model_name = "logreg"

        # Act
        pipeline = build_pipeline(features, model_name)

        # Assert
        assert isinstance(pipeline.named_steps["model"], LogisticRegression)

    def test_build_pipeline_com_rf(self):
        """Testa se build_pipeline cria pipeline com RandomForestClassifier"""
        # Arrange
        features = ["feature1", "feature2"]
        model_name = "rf"

        # Act
        pipeline = build_pipeline(features, model_name)

        # Assert
        assert isinstance(pipeline.named_steps["model"], RandomForestClassifier)

    def test_build_pipeline_ordem_steps(self):
        """Testa se os steps estão na ordem correta"""
        # Arrange
        features = ["feature1", "feature2"]
        model_name = "logreg"

        # Act
        pipeline = build_pipeline(features, model_name)

        # Assert
        step_names = [name for name, _ in pipeline.steps]
        assert step_names[0] == "preprocessor"
        assert step_names[1] == "model"

    def test_build_pipeline_multiplas_features(self):
        """Testa se build_pipeline funciona com múltiplas features"""
        # Arrange
        features = ["feat1", "feat2", "feat3", "feat4", "feat5"]
        model_name = "logreg"

        # Act
        pipeline = build_pipeline(features, model_name)

        # Assert
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) == 2

    def test_build_pipeline_features_vazias(self):
        """Testa se build_pipeline funciona com lista vazia de features"""
        # Arrange
        features = []
        model_name = "logreg"

        # Act
        pipeline = build_pipeline(features, model_name)

        # Assert
        assert isinstance(pipeline, Pipeline)

    def test_build_pipeline_modelo_invalido(self):
        """Testa se build_pipeline lança erro com modelo inválido"""
        # Arrange
        features = ["feature1", "feature2"]
        model_name = "modelo_invalido"

        # Act & Assert
        with pytest.raises(ValueError, match="Modelo não suportado"):
            build_pipeline(features, model_name)
