import json
import tempfile
from pathlib import Path

import pytest
import yaml

from src.utils import load_config, load_features


class TestLoadFeatures:
    """Testes para a função load_features"""

    def test_load_features_retorna_lista(self):
        """Testa se load_features retorna uma lista"""
        # Arrange
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"features": ["feat1", "feat2"]}, f)
            temp_path = f.name

        try:
            # Act
            features = load_features(temp_path)

            # Assert
            assert isinstance(features, list)
        finally:
            Path(temp_path).unlink()

    def test_load_features_conteudo_correto(self):
        """Testa se load_features carrega as features corretamente"""
        # Arrange
        features_esperadas = ["feature1", "feature2", "feature3"]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"features": features_esperadas}, f)
            temp_path = f.name

        try:
            # Act
            features = load_features(temp_path)

            # Assert
            assert features == features_esperadas
        finally:
            Path(temp_path).unlink()

    def test_load_features_lista_vazia(self):
        """Testa se load_features funciona com lista vazia"""
        # Arrange
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"features": []}, f)
            temp_path = f.name

        try:
            # Act
            features = load_features(temp_path)

            # Assert
            assert features == []
        finally:
            Path(temp_path).unlink()

    def test_load_features_arquivo_nao_existe(self):
        """Testa se load_features lança erro quando arquivo não existe"""
        # Arrange
        caminho_invalido = "/caminho/inexistente/features.json"

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            load_features(caminho_invalido)

    def test_load_features_json_invalido(self):
        """Testa se load_features lança erro com JSON inválido"""
        # Arrange
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("isso não é json válido {]")
            temp_path = f.name

        try:
            # Act & Assert
            with pytest.raises(json.JSONDecodeError):
                load_features(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_features_chave_inexistente(self):
        """Testa se load_features lança erro quando chave 'features' não existe"""
        # Arrange
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"outras_chaves": ["valor1"]}, f)
            temp_path = f.name

        try:
            # Act & Assert
            with pytest.raises(KeyError):
                load_features(temp_path)
        finally:
            Path(temp_path).unlink()


class TestLoadConfig:
    """Testes para a função load_config"""

    def test_load_config_retorna_dict(self):
        """Testa se load_config retorna um dicionário"""
        # Arrange
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"key": "value"}, f)
            temp_path = f.name

        try:
            # Act
            config = load_config(temp_path)

            # Assert
            assert isinstance(config, dict)
        finally:
            Path(temp_path).unlink()

    def test_load_config_conteudo_correto(self):
        """Testa se load_config carrega a configuração corretamente"""
        # Arrange
        config_esperada = {
            "database": "postgres",
            "port": 5432,
            "debug": True,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_esperada, f)
            temp_path = f.name

        try:
            # Act
            config = load_config(temp_path)

            # Assert
            assert config == config_esperada
        finally:
            Path(temp_path).unlink()

    def test_load_config_estrutura_aninhada(self):
        """Testa se load_config funciona com estrutura aninhada"""
        # Arrange
        config_esperada = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "credentials": {"user": "admin", "password": "secret"},
            },
            "debug": True,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_esperada, f)
            temp_path = f.name

        try:
            # Act
            config = load_config(temp_path)

            # Assert
            assert config == config_esperada
            assert config["database"]["credentials"]["user"] == "admin"
        finally:
            Path(temp_path).unlink()

    def test_load_config_vazio(self):
        """Testa se load_config funciona com arquivo YAML vazio"""
        # Arrange
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            temp_path = f.name

        try:
            # Act
            config = load_config(temp_path)

            # Assert
            assert config is None
        finally:
            Path(temp_path).unlink()

    def test_load_config_arquivo_nao_existe(self):
        """Testa se load_config lança erro quando arquivo não existe"""
        # Arrange
        caminho_invalido = "/caminho/inexistente/config.yaml"

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            load_config(caminho_invalido)

    def test_load_config_yaml_invalido(self):
        """Testa se load_config lança erro com YAML inválido"""
        # Arrange
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("key:\n  - invalid: yaml:\n    malformed")
            temp_path = f.name

        try:
            # Act & Assert
            with pytest.raises(yaml.YAMLError):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
