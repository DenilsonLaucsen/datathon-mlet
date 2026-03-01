from pathlib import Path
from unittest.mock import patch

from src.sanity_check import check_directories, check_environment, run


class TestCheckDirectories:
    """Testes para a função check_directories"""

    def test_all_directories_exist(self, capsys):
        """Testa quando todos os diretórios obrigatórios existem"""
        with patch.object(Path, "exists", return_value=True):
            result = check_directories()
            assert result is True
            captured = capsys.readouterr()
            assert "[OK] Estrutura de diretórios válida" in captured.out

    def test_missing_directory(self, capsys):
        """Testa quando um diretório está faltando"""

        def exists_side_effect():
            # Simula que o primeiro diretório não existe
            return False

        with patch.object(Path, "exists", side_effect=exists_side_effect):
            result = check_directories()
            assert result is False
            captured = capsys.readouterr()
            assert "[ERRO] Diretório ausente:" in captured.out


class TestCheckEnvironment:
    """Testes para a função check_environment"""

    def test_check_environment_returns_true(self):
        """Testa se check_environment retorna True"""
        result = check_environment()
        assert result is True

    def test_python_version_printed(self, capsys):
        """Testa se versão do Python é exibida"""
        check_environment()
        captured = capsys.readouterr()
        assert "Python version:" in captured.out

    def test_fastapi_version_printed(self, capsys):
        """Testa se versão do FastAPI é exibida"""
        check_environment()
        captured = capsys.readouterr()
        assert "FastAPI version:" in captured.out

    def test_sklearn_version_printed(self, capsys):
        """Testa se versão do Scikit-learn é exibida"""
        check_environment()
        captured = capsys.readouterr()
        assert "Scikit-learn version:" in captured.out


class TestRun:
    """Testes para a função run"""

    def test_run_success(self, capsys):
        """Testa run quando tudo está OK"""
        with patch.object(Path, "exists", return_value=True):
            run()
            captured = capsys.readouterr()
            assert "=== SANITY CHECK ===" in captured.out
            assert "Projeto pronto para desenvolvimento" in captured.out

    def test_run_with_errors(self, capsys):
        """Testa run quando há erros na configuração"""
        with patch.object(Path, "exists", return_value=False):
            run()
            captured = capsys.readouterr()
            assert "[ERRO] Diretório ausente:" in captured.out
            assert "Há problemas na configuração." in captured.out

    def test_run_prints_header(self, capsys):
        """Testa se o header do sanity check é exibido"""
        with patch.object(Path, "exists", return_value=True):
            run()
            captured = capsys.readouterr()
            assert "=== SANITY CHECK ===" in captured.out
