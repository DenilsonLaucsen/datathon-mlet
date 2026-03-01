import platform
from pathlib import Path

import fastapi
import sklearn


def check_directories():
    required_dirs = ["data", "artifacts", "tests", "notebooks"]
    for d in required_dirs:
        if not Path(d).exists():
            print(f"[ERRO] Diretório ausente: {d}")
            return False
    print("[OK] Estrutura de diretórios válida")
    return True


def check_environment():
    print(f"Python version: {platform.python_version()}")
    print(f"FastAPI version: {fastapi.__version__}")
    print(f"Scikit-learn version: {sklearn.__version__}")
    return True


def run():
    print("=== SANITY CHECK ===")
    dir_ok = check_directories()
    env_ok = check_environment()

    if dir_ok and env_ok:
        print("Projeto pronto para desenvolvimento 🚀")
    else:
        print("Há problemas na configuração.")


if __name__ == "__main__":
    run()
