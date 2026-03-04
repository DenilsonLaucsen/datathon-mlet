# DATATHON-MLET

Projeto desenvolvido para o Datathon com foco em boas práticas de Engenharia de Machine Learning.

## Objetivo

Construir uma pipeline completa de ML, incluindo:
- ETL
- Feature Engineering
- Treinamento de modelo
- Avaliação
- API para predição
- Dockerização
- Testes automatizados
- Monitoramento

## Dataset - Projeto Defasagem

### Origem
BASE DE DADOS PEDE 2024 - DATATHON.xlsx

### Estrutura
- **3030 registros**
- **50 colunas**
- Dados longitudinais (2022–2024)

### Target
`defasado_bin` (variável binária derivada de `Defasagem < 0`)

A variável `Defasagem` indica a diferença entre a fase ideal e a fase real do aluno. Transformada em variável binária para interpretabilidade e aplicabilidade prática, indicando se o aluno está ou não defasado. A distribuição é aproximadamente balanceada (55% vs 45%).

### Problemas conhecidos
- Colunas com alto missing (>70%)
- Colunas 100% vazias removidas
- Algumas variáveis duplicadas
- Necessário remover variáveis com colinearidade para evitar vazamento de informação

Detalhes completos da exploração estão em [notebooks/1_data_overview.ipynb](notebooks/1_data_overview.ipynb).

## Estrutura de Pastas

- **app/** - Aplicação FastAPI com rotas e modelo
- **src/** - Código principal para processamento, treinamento e avaliação
- **data/** - Dados brutos e processados
- **artifacts/** - Modelos treinados e artefatos
- **notebooks/** - Notebooks exploratórios e análises
- **tests/** - Testes automatizados

## Pré-requisitos

- Python 3.9+
- pip ou conda

## Setup

### Usando Make (recomendado)
```bash
make setup
```

Isso irá:
- Criar ambiente virtual Python
- Atualizar pip
- Instalar dependências de produção (`requirements.txt`)
- Instalar dependências de desenvolvimento (`requirements-dev.txt`)
- Configurar git hooks com pre-commit

### Manual
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install
```

## Sanity Check

Validar que o projeto está bem configurado:
```bash
make sanity
```

Isso valida:
- Estrutura de diretórios
- Dependências principais
- Versões do ambiente

## Testes

Executar suite de testes com cobertura:
```bash
make test
```

Ou manualmente:
```bash
source .venv/bin/activate
pytest tests/ --cov=src --cov-report=term-missing -v
```

Para ver relatório de cobertura em HTML:
```bash
source .venv/bin/activate
pytest tests/ --cov=src --cov-report=html
```

## Qualidade de Código

### Formatação

Formatar código com Black e isort:
```bash
make format
```

### Linting

Verificar código com Flake8:
```bash
make lint
```

### Type Checking

Verificar tipos com mypy:
```bash
make typecheck
```

## API

### Desenvolvimento

Rodar API em modo desenvolvimento com reload automático:
```bash
make run
```

Ou manualmente:
```bash
source .venv/bin/activate
python -m uvicorn app.main:app --reload
```

### Produção

Rodar API em modo produção:
```bash
make run-prod
```

Ou manualmente:
```bash
source .venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Endpoints

- **Health Check**: `GET /health`
- **Documentação interativa**: [http://localhost:8000/docs](http://localhost:8000/docs)

## Docker
