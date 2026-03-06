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
- **scripts/** - Códigos auxiliares para preparar dados e treinar baseline
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

## Baseline Acadêmico

O projeto implementa um baseline de classification usando apenas variáveis de desempenho acadêmico para prever `defasado_bin` (se o aluno está defasado).

### Pipeline do Baseline

#### 1. Preparação de Dados
Executar o script de preparação:
```bash
python src/data/prepare_academic_dataset.py
```

**Responsabilidades:**
- Seleciona features acadêmicas relevantes: `Matematica`, `Portugues`, `Ingles`, `IDA`, `Cg`, `Cf`, `Ct`
- Remove colunas irrelevantes
- Define target como `defasado_bin`
- Salva processos em `data/processed/dataset_academic.csv`
- Exporta metadados em `artifacts/feature_cols.json`

#### 2. Treinamento
Treinar o modelo baseline:
```bash
python src/baseline/train_baseline.py
```

**Responsabilidades:**
- Pipeline com: Imputação → Normalização → Regressão Logística
- Validação cruzada estratificada (5 folds)
- Métricas: ROC-AUC, AUPRC, F1-Score
- Salva modelo em `artifacts/models/baseline_academico_model.joblib`
- Salva relatório em `artifacts/models/baseline_academico_report.json`

#### 3. Exploração e Análise
Notebook interativo com insights:
```bash
jupyter notebook notebooks/2_eda_baseline.ipynb
```

Demonstra:
- Distribuição do target
- Dados faltantes
- Análise de correlações
- Distribuições de features e relação com o target

### Objetivo

Estabelecer um **ponto de referência** (benchmark) para comparar modelos mais sofisticados que utilizem features adicionais ou técnicas avançadas (ensemble, deep learning, etc.).

## Experimentação de Modelos (Notebook 3)

O notebook `notebooks/3_model_experiments.ipynb` realiza uma **comparação experimental de 5 modelos diferentes**:

- Regressão Logística
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

### Fluxo

1. Carrega features e dataset processados
2. Realiza split treino/validação (80/20 com estratificação)
3. Treina todos os 5 modelos e avalia com métricas: Accuracy, Precision, Recall, F1-Score, ROC-AUC
4. Compara resultados e seleciona o melhor modelo (maior ROC-AUC)
5. Analisa importância das features para o melhor modelo
6. Salva o modelo final em `artifacts/models/model_experiment.joblib`

### Melhor Modelo Encontrado

Através deste notebook foi identificado que o **Gradient Boosting** é o melhor modelo, com:
- ROC-AUC: 0.749
- F1-Score: 0.746
- Recall: 0.804

As features mais importantes são: **IDA** (Indicador de Aprendizagem), **CG** (Classificação Geral), **CF** (Classificação na Fase) e **CT** (Classificação na Turma).

## Experimentos com Train Experiment

O script `train_experiment.py` permite treinar diferentes modelos com logging detalhado de todo o processo, facilitando a comparação de resultados entre experimentos.

### Uso

Para executar um experimento com um modelo específico:

```bash
python -m scripts.train_experiment <model_name>
```

**Exemplos:**

Treinar modelo de Regressão Logística:
```bash
python -m scripts.train_experiment logreg
```

Treinar modelo de Random Forest:
```bash
python -m scripts.train_experiment rf
```

### Fluxo do Experimento

1. **Validação de Argumentos**: Verifica se o modelo foi especificado
2. **Carregamento de Configurações**: Lê `config.yaml` para parâmetros (seed, n_splits, target)
3. **Carregamento de Features**: Lê features de `artifacts/feature_cols.json`
4. **Carregamento de Dados**: Carrega dataset processado ou consolidado
5. **Preparação do Target**: Cria coluna target se não existir
6. **Construção do Pipeline**: Monta pipeline com preprocessamento e modelo
7. **Validação Cruzada**: Executa estratificada com métricas ROC-AUC, AUPRC e F1-Score
8. **Treinamento Final**: Treina modelo com todos os dados
9. **Persistência**: Salva modelo e relatório com resultados

### Saídas do Experimento

Ao executar um experimento, os seguintes arquivos são gerados:

- **Modelo**: `artifacts/models/{model_name}_experiment_model.joblib`
- **Relatório**: `artifacts/models/{model_name}_experiment_report.json`
- **Logs**: `artifacts/logs/train_experiment_{timestamp}.log`

### Fluxo Completo: Do Notebook ao Modelo em Produção

1. **Notebook 3 (Experimentação)**: Testa 5 modelos e identifica o melhor
2. **Train Experiment Script**: Registra o melhor modelo encontrado com logs detalhados
   ```bash
   python -m scripts.train_experiment gb
   ```
3. **Resultado**: Modelo e relatório salvos em `artifacts/models/` para cada modelo testado

### Sistema de Logs

O projeto utiliza um sistema centralizado de logging para rastrear todos os eventos da execução.

#### Estrutura de Logs

Os logs são organizados em:
- **Diretório**: `artifacts/logs/`
- **Nomeação**: `train_experiment_{YYYY-MM-DD_HH-MM}.log`
- **Exemplo**: `train_experiment_2025-03-05_14-30.log`

#### Criação e Organização

1. **Inicialização**: Ao iniciar um script, um timestamp é gerado com formato `YYYY-MM-DD_HH-MM`
2. **Arquivo Log**: Um novo arquivo de log é criado com esse timestamp, garantindo unicidade
3. **Logger Configurado**: O logger é configurado pela função `setup_logger()` em `src/logger.py`
4. **Saída Mista**: Logs aparecem tanto no console quanto no arquivo
5. **Conteúdo**: Inclui informativos (info), erros (error) e avisos (warning)

#### Informações Registradas

Cada log captura:
- Configurações carregadas
- Número de features e linhas do dataset
- Etapas do pipeline
- Resultados da validação cruzada (métricas por fold)
- Caminhos de artefatos salvos
- Erros e exceções (com stack trace completo)

#### Exemplo de Consulta nos Logs

Para acompanhar um experimento em tempo real:
```bash
tail -f artifacts/logs/train_experiment_2025-03-05_14-30.log
```

Para buscar experimentos de um modelo específico:
```bash
grep -l "logreg\|rf" artifacts/logs/*.log
```

Para analisar erros em experimentos:
```bash
grep "ERROR" artifacts/logs/*.log
```

## Docker
