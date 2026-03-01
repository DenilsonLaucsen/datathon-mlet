PYTHON=.venv/bin/python

setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -r requirements-dev.txt
	.venv/bin/pre-commit install

sanity:
	$(PYTHON) -m src.sanity_check

test:
	$(PYTHON) -m pytest tests/ --cov=src --cov-report=term-missing -v

format:
	$(PYTHON) -m black app src tests
	$(PYTHON) -m isort app src tests

lint:
	$(PYTHON) -m flake8 app src tests

typecheck:
	$(PYTHON) -m mypy src/

run:
	$(PYTHON) -m uvicorn app.main:app --reload

run-prod:
	$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000
