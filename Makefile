PYTHON=.venv/bin/python

setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -r requirements-dev.txt
	.venv/bin/pre-commit install

train:
	$(PYTHON) -m src.train

sanity:
	$(PYTHON) -m src.sanity_check

test:
	$(PYTHON) -m pytest tests/ --cov=. --cov-report=term-missing -v

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

IMAGE_NAME=datathon-ml-api
CONTAINER_NAME=datathon-ml-api

docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run:
	docker run -p 8000:8000 $(IMAGE_NAME)

docker-run-detached:
	docker run -d -p 8000:8000 --name $(CONTAINER_NAME) $(IMAGE_NAME)

docker-stop:
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true
