.PHONY: help install install-dev lint format test clean serve train-baseline evaluate-baseline evaluate-random train-itemknn evaluate-itemknn train-mf evaluate-mf train-bpr evaluate-bpr

help:
	@echo "Available commands:"
	@echo "  make install           : Install core dependencies"
	@echo "  make install-dev       : Install development dependencies"
	@echo "  make lint              : Run ruff check"
	@echo "  make format            : Run ruff format & fix"
	@echo "  make test              : Run unit and integration tests"
	@echo "  make train-baseline    : Train popularity baseline on MIND dataset"
	@echo "  make evaluate-baseline : Run multi-slice offline evaluation on dev impressions"
	@echo "  make evaluate-random   : Run random baseline evaluation on dev impressions"
	@echo "  make train-itemknn     : Train ItemKNN baseline on MIND dataset"
	@echo "  make evaluate-itemknn  : Run ItemKNN evaluation on dev impressions"
	@echo "  make train-mf          : Train Matrix Factorization model"
	@echo "  make evaluate-mf       : Evaluate Matrix Factorization model on dev impressions"
	@echo "  make train-bpr         : Train BPR model"
	@echo "  make evaluate-bpr      : Evaluate BPR model on dev impressions"
	@echo "  make serve             : Run FastAPI recommendation service"
	@echo "  make clean             : Clean bytecode and temporary cache files"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,deep]"

lint:
	ruff check .

format:
	ruff check --fix .
	ruff format .

test:
	pytest tests/

train-baseline:
	python -m src.train --config configs/experiment/exp01_mind_popularity.yaml

evaluate-baseline:
	python -m src.evaluate --config configs/experiment/exp01_mind_popularity.yaml

evaluate-random:
	python -m src.evaluate --config configs/experiment/exp00_mind_random.yaml

train-itemknn:
	python -m src.train --config configs/experiment/exp02_mind_itemknn.yaml

evaluate-itemknn:
	python -m src.evaluate --config configs/experiment/exp02_mind_itemknn.yaml

train-mf:
	python -m src.train --config configs/experiment/exp03_mind_mf.yaml

evaluate-mf:
	python -m src.evaluate --config configs/experiment/exp03_mind_mf.yaml

train-bpr:
	python -m src.train --config configs/experiment/exp04_mind_bpr.yaml

evaluate-bpr:
	python -m src.evaluate --config configs/experiment/exp04_mind_bpr.yaml


serve:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +

