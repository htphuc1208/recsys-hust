.PHONY: help install install-dev lint format test clean serve serve-dev \
	train-baseline evaluate-baseline evaluate-random train-itemknn evaluate-itemknn \
	train-mf evaluate-mf train-bpr evaluate-bpr train-baselines train-proposed \
	build-serving-bundle evaluate evaluate-retrieval benchmark-serving

PROPOSED_CONFIG ?= configs/experiment/exp10_mind_hybrid.yaml
SERVING_CONFIG ?= configs/serving/mind_production.yaml
SERVING_CHECKPOINT ?= artifacts/checkpoints/exp10_mind_hybrid_hybrid.pkl
SERVING_BUNDLE ?= artifacts/serving/current
MODEL_VERSION ?= dev
CONTENT_DIM ?= 64
GIT_COMMIT ?= unknown

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
	@echo "  make train-baselines   : Train all implemented baseline models"
	@echo "  make train-proposed    : Train proposed model using PROPOSED_CONFIG"
	@echo "  make build-serving-bundle : Package checkpoint, mappings, popularity and FAISS"
	@echo "  make evaluate          : Evaluate proposed ranker and candidate retrieval"
	@echo "  make benchmark-serving : Benchmark a running recommendation API"
	@echo "  make serve             : Run production-style FastAPI serving"
	@echo "  make serve-dev         : Run FastAPI with auto-reload"
	@echo "  make clean             : Clean bytecode and temporary cache files"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,deep,serving]"

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

train-baselines: train-baseline train-itemknn train-mf train-bpr

train-proposed:
	python -m src.train --config $(PROPOSED_CONFIG)

build-serving-bundle:
	python -m src.indexing.build_bundle \
		--checkpoint $(SERVING_CHECKPOINT) \
		--users-mapping data/processed/mind_large/mappings/users \
		--items-mapping data/processed/mind_large/mappings/items \
		--output-dir $(SERVING_BUNDLE) \
		--model-version $(MODEL_VERSION) \
		--git-commit $(GIT_COMMIT) \
		--overwrite

evaluate:
	python -m src.evaluate --config $(PROPOSED_CONFIG)
	python -m src.indexing.evaluate_retrieval --config $(SERVING_CONFIG)

evaluate-retrieval:
	python -m src.indexing.evaluate_retrieval --config $(SERVING_CONFIG)

benchmark-serving:
	python -m src.serving.benchmark --config configs/benchmark/serving.yaml $(if $(SERVER_PID),--server-pid $(SERVER_PID),)

serve:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

serve-dev:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +

