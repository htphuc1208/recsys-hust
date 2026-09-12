# recsys-hust

A modular, reproducible, and production-ready Recommender System framework designed for academic research and production deployment.

## Features

- **Multi-Stage Pipeline**: Decoupled ingestion, feature engineering, candidate generation (retrieval), scoring (ranking), and diversity reranking.
- **Leakage-Free Splitting**: Global temporal and user-level leave-last-k-out splitters to prevent temporal data leakage.
- **Unified Model Abstraction**: Standardized `BaseRecommender` interface supporting baselines, collaborative filtering, sequential architectures (SASRec/GRU4Rec), and two-tower embeddings.
- **Rigorous Offline Evaluation**: Full top-$K$ ranking metrics (NDCG, Recall, HitRate, MRR) and beyond-accuracy metrics (Coverage, Gini inequality, Novelty).
- **Production Serving**: Fast, asynchronous REST API via FastAPI with Pydantic request/response validation and Docker support.

---

## Directory Structure

```text
recsys-hust/
├── configs/                      # Hierarchical YAML experiment configs
├── data/                         # Datasets (raw, interim, processed, splits)
├── src/
│   ├── common/                   # Global logger, seed, and file I/O utilities
│   ├── ingest/                   # Raw data downloaders and parsers
│   ├── features/                 # ID encoders, sparse matrices, sequence builders
│   ├── splits/                   # Temporal & leave-k-out splitting strategies
│   ├── models/                   # Recommender architectures (baselines, CF, sequential, hybrid)
│   ├── evaluation/               # Ranking & beyond-accuracy metrics engine
│   ├── reranking/                # MMR, DPP diversity, and business constraint filters
│   ├── indexing/                 # FAISS vector indexing for ANN candidate retrieval
│   ├── serving/                  # FastAPI inference service & schemas
│   ├── train.py                  # CLI training entrypoint
│   └── evaluate.py               # CLI evaluation entrypoint
├── notebooks/                    # Exploratory analysis and benchmark plots
├── tests/                        # Unit and integration test suite
├── artifacts/                    # Model weights and FAISS vector index dumps
├── reports/                      # Visualizations and metric tables
├── Dockerfile                    # Containerization image
├── docker-compose.yml            # Deployment stack
├── Makefile                      # CLI shortcuts
└── pyproject.toml                # Project packaging & dependencies
```

---

## Quickstart

### 1. Installation

```bash
# Clone and enter the repository
git clone https://github.com/htphuc1208/recsys-hust.git
cd recsys-hust

# Install dependencies in editable mode
make install-dev
```

### 2. Run Tests

```bash
make test
```

### 3. Run Recommendation Serving API

```bash
make serve
# API will be available at http://localhost:8000
# Interactive Swagger docs: http://localhost:8000/docs
```
