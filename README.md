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

### 2. Prepare the MINDlarge data from ZIP files

Dataset files are intentionally not stored in GitHub. Obtain the official
`MINDlarge_train.zip`, `MINDlarge_dev.zip`, and `MINDlarge_test.zip` files, then
place and extract them into the following layout:

```text
data/raw/mind_large/
├── archives/                     # Optional: keep the original ZIP files here
│   ├── MINDlarge_train.zip
│   ├── MINDlarge_dev.zip
│   └── MINDlarge_test.zip
├── MINDlarge_train/
│   ├── behaviors.tsv
│   └── news.tsv
├── MINDlarge_dev/
│   ├── behaviors.tsv
│   └── news.tsv
└── MINDlarge_test/
    ├── behaviors.tsv
    └── news.tsv
```

PowerShell (Windows):

```powershell
New-Item -ItemType Directory -Force data/raw/mind_large/archives | Out-Null
# Copy the downloaded ZIP files into data/raw/mind_large/archives first.

Expand-Archive data/raw/mind_large/archives/MINDlarge_train.zip data/raw/mind_large/MINDlarge_train
Expand-Archive data/raw/mind_large/archives/MINDlarge_dev.zip data/raw/mind_large/MINDlarge_dev
Expand-Archive data/raw/mind_large/archives/MINDlarge_test.zip data/raw/mind_large/MINDlarge_test
```

Linux/macOS:

```bash
mkdir -p data/raw/mind_large/{archives,MINDlarge_train,MINDlarge_dev,MINDlarge_test}
# Copy the downloaded ZIP files into data/raw/mind_large/archives first.

unzip data/raw/mind_large/archives/MINDlarge_train.zip -d data/raw/mind_large/MINDlarge_train
unzip data/raw/mind_large/archives/MINDlarge_dev.zip -d data/raw/mind_large/MINDlarge_dev
unzip data/raw/mind_large/archives/MINDlarge_test.zip -d data/raw/mind_large/MINDlarge_test
```

Each extracted directory may also contain the embedding files supplied by MIND;
the current ingestion step requires `news.tsv` and `behaviors.tsv`. Convert all
available splits to chunked Parquet files with:

```bash
python -m src.ingest.mind --split all
```

MIND covers six weeks (12 October–22 November 2019), but the first four weeks
are represented through each user's click `history`, not as separate training
impressions. The released impression timeline is:

```text
12 Oct–08 Nov: click history used for user modeling
09 Nov–14 Nov: train impressions
15 Nov:        dev impressions (last day of week 5)
16 Nov–22 Nov: test impressions
```

See the [official MIND dataset introduction](https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md)
for the collection and split methodology.

The generated files are written to `data/interim/mind_large/`. If you only have
one ZIP file, extract it into its matching directory and replace `all` with
`train`, `dev`, or `test`. Add `--overwrite` when intentionally rebuilding an
existing output.

### 3. Explore the dataset (EDA)

After ingestion, generate summary tables and charts for every split:

```bash
python -m src.analysis.mind_eda --split all
```

Results are saved in `reports/mind_eda/` as `summary.json`, `summary.csv`, and
PNG charts for dataset size, history/candidate distributions, top news
categories, and impressions over time. For a quick smoke run before analyzing
the full MINDlarge dataset, limit each table to 100,000 rows:

```bash
python -m src.analysis.mind_eda --split all --max-rows 100000
```

When `--max-rows` is used, the JSON report marks the result as sampled. Run the
command without this option for the complete statistics. Use `--output-dir` to
choose another report location.

### 4. Build train-ready features

Convert the interim Parquet files into encoded model inputs:

```bash
python -m src.features.prepare_mind --split all
```

This writes the following leakage-safe outputs to `data/processed/mind_large/`:

```text
data/processed/mind_large/
├── manifest.json
├── mappings/
│   ├── users/                    # user_id -> user_idx, fitted on train only
│   └── items/                    # All catalog IDs, with seen_in_train flags
├── train/
│   ├── news/
│   ├── impressions/              # Encoded histories and ranking candidates
│   ├── positive_interactions/    # Clicks for popularity/CF models
│   └── pointwise/                # Positives plus sampled negatives
├── dev/
│   ├── news/
│   ├── impressions/
│   └── positive_interactions/
└── test/
    ├── news/
    └── impressions/              # No labels in the official MIND test split
```

By default, train pointwise data contains up to four negatives per positive.
Change this with `--negative-ratio`, for example:

```bash
python -m src.features.prepare_mind --negative-ratio 2 --overwrite
```

Users are fitted from train behaviors; a user first observed later receives
index `-1`. News IDs and metadata are not labels, so every available catalog is
indexed and each article gets a distinct `item_idx`. The `seen_in_train` fields
and `*_seen_in_train_mask` lists identify cold-start articles for collaborative
models. An item receives `-1` only if it is referenced by a behavior row but is
missing from every available news catalog. The generated `manifest.json`
reports both unmapped and unseen-in-train rates.

All files under `data/raw/`, `data/interim/`, `data/processed/`, and
`data/splits/`, as well as common archive formats, are ignored by Git. You can
confirm this before committing with `git status`.

### 5. Run Tests

```bash
make test
```

### 6. Run Recommendation Serving API

```bash
make serve
# API will be available at http://localhost:8000
# Interactive Swagger docs: http://localhost:8000/docs
```
