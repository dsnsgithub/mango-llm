# LLM

**Work in progress.**

Goal: Create a LLM that can produce English text using training (initially from stories, later web text).

## Requirements

- Python **3.13+**
- Dependencies: `torch`, `pandas` (see [`pyproject.toml`](pyproject.toml))

On Linux and Windows, `pyproject.toml` pins PyTorch to the **CUDA 12.4** wheel index where applicable; adjust or remove `[[tool.uv.index]]` / `[tool.uv.sources]` if you want CPU-only or a different CUDA build.

## Run

To train, download the required datasets from Kaggle:

Create directory `./dataset/TinyStories`: Download `train.csv` and `validation.csv` and place in this directory.
Link: https://www.kaggle.com/datasets/thedevastator/tinystories-narrative-classification/data

---

With [uv](https://docs.astral.sh/uv/) (recommended given the project config):

```bash
uv sync
uv run src/train.py
uv run src/run.py
```

Or with any environment where `torch` and `pandas` are installed:

```bash
uv run src/train.py
uv run src/run.py
```

## Layout

| Path | Role |
|------|------|
| `src/` | LLM code, main files being `src/run.py` and `src/train.py` |
| `dist/` | Contains trained LLM that can be run with `src/run.py` |
| `old/` | Older NumPy / PyTorch experiments (not required anymore) |
