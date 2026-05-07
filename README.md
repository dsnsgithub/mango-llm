# LLM

### **Work in progress.**

Goal: Create a LLM that can produce coherent English text (initially trained on stories, later web text), in the process learning machine learning concepts.

Each layer is built from basic components and fundamental matrix operations (instead of using pre-made components).

AI was used to assist/help me understand LLM concepts, but almost all of the code in this repo (apart from some parts of the README) was handwritten and loosely based off of GPT-2 and the original Attention is All You Need paper.

## Requirements

- Python **3.13+**
- Dependencies: `torch`, `pandas` (see [`pyproject.toml`](pyproject.toml))

On Linux and Windows, `pyproject.toml` pins PyTorch to **CUDA 12.4**. On macOS, it uses the default MPS (Metal Performance Shaders) accelerator for Apple Silicon if possible. 

## Run

To train, download the required datasets from Kaggle:

Download `train.csv` and `validation.csv` and create/place the files in `./dataset/TinyStories`.

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
python src/train.py
python src/run.py
```

## Layout

| Path | Role |
|------|------|
| `src/` | LLM code, main files being `src/run.py` and `src/train.py` |
| `dist/` | Contains trained LLM that can be run with `src/run.py` |
| `old/` | Contains original LLM, useful for beginners trying to understand the basics |
