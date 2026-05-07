# 🥭 Mango LLM
> **M**y **A**nswers **N**eed **G**enuine **O**versight => **MANGO**

### **Work in progress.**


Goal: Create a LLM that can produce coherent English text (initially trained on stories, later web text), in the process learning machine learning concepts.

As many layers/parts of the LLM as possible are built from basic components and individual matrix multiplications (instead of using pre-made components).

AI was used to assist/help me understand LLM concepts, but almost all of the code in this repo was handwritten and loosely based off of GPT-2 and the original Attention is All You Need paper.

## Requirements
- [uv](https://docs.astral.sh/uv/)

uv can automatically install the required Python version (even if you don't have Python installed), along with any required packages.

On Linux and Windows, this LLM uses the CUDA accelerator. On macOS, it uses the default MPS (Metal Performance Shaders) accelerator for Apple Silicon if possible.

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

## Layout

| Path | Role |
|------|------|
| `src/` | LLM code, main files being `src/run.py` and `src/train.py` |
| `dist/` | Contains trained LLM that can be run with `src/run.py` |
| `old/` | Contains original LLM, useful for beginners trying to understand the basics |
