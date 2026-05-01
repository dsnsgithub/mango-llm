# LLM

**Work in progress.** Goal: grow this into a **full transformer LLM** in PyTorch (training, inference, sensible tokenizer and architecture), then **scale exposure to the web** (API, hosting, and whatever serving stack fits). The repository will evolve toward that; the notes below describe what exists today.


### What it does today

- Loads `roneneldan/TinyStories` (`train[:10]`) via [Hugging Face `datasets`](https://huggingface.co/docs/datasets).
- Builds a word-level vocabulary with a small regex tokenizer (`list_tokens`).
- Defines an `Embedding` layer, a stub `Attention` module, and an `LLM` that sums token and position embeddings. Transformer layers are currently an empty `ModuleList`.
- Runs a fixed two-token input `["this", "was"]` and prints the resulting tensor.

## Requirements

- Python **3.13+**
- Dependencies: `torch`, `datasets` (see [`pyproject.toml`](pyproject.toml))

On Linux and Windows, `pyproject.toml` pins PyTorch to the **CUDA 12.4** wheel index where applicable; adjust or remove `[[tool.uv.index]]` / `[tool.uv.sources]` if you want CPU-only or a different CUDA build.

## Run

With [uv](https://docs.astral.sh/uv/) (recommended given the project config):

```bash
uv sync
uv run python main.py
```

Or with any environment where `torch` and `datasets` are installed:

```bash
python main.py
```

## Layout

| Path | Role |
|------|------|
| `main.py` | Current PyTorch experiment |
| `numpy-old/` | Older NumPy / PyTorch experiments (not required by `main.py`) |
