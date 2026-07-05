# BPE Tokenizer (from scratch)

A byte-level Byte Pair Encoding tokenizer built from scratch in a Jupyter notebook,
for learning how tokenization actually works. No libraries doing the heavy lifting —
just Python and the `regex` module for pre-tokenization.

Everything lives in [`tokenizer.ipynb`](tokenizer.ipynb), built up cell by cell.
See [`NOTES.md`](NOTES.md) for a step-by-step log of how it was built, with examples.

## What it does

- **Train** a vocabulary from text by repeatedly merging the most frequent byte pair.
- **Save / load** the trained model as a plain-text file (`tokenizer.model`).
- **Encode** text into token ids.
- **Decode** token ids back into text.
- Handles **any** input (emoji, accents, etc.) with no unknown tokens, because it
  works on raw UTF-8 bytes.
- **Pre-tokenization** (GPT-2 style regex split) so merges don't cross word boundaries.
- **Special tokens** like `<|endoftext|>` that get reserved ids and bypass BPE.

## How it works (the short version)

1. Text is converted to raw bytes (values 0–255) — the base vocabulary.
2. Training counts adjacent byte pairs, merges the most frequent one into a new id,
   and repeats until it reaches the target `vocab_size`.
3. The learned merges *are* the model. Encoding replays them in the order they were
   learned; decoding expands each id back to its bytes.

## Usage

Open the notebook and run the cells top to bottom. The main knob is:

```python
vocab_size = 276   # 256 base bytes + however many merges you want
```

The core functions:

```python
merges, vocab = ...                       # produced by the training cell
save("tokenizer.model", merges, pattern, special_tokens)
merges, vocab, pattern, special_tokens = load("tokenizer.model")

ids  = encode("some text", merges, special_tokens)
text = decode(ids, vocab, special_tokens)
```

## What this is (and isn't)

This is a **learning implementation** — small, readable, and correct, meant to make the
BPE algorithm easy to follow. It is not optimized for speed and isn't trained on a large
corpus, so it won't compress like a production tokenizer. But it's the same algorithm
GPT-2 and other real byte-level BPE tokenizers use.

## Requirements

- Python 3
- `regex` (`pip install regex`) — needed for the pre-tokenization pattern

## References

- Sebastian Raschka — *Byte Pair Encoding (BPE) From Scratch*:
  https://sebastianraschka.com/blog/2025/bpe-from-scratch.html
- Andrej Karpathy — *Let's build the GPT Tokenizer* (video):
  https://www.youtube.com/watch?v=zduSFxRajkE
