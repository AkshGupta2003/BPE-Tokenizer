# BPE Tokenizer (from scratch)

A byte-level Byte Pair Encoding tokenizer built from scratch in a Jupyter notebook,
for learning how tokenization actually works. No libraries doing the heavy lifting —
just Python and the `regex` module for pre-tokenization.

The reusable implementation is [`bpe_tokenizer.py`](bpe_tokenizer.py). It was worked
out cell by cell first in [`tokenizer.ipynb`](tokenizer.ipynb).

Consumed by the companion data pipeline repo:
[LLM-Data-Pipeline](https://github.com/AkshGupta2003/LLM-Data-Pipeline).

## What it does

- **Train** a vocabulary from text by repeatedly merging the most frequent byte pair.
- **Save / load** the trained model as JSON (`tokenizer.json`).
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

```python
from bpe_tokenizer import BPETokenizer

tok = BPETokenizer()
tok.train(text, vocab_size=1000, special_tokens=["<|endoftext|>"])

ids  = tok.encode("some text")
text = tok.decode(ids)

tok.save("tokenizer.json")
tok = BPETokenizer.load("tokenizer.json")
```

The main knob is `vocab_size`: 256 base bytes plus however many merges you want.

**One gotcha.** Special tokens are numbered *above* all byte and merge ids, so a
`vocab_size=1000` model with one special token has **1001** distinct ids. Derive the
count from the tokenizer rather than hardcoding it — an embedding table sized 1000
will crash the moment `<|endoftext|>` appears:

```python
vocab_size = len(tok.vocab) + len(tok.special_tokens)
```

## What this is (and isn't)

This is a **learning implementation** — small, readable, and correct, meant to make the
BPE algorithm easy to follow. It is not optimized for speed and isn't trained on a large
corpus, so it won't compress like a production tokenizer. But it's the same algorithm
GPT-2 and other real byte-level BPE tokenizers use.

## Requirements

- Python 3.11+
- `regex` — needed for the pre-tokenization pattern
- `loguru` — logging to `log/tokenizer.log`

```bash
pip install regex loguru
```

## References

- Sebastian Raschka — *Byte Pair Encoding (BPE) From Scratch*:
  https://sebastianraschka.com/blog/2025/bpe-from-scratch.html
- Andrej Karpathy — *Let's build the GPT Tokenizer* (video):
  https://www.youtube.com/watch?v=zduSFxRajkE
