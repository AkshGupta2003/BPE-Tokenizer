# BPE Tokenizer — Build Notes

A running log of how we built the tokenizer in [`tokenizer.ipynb`](tokenizer.ipynb),
step by step, with short explanations and examples. Meant as a refresher if I come
back to this later.

The order below is the order the cells were built in — each step assumes the ones
before it.

---

## Step 1 — Text to UTF-8 bytes

The base of everything. Take the training text and turn it into a list of raw byte
values (integers 0–255).

```python
tokens = list(training_text.encode("utf-8"))
```

- `.encode("utf-8")` gives a `bytes` object; iterating it yields integers, so
  `list()` gives the raw byte values.
- **Key idea:** the base vocabulary is just the 256 possible byte values. *Any* text —
  emoji, accents — decomposes into these, so there's no such thing as an unknown token.
- Multi-byte characters make the byte count larger than the character count. `é` is 2
  bytes, `😀` is 4.

---

## Step 2 — Count adjacent pairs

Count how often each neighbouring pair of tokens appears.

```python
def get_pair_counts(tokens, counts=None):
    counts = {} if counts is None else counts
    for pair in zip(tokens, tokens[1:]):   # every adjacent pair, one pass
        counts[pair] = counts.get(pair, 0) + 1
    return counts
```

- `zip(tokens, tokens[1:])` is the trick — it pairs each element with its neighbour.
- The optional `counts` accumulator was added later (Step 6) so we could tally pairs
  across many chunks without them bleeding together.
- **Example:** in `"...s ..."`, the pair `(115, 32)` = `s` + space shows up a lot.

---

## Step 3 — Merge a pair

Replace every occurrence of a chosen pair with a single new token id (starting at 256,
the first free id past the byte range).

```python
def merge(tokens, pair, new_id):
    merged, i = [], 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
            merged.append(new_id); i += 2   # matched the pair, skip both
        else:
            merged.append(tokens[i]); i += 1
    return merged
```

- Walks with an index because a merge consumes **two** elements at once.
- Each occurrence merged removes one element, so the sequence gets shorter.

---

## Step 4 — Readable logging (the GPT-2 byte→char table)

Raw bytes print badly (spaces are invisible, multi-byte chars show as garble). GPT-2
solves this with a reversible map from every byte to a distinct *printable* character.

```python
byte_encoder = bytes_to_unicode()   # 0..255 -> a visible char; space (32) -> 'Ġ'
```

- Bytes that are already printable map to themselves; the rest (space, control chars)
  get bumped to fresh code points, so nothing ever renders blank.
- Space becomes `Ġ`, which is why merged pieces log as e.g. `Ġbyte` (space + "byte").
- This replaced an earlier hand-rolled `render` helper that only handled the space case
  and crashed on merged ids.

---

## Step 5 — The training loop

Put it together: repeatedly count pairs, merge the most frequent one, until we hit the
target vocab size.

```python
vocab_size = 276                 # 256 base bytes + 20 merges
num_merges = vocab_size - 256

merges = {}                                  # (int,int) -> new_id   <- the model
vocab  = {i: bytes([i]) for i in range(256)} # id -> the bytes it expands to

for k in range(num_merges):
    counts   = get_pair_counts(...)
    top_pair = max(counts, key=counts.get)   # most frequent pair
    new_id   = 256 + k
    ...merge it everywhere...
    merges[top_pair] = new_id
    vocab[new_id]    = vocab[top_pair[0]] + vocab[top_pair[1]]
```

- Two artifacts come out of this and *are* the model:
  - `merges` — the ordered rules (which pair became which id). Insertion order matters.
  - `vocab` — the inverse, id → bytes, used for decoding.
- **Example run:** `te` → `Ġa` → `Ġbyte` → ... merges climb from letter-pairs into
  whole fragments.

---

## Step 6 — Pre-tokenization (regex split)

Before Step 6, BPE ran over one continuous byte stream, so merges could cross word and
even sentence boundaries (we saw `ingĠisĠ` — spanning "ing", a space, and "is"). Real
GPT-2 first splits text into word-ish chunks with a regex, then runs BPE *inside* each
chunk.

```python
import regex as re     # NOT stdlib 're' — needs \p{L}, \p{N}
SPLIT_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
```

- Training and encoding both now work chunk-by-chunk; a merge can never span two chunks.
- **Effect:** merges became sensible word-starts like `Ġbyte`, `Ġis` — the space
  attaches to the start of a word, never bleeding across into the next one.
- This is why `get_pair_counts` gained the accumulator (Step 2) — to count across all
  chunks at once.

---

## Step 7 — Special tokens

Tokens like `<|endoftext|>` that should get a reserved id and bypass BPE entirely.

```python
special_tokens = {"<|endoftext|>": vocab_size}   # id above all byte + merge ids
```

- On **encode**, the text is first split on the special-token strings; those emit their
  reserved id directly, everything else goes through pre-tokenization + BPE.
- On **decode**, a reserved id emits its literal string back.
- **Note:** special tokens are a lookup on literal marker strings, layered on top of
  BPE — never something the merge algorithm invents on its own. The marker must be
  present in the text *and* you must pass the `special_tokens` dict.

---

## Step 8 — Save / load (GPT-2 style)

Persist the model as a plain-text file so it can be reused without retraining.

```
bpe v2
<regex pattern>
<num special tokens>
<token> <id>
<num merges>
<a> <b>          # one line per merge, in learned order
```

- **Key insight:** we store only the *pairs*, not the id each merge produces — the id is
  implicit from line order (256 for the first line, 257 for the next, ...). The ordered
  list of merges *is* the model.
- `load()` rebuilds `vocab` by replaying merges in order; this works because when you
  process line `k`, both ids in that pair already exist (base bytes or earlier merges).
- Started as `bpe v1` (merges only); bumped to `bpe v2` when we added the pattern and
  special tokens.

---

## Step 9 — Encode and decode

The pieces that make it a usable tokenizer rather than just a trainer.

**Decode (ids → text)** — the easy direction, pure lookup:

```python
data = b"".join(vocab[idx] for idx in ids)
return data.decode("utf-8", errors="replace")
```

**Encode (text → ids)** — the interesting one. A *different* algorithm from training:
training merges by **frequency**; encoding merges by **rank** (earliest-learned first).

```python
pair = min(counts, key=lambda p: merges.get(p, float("inf")))
```

- Read as: "of the pairs currently present, take the one with the smallest merge rank."
  Pairs we never learned get `inf` and are skipped; if the best candidate isn't a learned
  merge, stop.
- Applying merges in learned order is what makes new text tokenize *consistently* with
  training.
- **Example:** `"bytes "` → bytes `[98,121,116,101,115,32]` → apply `yt`, then `byte`,
  then `s ` → `[265, 256]`.

---

## What "done" looks like

The checks cell confirms, all `True`:

- `encode(training_text)` reproduces the exact ids the trainer produced.
- `decode(encode(x)) == x` for the training text and for unseen text (emoji + accents).
- Special tokens survive a round-trip and get their reserved id.
- The same works with a model reloaded from disk.

The two algorithms to keep straight:

| | merges by | question it asks |
|---|---|---|
| **Training** | frequency | "what's the most common pair right now?" (builds the rulebook) |
| **Encoding** | rank | "what's the earliest rule I can apply right now?" (follows the rulebook) |

Decoding needs no rulebook logic at all — it's pure lookup.

---

## References

- Sebastian Raschka — *Byte Pair Encoding (BPE) From Scratch*:
  https://sebastianraschka.com/blog/2025/bpe-from-scratch.html
- Andrej Karpathy — *Let's build the GPT Tokenizer* (video):
  https://www.youtube.com/watch?v=zduSFxRajkE
