"""
Byte-level BPE tokenizer — the class version of `tokenizer.ipynb`.

Same algorithm as the notebook (train / encode / decode), packaged as a reusable
class with JSON save/load and loguru logging to `log/tokenizer.log`.

    from bpe_tokenizer import BPETokenizer

    tok = BPETokenizer()
    tok.train(text, vocab_size=276, special_tokens=["<|endoftext|>"])
    ids  = tok.encode("some text")
    text = tok.decode(ids)
    tok.save("tokenizer.json")
    tok = BPETokenizer.load("tokenizer.json")
"""

import json
import sys
from pathlib import Path

import regex as re
from loguru import logger

# --- logging: concise on the console (INFO), everything to a dedicated file (DEBUG) ---
LOG_DIR = Path(__file__).resolve().parent / "log"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()  # drop loguru's default sink so we don't stack duplicates on re-import
logger.add(sys.stderr, level="INFO")
logger.add(LOG_DIR / "tokenizer.log", level="DEBUG", rotation="1 MB", encoding="utf-8")


def _bytes_to_unicode():
    """GPT-2's reversible map: every byte 0..255 -> its own distinct printable char.

    Used only for readable logging of merged pieces. Byte 32 (space) becomes 'Ġ'.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class BPETokenizer:
    """A byte-level Byte Pair Encoding tokenizer (GPT-2 / GPT-4 style)."""

    # GPT-2/GPT-4 pre-tokenization pattern: split text into word-ish chunks so BPE
    # merges never cross a chunk boundary.
    SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    _BYTE_ENCODER = _bytes_to_unicode()  # byte value -> visible char, for logging

    def __init__(self, pattern=None):
        self.pattern = pattern or self.SPLIT_PATTERN
        self._compiled = re.compile(self.pattern)
        self.merges = {}                               # (int, int) -> new_id
        self.vocab = {i: bytes([i]) for i in range(256)}  # id -> bytes it expands to
        self.special_tokens = {}                       # str -> int
        self._inverse_special = {}                     # int -> str

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _get_pair_counts(ids, counts=None):
        """Count adjacent pairs; pass `counts` to accumulate across chunks."""
        counts = {} if counts is None else counts
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def _merge(ids, pair, new_id):
        """Replace every occurrence of `pair` with the single token `new_id`."""
        merged, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                merged.append(new_id)
                i += 2
            else:
                merged.append(ids[i])
                i += 1
        return merged

    @classmethod
    def _render(cls, idx, vocab):
        """Visible form of a token, for logging."""
        return "".join(cls._BYTE_ENCODER[b] for b in vocab[idx])

    # ------------------------------------------------------------------- train
    def train(self, text, vocab_size, special_tokens=None):
        """Learn `vocab_size - 256` merges from `text` (byte-level, chunk-aware)."""
        assert vocab_size >= 256, "vocab_size must be at least 256 (the base byte count)"
        num_merges = vocab_size - 256
        logger.info("train: vocab_size={} ({} merges) on {} chars", vocab_size, num_merges, len(text))

        # pre-tokenize: each chunk is its own list of bytes; merges stay within a chunk
        chunk_ids = [list(ch.encode("utf-8")) for ch in self._compiled.findall(text)]
        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}

        for k in range(num_merges):
            counts = {}
            for chunk in chunk_ids:
                self._get_pair_counts(chunk, counts)
            if not counts:
                logger.warning("train: no pairs left after {} merges; stopping early", k)
                break
            top_pair = max(counts, key=counts.get)
            new_id = 256 + k
            chunk_ids = [self._merge(chunk, top_pair, new_id) for chunk in chunk_ids]
            merges[top_pair] = new_id
            vocab[new_id] = vocab[top_pair[0]] + vocab[top_pair[1]]
            logger.debug("merge {:>3}: {} -> {} ({}x) {!r}",
                         k + 1, top_pair, new_id, counts[top_pair], self._render(new_id, vocab))

        self.merges = merges
        self.vocab = vocab
        self.special_tokens = {}
        self._inverse_special = {}
        if special_tokens:
            self.register_special_tokens(special_tokens)
        logger.success("train: done — vocab={} (+{} special)", len(self.vocab), len(self.special_tokens))

    def register_special_tokens(self, names):
        """Reserve ids for special-token strings, above all byte + merge ids."""
        start = 256 + len(self.merges)
        for offset, name in enumerate(names):
            self.special_tokens[name] = start + offset
        self._inverse_special = {i: s for s, i in self.special_tokens.items()}
        logger.info("registered special tokens: {}", self.special_tokens)

    # ------------------------------------------------------------ encode/decode
    def _encode_chunk(self, text):
        """Byte-level BPE on one pre-token chunk: apply merges in rank order."""
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            counts = self._get_pair_counts(ids)
            pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = self._merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text, allowed_special=True):
        """text -> list of token ids."""
        if allowed_special and self.special_tokens:
            specials_re = "(" + "|".join(re.escape(s) for s in self.special_tokens) + ")"
            parts = re.split(specials_re, text)
        else:
            parts = [text]

        ids = []
        for part in parts:
            if allowed_special and part in self.special_tokens:
                ids.append(self.special_tokens[part])       # reserved id directly
            else:
                for chunk in self._compiled.findall(part):  # pre-tokenize
                    ids.extend(self._encode_chunk(chunk))   # then BPE within the chunk
        logger.debug("encode: {!r}{} -> {} ids", text[:40], "…" if len(text) > 40 else "", len(ids))
        return ids

    def decode(self, ids):
        """list of token ids -> text."""
        out, buf = [], []
        for idx in ids:
            if idx in self._inverse_special:
                if buf:                                     # flush pending bytes first
                    out.append(b"".join(buf).decode("utf-8", errors="replace"))
                    buf = []
                out.append(self._inverse_special[idx])
            elif idx in self.vocab:
                buf.append(self.vocab[idx])
            else:
                raise ValueError(f"decode: unknown token id {idx}")
        if buf:
            out.append(b"".join(buf).decode("utf-8", errors="replace"))
        text = "".join(out)
        logger.debug("decode: {} ids -> {!r}{}", len(ids), text[:40], "…" if len(text) > 40 else "")
        return text

    # ------------------------------------------------------------- save / load
    def save(self, path):
        """Persist the model as JSON. Merges are stored as an ordered list of
        pairs — the id each produces is implicit from its position (256 + index)."""
        data = {
            "version": "bpe-json-1",
            "pattern": self.pattern,
            "special_tokens": self.special_tokens,          # str -> int
            "merges": [[a, b] for (a, b) in self.merges],   # ordered; ids implicit
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("save: {} merges + {} special tokens -> {}",
                    len(self.merges), len(self.special_tokens), path)

    @classmethod
    def load(cls, path):
        """Rebuild a tokenizer from a JSON file written by `save`."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        tok = cls(pattern=data["pattern"])
        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}
        for k, (a, b) in enumerate(data["merges"]):        # replay in saved order
            new_id = 256 + k
            merges[(a, b)] = new_id
            vocab[new_id] = vocab[a] + vocab[b]
        tok.merges = merges
        tok.vocab = vocab
        tok.special_tokens = {s: int(i) for s, i in data["special_tokens"].items()}
        tok._inverse_special = {i: s for s, i in tok.special_tokens.items()}
        logger.info("load: {} merges + {} special tokens <- {}",
                    len(merges), len(tok.special_tokens), path)
        return tok


if __name__ == "__main__":
    training_text = (
        "Byte Pair Encoding is a simple tokenization algorithm. It starts from raw "
        "bytes and repeatedly merges the most frequent pair. Tokenizers like GPT-2's "
        "use byte-level BPE — even emoji 😀 and accented characters like café work, "
        "because everything is just bytes underneath."
    )

    tok = BPETokenizer()
    tok.train(training_text, vocab_size=276, special_tokens=["<|endoftext|>"])

    sample = "café bytes and emoji 😀 <|endoftext|>"
    ids = tok.encode(sample)
    print("ids:    ", ids)
    print("decoded:", tok.decode(ids))

    tok.save("tokenizer.json")
    reloaded = BPETokenizer.load("tokenizer.json")
    print("round-trip (loaded):", reloaded.decode(reloaded.encode(sample)) == sample)
