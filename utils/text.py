"""IMDB text pipeline for Question 3. Provided so you can spend your time on the models.

Tokenising, vocabulary building and padding are plumbing, not learning outcomes. What you
have to understand is why `lengths` travels alongside the padded batch - see the
TODO(student) blocks in starter/q3_rnn_lstm.py.

    from utils.text import ImdbDataset, build_vocab, collate_pad
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset

PAD, UNK = 0, 1
TOKEN_RE = re.compile(r"[a-z0-9']+")


def simple_tokenize(text: str) -> list:
    """Lowercase, drop the stray HTML line breaks IMDB contains, keep word-ish tokens."""
    text = text.lower().replace("<br />", " ")
    return TOKEN_RE.findall(text)


def read_imdb_split(root, split: str) -> list:
    """Return [(text, label), ...] with label 1 = positive, 0 = negative.

    Expects the layout the download script produces: data/aclImdb/{train,test}/{pos,neg}/*.txt
    """
    root = Path(root)
    base = root / "aclImdb" / split
    if not base.is_dir():
        raise SystemExit(
            "{} not found. Run: python setup/download_datasets.py --imdb".format(base))
    rows = []
    for label_name, label in (("pos", 1), ("neg", 0)):
        for path in sorted((base / label_name).glob("*.txt")):
            rows.append((path.read_text(encoding="utf-8", errors="replace"), label))
    return rows


def build_vocab(texts, max_size: int = 20000, min_freq: int = 2) -> dict:
    """Token -> id, built from the TRAINING split only.

    Ids 0 and 1 are reserved for <pad> and <unk>. Building the vocabulary on the test split
    as well would leak information; the starter never does it and neither should you.
    """
    counter = Counter()
    for text in texts:
        counter.update(simple_tokenize(text))
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for token, freq in counter.most_common():
        if freq < min_freq or len(vocab) >= max_size:
            break
        vocab[token] = len(vocab)
    return vocab


def encode(text: str, vocab: dict, max_len: int) -> list:
    """Tokens -> ids, truncated to max_len. Unknown tokens become <unk>."""
    ids = [vocab.get(tok, UNK) for tok in simple_tokenize(text)]
    return ids[:max_len] if max_len else ids


class ImdbDataset(Dataset):
    """IMDB reviews as (ids, length, label).

    Build the vocabulary once on the training split and pass it to both splits:

        train_rows = read_imdb_split("data", "train")
        vocab = build_vocab([t for t, _ in train_rows], max_size=20000)
        train_set = ImdbDataset(train_rows, vocab, max_len=200)
    """

    def __init__(self, rows, vocab: dict, max_len: int = 200) -> None:
        self.vocab = vocab
        self.max_len = max_len
        self.samples = []
        for text, label in rows:
            ids = encode(text, vocab, max_len)
            if not ids:                      # a review with no in-vocabulary tokens
                ids = [UNK]
            self.samples.append((ids, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        ids, label = self.samples[idx]
        return torch.tensor(ids, dtype=torch.long), len(ids), label

    def length_stats(self) -> dict:
        lengths = [len(ids) for ids, _ in self.samples]
        lengths_sorted = sorted(lengths)
        n = len(lengths_sorted)
        return {
            "n": n,
            "mean_len": round(sum(lengths) / n, 1),
            "median_len": lengths_sorted[n // 2],
            "p90_len": lengths_sorted[int(0.9 * n)],
            "max_len_seen": lengths_sorted[-1],
            "truncated_fraction": round(
                sum(1 for v in lengths if v >= self.max_len) / n, 4) if self.max_len else 0.0,
        }

    def oov_rate(self, rows, sample: int = 2000) -> float:
        """Fraction of tokens mapped to <unk>. Report this in task 3.1."""
        total = unknown = 0
        for text, _ in rows[:sample]:
            for tok in simple_tokenize(text):
                total += 1
                unknown += int(tok not in self.vocab)
        return round(unknown / max(total, 1), 4)


def collate_pad(batch):
    """Pad a batch to its longest sequence.

    Returns (tokens[B, T] padded with <pad>, lengths[B] on CPU, labels[B]).

    `lengths` must stay on the CPU - that is what pack_padded_sequence expects - and it is
    sorted nowhere, so pass enforce_sorted=False when you pack.
    """
    batch = sorted(batch, key=lambda item: item[1], reverse=True)
    sequences, lengths, labels = zip(*batch)
    max_len = max(lengths)
    padded = torch.full((len(sequences), max_len), PAD, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, : len(seq)] = seq
    return padded, torch.tensor(lengths, dtype=torch.long), torch.tensor(labels,
                                                                         dtype=torch.long)
