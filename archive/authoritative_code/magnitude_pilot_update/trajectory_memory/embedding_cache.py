"""Persistent, text-addressed embeddings. SQLite commits each completed batch.

No evidence labels, session IDs, budgets or question types enter cache keys.
The encoder identity includes the actual model/tokenizer or fitted LSA state.
"""
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np


DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "synthetic_memory_pilot" / "cache"


class CachedEncoder:
    def __init__(self, encoder, cache_dir=DEFAULT_CACHE_DIR, batch_size=64, verbose=True):
        if batch_size < 1:
            raise ValueError("cache batch size must be positive")
        self.encoder, self.batch_size, self.verbose = encoder, batch_size, verbose
        self.metadata = encoder.metadata
        identity = dict(schema=1, **encoder.cache_identity)
        self.identity_json = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.namespace = hashlib.sha256(self.identity_json.encode()).hexdigest()
        root = Path(cache_dir).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "embeddings.sqlite3"
        self.db = sqlite3.connect(self.path, timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS encoders (
                namespace TEXT PRIMARY KEY, identity TEXT NOT NULL, dimensions INTEGER
            );
            CREATE TABLE IF NOT EXISTS embeddings (
                namespace TEXT NOT NULL, text_hash TEXT NOT NULL, text TEXT NOT NULL,
                vector BLOB NOT NULL, PRIMARY KEY(namespace, text_hash)
            );
        """)
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO encoders(namespace, identity) VALUES (?, ?)",
                            (self.namespace, self.identity_json))
        identity_stored, self.dimensions = self.db.execute(
            "SELECT identity, dimensions FROM encoders WHERE namespace=?", (self.namespace,)).fetchone()
        if identity_stored != self.identity_json:
            self.close()
            raise ValueError("Cache encoder identity mismatch")
        self.requested = self.reused = self.encoded = 0

    def count(self, text):
        return self.encoder.count(text)

    def stats(self):
        return dict(path=str(self.path), namespace=self.namespace, requested_texts=self.requested,
                    reused_texts=self.reused, newly_encoded_texts=self.encoded,
                    batch_size=self.batch_size)

    def encode(self, texts):
        self.dimensions = self.db.execute("SELECT dimensions FROM encoders WHERE namespace=?",
                                          (self.namespace,)).fetchone()[0]
        texts = list(texts)
        if not all(isinstance(t, str) for t in texts):
            raise TypeError("Embedding inputs must be strings")
        if not texts:
            return np.empty((0, self.dimensions or 0), dtype=np.float32)
        # Preserve exact text, caller order and multiplicity; only inference is deduplicated.
        unique = list(dict.fromkeys(texts))
        keys = {t: hashlib.sha256(t.encode("utf-8")).hexdigest() for t in unique}
        found, missing = {}, []
        for text in unique:
            row = self.db.execute("SELECT text, vector FROM embeddings WHERE namespace=? AND text_hash=?",
                                  (self.namespace, keys[text])).fetchone()
            if row is None:
                missing.append(text)
                continue
            if row[0] != text:
                raise ValueError("Cache text-hash collision")
            value = np.frombuffer(row[1], dtype="<f4")
            if len(value) != self.dimensions or not np.isfinite(value).all():
                raise ValueError("Invalid cached vector; use a new --cache-dir to rebuild")
            found[text] = value
        self.requested += len(texts)
        self.reused += len(texts) - len(missing)
        if self.verbose:
            print(f"Embedding cache: {len(texts)} requested, {len(unique)} unique, "
                  f"{len(unique) - len(missing)} already stored, {len(missing)} to encode", flush=True)
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start:start + self.batch_size]
            vectors = np.asarray(self.encoder.encode(batch), dtype="<f4")
            if vectors.ndim != 2 or vectors.shape[0] != len(batch) or vectors.shape[1] < 1 or not np.isfinite(vectors).all():
                raise ValueError("Encoder returned invalid vectors")
            width = vectors.shape[1]
            # Inference is outside the transaction; concurrent processes never hold a
            # write lock while doing model work. Each batch is all-or-nothing.
            with self.db:
                existing_width = self.db.execute("SELECT dimensions FROM encoders WHERE namespace=?",
                                                 (self.namespace,)).fetchone()[0]
                if existing_width is not None and existing_width != width:
                    raise ValueError("Encoder dimension changed inside one cache namespace")
                self.db.execute("UPDATE encoders SET dimensions=? WHERE namespace=?", (width, self.namespace))
                self.db.executemany("INSERT OR IGNORE INTO embeddings VALUES (?, ?, ?, ?)",
                                    [(self.namespace, keys[t], t, v.tobytes()) for t, v in zip(batch, vectors)])
            self.dimensions = width
            # Read the committed value in case another writer saved the same text.
            for text in batch:
                found[text] = np.frombuffer(self.db.execute(
                    "SELECT vector FROM embeddings WHERE namespace=? AND text_hash=?",
                    (self.namespace, keys[text])).fetchone()[0], dtype="<f4")
            self.encoded += len(batch)
            if self.verbose:
                print(f"  Saved embeddings: {start + len(batch)}/{len(missing)} new texts", flush=True)
        return np.stack([found[t] for t in texts])

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
