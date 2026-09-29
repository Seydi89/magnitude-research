"""An offline lexical/latent baseline and an optional pinned pretrained encoder."""
import re
from pathlib import Path
import numpy as np


def normalized(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


class LSAEncoder:
    def __init__(self, dimensions=128, seed=2026):
        self.dimensions, self.seed = dimensions, seed
        self.metadata = dict(name="TF-IDF + truncated SVD (LSA)", pretrained=False,
                             note="Lexical/latent baseline; not pretrained semantic embeddings.",
                             token_unit="regex word-or-punctuation tokens (not LLM tokens)")

    def fit(self, corpus):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=40000,
                                         sublinear_tf=True, strip_accents="unicode")
        x = self.vectorizer.fit_transform(corpus)
        dimensions = min(self.dimensions, x.shape[0] - 1, x.shape[1] - 1)
        if dimensions < 2:
            raise ValueError("Need a larger corpus for LSA")
        self.svd = TruncatedSVD(n_components=dimensions, n_iter=7, random_state=self.seed)
        self.svd.fit(x)
        self.metadata.update(dimensions=dimensions, fitted_on="unlabeled static corpus only",
                             seed=self.seed, explained_variance=float(self.svd.explained_variance_ratio_.sum()))

    def encode(self, texts):
        return normalized(self.svd.transform(self.vectorizer.transform(texts))).astype(np.float32)

    def count(self, text):
        return max(1, len(re.findall(r"\w+|[^\w\s]", text)))


class MiniLMEncoder:
    def __init__(self, model_dir=None,
                 revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41", threads=2):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        from .data import digest
        if model_dir:
            root = Path(model_dir)
            token_path, model_path = root / "tokenizer.json", root / "onnx/model.onnx"
        else:
            from huggingface_hub import hf_hub_download
            name = "sentence-transformers/all-MiniLM-L6-v2"
            token_path = Path(hf_hub_download(name, "tokenizer.json", revision=revision))
            model_path = Path(hf_hub_download(name, "onnx/model.onnx", revision=revision))
        self.tokenizer = Tokenizer.from_file(str(token_path))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        self.cls, self.sep, self.pad = [self.tokenizer.token_to_id(t) for t in ["[CLS]", "[SEP]", "[PAD]"]]
        if None in [self.cls, self.sep, self.pad]:
            raise ValueError("This adapter requires the MiniLM BERT tokenizer")
        self.metadata = dict(name="all-MiniLM-L6-v2 ONNX", pretrained=True, revision=revision,
                             model_sha256=digest(model_path), tokenizer_sha256=digest(token_path),
                             token_unit="MiniLM WordPiece tokens, exact stored-memory payload",
                             long_text="Nonoverlapping 254-content-token windows; attention-masked token mean over all windows; L2 normalization")

    def fit(self, corpus):
        pass

    def count(self, text):
        return max(1, len(self.tokenizer.encode(text, add_special_tokens=False).ids))

    def encode(self, texts):
        windows = []
        for i, text in enumerate(texts):
            ids = self.tokenizer.encode(text, add_special_tokens=False).ids
            for start in range(0, max(1, len(ids)), 254):
                windows.append((i, [self.cls] + ids[start:start + 254] + [self.sep]))
        windows.sort(key=lambda p: len(p[1]))
        sums, counts = None, np.zeros(len(texts))
        input_names = {i.name for i in self.session.get_inputs()}
        for offset in range(0, len(windows), 32):
            batch = windows[offset:offset + 32]
            width = max(len(ids) for _, ids in batch)
            ids = np.full((len(batch), width), self.pad, dtype=np.int64)
            mask = np.zeros_like(ids)
            for j, (_, values) in enumerate(batch):
                ids[j, :len(values)] = values
                mask[j, :len(values)] = 1
            inputs = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
            output = self.session.run(None, {k: v for k, v in inputs.items() if k in input_names})[0]
            if output.ndim != 3:
                raise ValueError("Expected token-level ONNX output")
            pooled = (output.astype(np.float64) * mask[:, :, None]).sum(axis=1)
            if sums is None:
                sums = np.zeros((len(texts), output.shape[-1]))
            for j, (i, _) in enumerate(batch):
                sums[i] += pooled[j]
                counts[i] += mask[j].sum()
            if offset % 1024 == 0:
                print(f"Encoded {offset}/{len(windows)} windows", flush=True)
        return normalized(sums / counts[:, None]).astype(np.float32)
