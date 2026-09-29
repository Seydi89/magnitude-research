"""Public inputs and evaluation annotations are deliberately stored separately."""
import hashlib
import json
import re
import urllib.request
import zipfile
from pathlib import Path
import numpy as np

SOURCES = {
    "topiocqa.zip": "https://zenodo.org/records/6193037/files/topiocqa.zip?download=1",
    "passages_dev.json": "https://zenodo.org/records/6151011/files/data/retriever/all_history/dev.json",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def new_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Use a new empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def fetch(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for filename, url in SOURCES.items():
        target = out / filename
        if not target.exists():
            print(f"Downloading {filename}", flush=True)
            temporary = target.with_suffix(target.suffix + ".partial")
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
                while block := response.read(1 << 20):
                    stream.write(block)
            temporary.rename(target)
    dump(out / "sources.json", {k: dict(url=v, sha256=digest(out / k)) for k, v in SOURCES.items()})


def normalize_text(text):
    return " ".join(re.findall(r"\w+", text.lower()))


def public_query(row):
    """This function cannot copy current answers, topics, rationales or passages."""
    context = row["Context"]
    if len(context) % 2:
        raise ValueError("Expected alternating previous questions and answers")
    turn = int(row["Turn_no"])
    if len(context) != 2 * (turn - 1):
        raise ValueError("Non-prefix context; refusing possible temporal misalignment")
    return dict(id=f"dev:{row['Conversation_no']}:{turn}", conversation=str(row["Conversation_no"]),
                turn=turn, question=row["Question"],
                history=[dict(turn=i // 2 + 1, question=context[i], answer=context[i + 1])
                         for i in range(0, len(context), 2)])


def payload(memory):
    return f"[{memory['id']}] {memory['title']}\n{memory['text']}\n"


def query_texts(query, history_turns=6):
    history = query["history"][-history_turns:]
    qa = [f"Question: {h['question']} Answer: {h['answer']}" for h in history]
    joined = "\n".join(qa + [f"Current question: {query['question']}"])
    return query["question"], qa, joined


def make_dataset(source_zip, passages_file, split_counts=(12, 20, 40), seed=2026,
                 corpus_file=None):
    with zipfile.ZipFile(source_zip) as archive:
        raw = json.loads(archive.read("topiocqa/topiocqa_dev.json"))
    passages = json.loads(Path(passages_file).read_text())
    by_key = {(str(r["conv_id"]), int(r["turn_id"])): r for r in passages}
    if len(by_key) != len(passages):
        raise ValueError("Duplicate reference keys")
    corpus, labels, all_queries = {}, {}, []
    previous_topics = {}
    for row in sorted(raw, key=lambda r: (int(r["Conversation_no"]), int(r["Turn_no"]))):
        query = public_query(row)
        key = (query["conversation"], query["turn"])
        if key not in by_key:
            raise ValueError(f"Missing reference passage row for {key}")
        passage_row = by_key[key]
        expected = " [SEP] ".join(row["Context"] + [row["Question"]])
        # Official DPR preprocessing lowercases and removes question marks.
        # Keep raw questions untouched; compare lexical content only for the join.
        if normalize_text(passage_row["question"]) != normalize_text(expected):
            raise ValueError(f"Question/context mismatch for {key}")
        gold = []
        for field in ["positive_ctxs", "negative_ctxs", "hard_negative_ctxs"]:
            for p in passage_row.get(field, []):
                if not isinstance(p, dict) or not p.get("text"):
                    continue
                # Match the official evaluator's title+text identity, not row position.
                mid = "p_" + hashlib.sha256((p["title"] + "\n" + p["text"]).encode()).hexdigest()[:20]
                memory = dict(id=mid, title=p["title"], text=p["text"],
                              source_passage_id=str(p.get("passage_id", "")))
                if mid in corpus and (corpus[mid]["title"], corpus[mid]["text"]) != (p["title"], p["text"]):
                    raise ValueError("Passage hash collision")
                corpus[mid] = memory
                if field == "positive_ctxs":
                    gold.append(mid)
        prior = previous_topics.setdefault(query["conversation"], [])
        topic = row.get("Topic", "")
        transition = ("first" if not prior else "continue" if topic == prior[-1]
                      else "return" if topic in prior else "switch")
        prior.append(topic)
        answers = [row["Answer"]] + [a["Answer"] for a in row.get("Additional_answers", []) if a.get("Answer")]
        answerable = row["Answer"].strip().upper() != "UNANSWERABLE" and bool(gold)
        answer_norm = normalize_text(row["Answer"])
        labels[query["id"]] = dict(gold_ids=sorted(set(gold)), answers=answers,
                                   answerable=answerable, topic=topic, transition=transition,
                                   reference_answer_in_history=bool(answer_norm) and any(
                                       answer_norm in normalize_text(h["answer"]) for h in query["history"]))
        all_queries.append(query)
    # Validate the actual source ordering of every prefix, independently of Context lengths.
    lookup = {(str(r["Conversation_no"]), int(r["Turn_no"])): r for r in raw}
    for q in all_queries:
        for h in q["history"]:
            old = lookup[(q["conversation"], h["turn"])]
            if (h["question"], h["answer"]) != (old["Question"], old["Answer"]):
                raise ValueError("Dataset prefix disagrees with earlier recorded turns")
    groups = sorted({q["conversation"] for q in all_queries},
                    key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest())
    if any(n < 1 for n in split_counts) or sum(split_counts) > len(groups):
        raise ValueError(f"Positive split counts must fit {len(groups)} available conversations")
    splits, offset = {}, 0
    for name, count in zip(["calibration", "validation", "test"], split_counts):
        splits.update({g: name for g in groups[offset:offset + count]})
        offset += count
    queries = []
    excluded = []
    for q in all_queries:
        if q["conversation"] not in splits:
            continue
        if not labels[q["id"]]["answerable"]:
            excluded.append(dict(id=q["id"], reason="unanswerable or missing gold passage"))
            continue
        queries.append(q | {"split": splits[q["conversation"]]})
    corpus_mode = "restricted_annotation_derived_corpus"
    if corpus_file:
        # External static corpus: no gold insertion. Missing gold remains a retrieval miss.
        external = [json.loads(line) for line in Path(corpus_file).read_text().splitlines() if line.strip()]
        corpus = {}
        for p in external:
            mid = "p_" + hashlib.sha256((p["title"] + "\n" + p["text"]).encode()).hexdigest()[:20]
            corpus[mid] = dict(id=mid, title=p["title"], text=p["text"], source_passage_id=str(p.get("id", "")))
        corpus_mode = "external_static_corpus"
    metadata = dict(dataset="TopiOCQA official dev; internal exploratory conversation splits",
                    source_urls=SOURCES, source_sha256=digest(source_zip),
                    passages_sha256=digest(passages_file), corpus_mode=corpus_mode,
                    corpus_note="All released dev positive/negative passages pooled globally; no per-query gold insertion. Not full-Wikipedia retrieval.",
                    history_note="Recorded previous answers are available (teacher-forced history); current/future answers never enter retrieval.",
                    source_conversations=len(groups), split_counts=dict(zip(["calibration", "validation", "test"], split_counts)),
                    seed=seed, excluded=excluded)
    return sorted(corpus.values(), key=lambda p: p["id"]), queries, {q["id"]: labels[q["id"]] for q in queries}, metadata


def prepare(args):
    from .encoders import LSAEncoder, MiniLMEncoder
    out = new_directory(args.out)
    corpus, queries, labels, meta = make_dataset(args.source, args.passages,
                                                tuple(args.splits), args.seed, args.corpus)
    encoder = (LSAEncoder(args.dimensions, args.seed) if args.encoder == "lsa"
               else MiniLMEncoder(args.model_dir, args.revision, args.threads))
    passages = [f"{m['title']}\n{m['text']}" for m in corpus]
    encoder.fit(passages)
    texts = set(passages)
    query_parts = []
    for q in queries:
        current, qa, joined = query_texts(q, args.history_turns)
        query_parts.append((current, qa, joined))
        texts.update([current, joined] + qa)
    texts = sorted(texts)
    print(f"Encoding {len(texts)} texts; {len(corpus)} passages; {len(queries)} scored turns", flush=True)
    from .embedding_cache import CachedEncoder, DEFAULT_CACHE_DIR
    with CachedEncoder(encoder, getattr(args, "cache_dir", DEFAULT_CACHE_DIR),
                       getattr(args, "cache_batch_size", 64)) as cached:
        vectors = cached.encode(texts)
        meta["embedding_cache"] = cached.stats()
    index = {t: i for i, t in enumerate(texts)}
    encoded = lambda t: vectors[index[t]]
    current, history, permuted, concat = [], [], [], []
    for q, (text, qa, joined) in zip(queries, query_parts):
        current.append(encoded(text))
        concat.append(encoded(joined))
        if qa:
            weights = args.decay ** np.arange(len(qa) - 1, -1, -1)
            weights /= weights.sum()
            h = np.array([encoded(t) for t in qa])
            history.append(weights @ h)
            permuted.append(weights @ h[::-1])
        else:
            history.append(np.zeros(vectors.shape[1]))
            permuted.append(np.zeros(vectors.shape[1]))
    for m in corpus:
        m["cost"] = encoder.count(payload(m))
    from .encoders import normalized
    arrays = dict(corpus=np.array([encoded(t) for t in passages], dtype=np.float32),
                  current=np.array(current, dtype=np.float32),
                  history=normalized(np.array(history)).astype(np.float32),
                  permuted=normalized(np.array(permuted)).astype(np.float32),
                  concat=np.array(concat, dtype=np.float32))
    np.savez_compressed(out / "embeddings.npz", **arrays)
    dump(out / "corpus.json", corpus)
    dump(out / "queries.json", queries)
    dump(out / "labels.json", labels)
    meta.update(encoder=encoder.metadata, history_turns=args.history_turns, decay=args.decay,
                empty_current_vectors=int(np.sum(np.linalg.norm(arrays["current"], axis=1) < 1e-10)),
                corpus_size=len(corpus), scored_turns=len(queries))
    meta["files_sha256"] = {f: digest(out / f) for f in ["corpus.json", "queries.json", "labels.json", "embeddings.npz"]}
    dump(out / "manifest.json", meta)
    print(f"Prepared: {out}", flush=True)
