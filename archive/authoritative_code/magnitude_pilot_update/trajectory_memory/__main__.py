import argparse


def main():
    parser = argparse.ArgumentParser(description="History-dependent retrieval with magnitude-profile selection")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("fetch", help="Download official TopiOCQA dev questions and reference passages")
    p.add_argument("--out", default="data/raw")
    p = commands.add_parser("prepare", help="Build fixed corpus, conversation splits, and cached embeddings")
    p.add_argument("--source", default="data/raw/topiocqa.zip")
    p.add_argument("--passages", default="data/raw/passages_dev.json")
    p.add_argument("--corpus", help="Optional static JSONL corpus with title,text,id; no gold insertion")
    p.add_argument("--out", default="data/prepared")
    p.add_argument("--encoder", choices=["lsa", "minilm"], default="lsa")
    from .embedding_cache import DEFAULT_CACHE_DIR
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--cache-batch-size", type=int, default=64)
    p.add_argument("--dimensions", type=int, default=128)
    p.add_argument("--model-dir", help="Local tokenizer.json and onnx/model.onnx directory")
    p.add_argument("--revision", default="1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
    p.add_argument("--splits", type=int, nargs=3, default=[12, 20, 40], metavar=("CAL", "VAL", "TEST"))
    p.add_argument("--history-turns", type=int, default=6)
    p.add_argument("--decay", type=float, default=.8)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--threads", type=int, default=2)
    p = commands.add_parser("run", help="Run retrieval, validation, held-out selection, and profile audits")
    p.add_argument("--data", default="data/prepared")
    p.add_argument("--out", default="runs/my_run")
    p.add_argument("--candidates", type=int, default=32)
    p.add_argument("--budgets", type=int, nargs="+", default=[256, 512])
    p.add_argument("--dense-nodes", type=int, default=129)
    p.add_argument("--max-scales", type=int, default=65)
    p.add_argument("--profile-tolerance", type=float, default=.002)
    p.add_argument("--threads", type=int, default=1)
    p = commands.add_parser("export-prompts", help="Export answer-generation inputs without evaluation labels")
    p.add_argument("--data", default="data/prepared")
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--methods", nargs="+", default=["current_relevance", "history_relevance", "history_profile_magnitude"])
    p.add_argument("--budget", type=int, default=512)
    p.add_argument("--limit", type=int, default=100)
    p = commands.add_parser("answer", help="Optional explicit call to an OpenAI-compatible reader endpoint")
    p.add_argument("--prompts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--max-output-tokens", type=int, default=256)
    p = commands.add_parser("score-answers", help="Score answer text and citation IDs; no entailment claim")
    p.add_argument("--answers", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--data", default="data/prepared")
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    if hasattr(args, "threads") and args.threads < 1:
        parser.error("threads must be positive")
    if args.command == "prepare" and (args.history_turns < 1 or not 0 < args.decay <= 1 or args.dimensions < 2 or args.cache_batch_size < 1):
        parser.error("Invalid history, decay or dimensions")
    if args.command == "run" and (args.candidates < 2 or min(args.budgets) < 1):
        parser.error("Positive budgets and at least two candidates required")
    if args.command == "fetch":
        from .data import fetch
        fetch(args.out)
    elif args.command == "prepare":
        from threadpoolctl import threadpool_limits
        from .data import prepare
        with threadpool_limits(limits=args.threads):
            prepare(args)
    elif args.command == "run":
        from .experiment import run
        run(args)
    else:
        from .reader import export_prompts, answer, score_answers
        {"export-prompts": export_prompts, "answer": answer, "score-answers": score_answers}[args.command](args)


if __name__ == "__main__":
    main()
