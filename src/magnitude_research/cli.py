from __future__ import annotations
import argparse, json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description="Magnitude research experiment index")
    p.add_argument("command",choices=["list","show"]); p.add_argument("experiment",nargs="?")
    args=p.parse_args(); path=Path(__file__).resolve().parents[2]/"experiments.json"
    data=json.loads(path.read_text())
    if args.command=="list":
        for x in data: print(f"{x['id']:<28} {x['status']:<13} {x['question']}")
    else:
        item=next((x for x in data if x["id"]==args.experiment),None)
        if item is None: p.error("unknown experiment")
        print(json.dumps(item,indent=2))


if __name__ == "__main__":
    main()
