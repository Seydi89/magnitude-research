"""Install the supplied code into an existing project, backing up replaced files."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil


def install(target):
    source = Path(__file__).resolve().parent
    target = Path(target).expanduser().resolve()
    if target == source:
        raise ValueError("Choose the existing project directory, not this extracted update directory")
    for required in ["synthetic_memory_pilot/evaluate.py", "trajectory_memory/magnitude.py"]:
        if not (target / required).is_file():
            raise ValueError(f"Existing project is missing {required}: {target}")
    files = []
    for folder in ["trajectory_memory", "synthetic_memory_pilot"]:
        files += [p for p in (source / folder).iterdir() if p.is_file() and p.suffix in {".py", ".sh", ".yml"}]
    files += list((source / "tests").glob("*.py"))
    files += [source / "README_CACHE_UPDATE.md", source / "VALIDATION.md"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = target / "update_backups" / stamp
    # Back up every replaced file before writing the first update.
    for path in files:
        rel = path.relative_to(source)
        existing = target / rel
        if existing.exists():
            saved = backup / rel
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(existing, saved)
    for path in files:
        dest = target / path.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    print(f"Installed {len(files)} code/documentation files into {target}")
    print(f"Previous files backed up under {backup}")
    print("Existing datasets, caches and result files were not changed.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("project", help="existing parent directory containing BOTH project folders")
    args = p.parse_args()
    try:
        install(args.project)
    except ValueError as error:
        p.error(str(error))
