"""Generate a matched synthetic benchmark for memory deduplication.

Every profile is instantiated at 1, 2, 4, and 8 copies.  The facts,
distractors, updates, query, and first copy of every fact are identical across
conditions; increasing the copy count only appends exact or paraphrased copies.
"""

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path


COPY_COUNTS = [1, 2, 4, 8]

NAMES = [
    "Amina", "Bruno", "Chloe", "Dario", "Elena", "Farah", "Gustav", "Hana",
    "Ivan", "Jia", "Klara", "Luca", "Mina", "Noah", "Oskar", "Priya",
]
PROJECTS = ["Atlas", "Borealis", "Cirrus", "Delta", "Ember", "Fjord", "Gaia", "Helix"]
LANGUAGES = ["Python", "Rust", "Java", "Go", "Kotlin", "C#"]
DATABASES = ["PostgreSQL", "MariaDB", "MongoDB", "CockroachDB", "SQLite", "DynamoDB"]
REGIONS = ["Zurich", "Frankfurt", "Paris", "Milan", "Vienna", "Amsterdam"]
CHANNELS = ["PagerDuty", "Slack", "Opsgenie", "email", "Microsoft Teams", "SMS"]
TEAMS = ["Platform", "Reliability", "Data Systems", "Core Services", "Cloud Operations", "Runtime"]
PLATFORMS = [
    ("Kubernetes 1.28", "Kubernetes 1.30"),
    ("Docker Swarm 24", "Docker Swarm 27"),
    ("Nomad 1.6", "Nomad 1.8"),
    ("OpenShift 4.13", "OpenShift 4.16"),
]

PARAPHRASES = {
    "language": [
        "{project} is implemented primarily in {value}.",
        "The main programming language for {project} is {value}.",
        "Most of the {project} codebase is written in {value}.",
        "Development of {project} mainly uses {value}.",
        "{value} is the principal language behind {project}.",
        "The engineering team chose {value} for the core of {project}.",
        "Core services in {project} are built with {value}.",
        "For its primary implementation, {project} relies on {value}.",
    ],
    "owner_team": [
        "The {value} team owns the {project} service.",
        "Operational ownership of {project} belongs to {value}.",
        "{project} is maintained by the {value} team.",
        "The team responsible for {project} is {value}.",
        "{value} has engineering ownership of {project}.",
        "Maintenance responsibility for {project} sits with {value}.",
        "The designated owner of {project} is the {value} team.",
        "{project}'s accountable engineering group is {value}.",
    ],
    "alert_channel": [
        "Operational alerts for {project} are sent through {value}.",
        "The {project} team receives production alerts in {value}.",
        "{value} is the alert channel used by {project} operations.",
        "When {project} fails, notifications arrive through {value}.",
        "The on-call alerts for {project} use {value}.",
        "{project}'s incident notifications are delivered via {value}.",
        "Production monitoring for {project} pages the team using {value}.",
        "The designated operations-alert destination for {project} is {value}.",
    ],
}

DISTRACTOR_TEMPLATES = [
    "{name} attended a conference about edge computing last month.",
    "The documentation team is redesigning the {project} landing page.",
    "A prototype from another team uses the {other_language} language.",
    "The office cafeteria recently changed its lunch menu.",
    "A retired test environment once ran in the {other_region} region.",
    "The team plans to buy new laptops during the next quarter.",
    "A public tutorial demonstrates {other_database} with an unrelated sample application.",
    "The weekly engineering meeting is held on Tuesday afternoon.",
    "A colleague prefers receiving personal messages through {other_channel}.",
    "The company is evaluating a new expense-reporting system.",
]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(
        json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def stable_order(seed, values):
    return sorted(
        values,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
    )


def choose_other(rng, values, excluded):
    return rng.choice([value for value in values if value != excluded])


def profile_values(profile_id, seed):
    rng = random.Random(f"{seed}:profile:{profile_id}")
    name = rng.choice(NAMES)
    project = rng.choice(PROJECTS)
    language = rng.choice(LANGUAGES)
    database = rng.choice(DATABASES)
    primary = rng.choice(REGIONS)
    backup = choose_other(rng, REGIONS, primary)
    channel = rng.choice(CHANNELS)
    owner_team = rng.choice(TEAMS)
    old_platform, current_platform = rng.choice(PLATFORMS)
    return {
        "name": name,
        "project": project,
        "language": language,
        "database": database,
        "primary_region": primary,
        "backup_region": backup,
        "alert_channel": channel,
        "owner_team": owner_team,
        "old_platform": old_platform,
        "current_platform": current_platform,
        "other_language": choose_other(rng, LANGUAGES, language),
        "other_database": choose_other(rng, DATABASES, database),
        "other_region": choose_other(rng, REGIONS, primary),
        "other_channel": choose_other(rng, CHANNELS, channel),
    }


def memory(mid, text, group, facts, kind, valid=True):
    return {
        "id": mid,
        "text": text,
        "equivalence_group": group,
        "fact_ids": list(facts),
        "kind": kind,
        "valid": bool(valid),
    }


def master_memories(profile_id, values, seed):
    prefix = f"p{profile_id:04d}"
    project = values["project"]
    result = []

    # Three paraphrase groups.
    for slot in ["language", "owner_team", "alert_channel"]:
        fact = f"{prefix}:fact:{slot}"
        group = f"{prefix}:eq:{slot}"
        for copy, template in enumerate(PARAPHRASES[slot], 1):
            result.append(memory(
                f"{prefix}:{slot}:c{copy}",
                template.format(project=project, value=values[slot]),
                group, [fact], "paraphrase-duplicate",
            ))

    # One exact-copy group. IDs differ; payloads deliberately do not.
    database_text = f"{project} stores its production records in {values['database']}."
    for copy in range(1, 9):
        result.append(memory(
            f"{prefix}:database:c{copy}", database_text,
            f"{prefix}:eq:database", [f"{prefix}:fact:database"],
            "exact-duplicate",
        ))

    # A deliberately close but complementary pair. These must both survive.
    result.extend([
        memory(
            f"{prefix}:primary-detail",
            f"The {project} deployment runs in {values['primary_region']} as the primary region.",
            f"{prefix}:eq:primary_region", [f"{prefix}:fact:primary_region"],
            "complementary",
        ),
        memory(
            f"{prefix}:backup-detail",
            f"The {project} deployment runs in {values['backup_region']} as the backup region.",
            f"{prefix}:eq:backup_region", [f"{prefix}:fact:backup_region"],
            "complementary",
        ),
    ])

    # A close temporal update pair. Only the current state is gold.
    result.extend([
        memory(
            f"{prefix}:platform:old",
            f"Previously, {project} ran on {values['old_platform']}; this configuration is obsolete.",
            f"{prefix}:eq:platform_old", [f"{prefix}:fact:platform_old"],
            "update-old", valid=False,
        ),
        memory(
            f"{prefix}:platform:current",
            f"After the upgrade, {project} now runs on {values['current_platform']}.",
            f"{prefix}:eq:platform_current", [f"{prefix}:fact:platform_current"],
            "update-current", valid=True,
        ),
    ])

    distractor_values = dict(values)
    for number, template in enumerate(DISTRACTOR_TEMPLATES, 1):
        result.append(memory(
            f"{prefix}:distractor:{number}", template.format(**distractor_values),
            f"{prefix}:eq:distractor:{number}", [], "distractor",
        ))
    return result


def case_for(profile_id, copies, split, values, master, seed):
    prefix = f"p{profile_id:04d}"
    repeated_slots = {"language", "owner_team", "alert_channel", "database"}
    selected = []
    for item in master:
        parts = item["id"].split(":")
        slot = parts[1] if len(parts) > 2 else ""
        if slot in repeated_slots and parts[-1].startswith("c"):
            if int(parts[-1][1:]) <= copies:
                selected.append(item)
        else:
            selected.append(item)
    # Order is fixed per profile and does not depend on copy count. Filtering the
    # master ordering therefore makes every smaller condition a strict subset.
    master_order = stable_order(f"{seed}:memory-order:{profile_id}", [m["id"] for m in master])
    rank = {mid: position for position, mid in enumerate(master_order)}
    selected = sorted(selected, key=lambda item: rank[item["id"]])
    gold = [
        f"{prefix}:fact:language",
        f"{prefix}:fact:database",
        f"{prefix}:fact:owner_team",
        f"{prefix}:fact:primary_region",
        f"{prefix}:fact:backup_region",
        f"{prefix}:fact:alert_channel",
        f"{prefix}:fact:platform_current",
    ]
    query = (
        f"Prepare the current {values['project']} deployment brief. State its main programming "
        "language, production database, and owning team; identify both the primary deployment "
        "region and the backup region; give the current platform version; and name the operations "
        "alert channel."
    )
    aspect_specs = [
        ("language", f"What is the main programming language used by {values['project']}?"),
        ("database", f"Which production database does {values['project']} use?"),
        ("owner_team", f"Which engineering team owns {values['project']}?"),
        ("primary_region", f"What is the primary deployment region for {values['project']}?"),
        ("backup_region", f"What is the backup deployment region for {values['project']}?"),
        ("platform_current", f"Which platform version does {values['project']} currently run on?"),
        ("alert_channel", f"Which channel receives operational alerts for {values['project']}?"),
    ]
    aspects = [
        {
            "id": name,
            "text": text,
            "critical": True,
            # This field is evaluation/oracle metadata. Deployable selectors are
            # tested to consume only id/text/critical.
            "gold_fact_id": f"{prefix}:fact:{name}",
        }
        for name, text in aspect_specs
    ]
    answer_by_aspect = {
        "language": values["language"],
        "database": values["database"],
        "owner_team": values["owner_team"],
        "primary_region": values["primary_region"],
        "backup_region": values["backup_region"],
        "platform_current": values["current_platform"],
        "alert_channel": values["alert_channel"],
    }
    leaks = [
        (answer, aspect["text"])
        for aspect in aspects for answer in [answer_by_aspect[aspect["id"]]]
        if re.search(r"(?<!\w)" + re.escape(answer.lower()) + r"(?!\w)", aspect["text"].lower())
    ]
    if leaks:
        raise AssertionError(f"An aspect description leaked an answer value: {leaks[0]}")
    return {
        "case_id": f"{prefix}:copies:{copies}",
        "profile_id": prefix,
        "split": split,
        "copies": copies,
        "query": query,
        "aspects": aspects,
        "gold_fact_ids": gold,
        "memories": selected,
    }


def generate(profiles=80, split_counts=(12, 20, 48), seed=2026):
    if sum(split_counts) != profiles or any(value < 1 for value in split_counts):
        raise ValueError("Positive --split-counts must sum to --profiles")
    profile_ids = stable_order(seed, list(range(profiles)))
    split_by_profile = {}
    offset = 0
    for name, count in zip(["calibration", "validation", "test"], split_counts):
        for profile_id in profile_ids[offset:offset + count]:
            split_by_profile[profile_id] = name
        offset += count
    cases = []
    for profile_id in range(profiles):
        values = profile_values(profile_id, seed)
        master = master_memories(profile_id, values, seed)
        for copies in COPY_COUNTS:
            cases.append(case_for(
                profile_id, copies, split_by_profile[profile_id], values, master, seed
            ))
    manifest = {
        "dataset": "controlled matched memory deduplication",
        "schema_version": 2,
        "seed": seed,
        "profiles": profiles,
        "copy_counts": COPY_COUNTS,
        "split_profiles": dict(zip(["calibration", "validation", "test"], split_counts)),
        "case_splits": dict(Counter(case["split"] for case in cases)),
        "design": [
            "Facts and distractors are identical across copy-count conditions.",
            "Increasing copies only appends exact or paraphrased equivalents.",
            "Close complementary facts and old/current update pairs are protected negative controls.",
            "Aspect descriptions are derived from the query and contain no answer values.",
        ],
    }
    return {"manifest": manifest, "cases": cases}


def validate(payload):
    cases = payload["cases"]
    by_profile = {}
    for case in cases:
        by_profile.setdefault(case["profile_id"], {})[case["copies"]] = case
        ids = [item["id"] for item in case["memories"]]
        if len(ids) != len(set(ids)):
            raise AssertionError("Duplicate memory ID")
        if len(case["gold_fact_ids"]) != len(set(case["gold_fact_ids"])):
            raise AssertionError("Duplicate gold fact ID")
        if {aspect["gold_fact_id"] for aspect in case["aspects"]} != set(case["gold_fact_ids"]):
            raise AssertionError("Aspect/gold alignment failed")
        if any(not aspect["text"].strip() for aspect in case["aspects"]):
            raise AssertionError("Empty aspect description")
    for conditions in by_profile.values():
        if set(conditions) != set(COPY_COUNTS):
            raise AssertionError("Missing copy condition")
        fixed = None
        previous = set()
        for copies in COPY_COUNTS:
            case = conditions[copies]
            ids = {item["id"] for item in case["memories"]}
            if not previous <= ids:
                raise AssertionError("Smaller condition is not a subset")
            current_fixed = {
                item["id"]: item["text"] for item in case["memories"]
                if item["kind"] not in {"exact-duplicate", "paraphrase-duplicate"}
            }
            if fixed is None:
                fixed = current_fixed
            elif fixed != current_fixed:
                raise AssertionError("Fixed memories changed across conditions")
            previous = ids
    return True


def self_test():
    payload = generate(6, (2, 2, 2), 17)
    validate(payload)
    one = next(case for case in payload["cases"] if case["copies"] == 1)
    eight = next(
        case for case in payload["cases"]
        if case["profile_id"] == one["profile_id"] and case["copies"] == 8
    )
    assert len(eight["memories"]) - len(one["memories"]) == 4 * 7
    assert sum(item["kind"] == "complementary" for item in one["memories"]) == 2
    assert sum(item["kind"] == "update-current" for item in one["memories"]) == 1
    print("Generator self-tests passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="controlled_magnitude_dedup/data/controlled_magnitude_dedup.json"
    )
    parser.add_argument("--profiles", type=int, default=80)
    parser.add_argument("--split-counts", type=int, nargs=3, default=[12, 20, 48])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    payload = generate(args.profiles, tuple(args.split_counts), args.seed)
    validate(payload)
    atomic_json(args.out, payload)
    print(f"Saved {len(payload['cases'])} matched cases to {args.out}")
    print("Cases by split:", payload["manifest"]["case_splits"])


if __name__ == "__main__":
    main()
