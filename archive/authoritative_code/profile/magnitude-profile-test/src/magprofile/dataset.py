from __future__ import annotations

from dataclasses import asdict, dataclass
import random

import pandas as pd


@dataclass(frozen=True)
class Example:
    example_id: str
    group_id: str
    template_family: str
    candidate_type: str
    base_memories: list[str]
    candidate: str
    candidate_facts: list[str]
    base_facts: list[str]
    info_delta: int
    has_new_information: int
    compatible: int
    subject_relevant: int


PEOPLE = [
    "Alex", "Sam", "Jordan", "Taylor", "Morgan", "Riley", "Casey", "Avery",
    "Robin", "Drew", "Jamie", "Cameron", "Quinn", "Emery", "Skyler", "Rowan",
    "Sasha", "Noel", "Reese", "Parker", "Hayden", "Blair", "Finley", "Dakota",
]
DATABASES = ["PostgreSQL", "MySQL", "MariaDB", "MongoDB", "SQLite", "CockroachDB"]
LANGUAGES = ["Python", "Java", "Rust", "Go", "TypeScript", "C++"]
CLOUDS = ["AWS", "Azure", "Google Cloud", "Hetzner", "DigitalOcean"]
CITIES = ["Zurich", "Berlin", "Vienna", "Amsterdam", "Copenhagen", "Munich"]
HOBBIES = ["Japanese", "Italian", "photography", "climbing", "piano", "painting"]

DB_BASE_TEMPLATES = [
    "{p} uses {db}.", "{p}'s database is {db}.", "For databases, {p} relies on {db}.",
    "{db} supports {p}'s application.", "The data store chosen by {p} is {db}.",
    "{p} works with {db} as the primary database.", "In the current stack, {p} has {db}.",
    "{p}'s application persists its data in {db}.", "Database work for {p} is done with {db}.",
    "The application maintained by {p} is backed by {db}.",
]
PARAPHRASE_TEMPLATES = [
    "{db} is the database used by {p}.", "{p} employs {db} for database work.",
    "The database technology in {p}'s stack is {db}.", "{p}'s data is kept with {db}.",
    "When persistence is needed, {p} chooses {db}.", "{db} serves as {p}'s primary data store.",
    "The application belonging to {p} depends on {db} for storage.",
    "For persistent records, {p} has selected {db}.", "{p}'s database platform of choice is {db}.",
    "Data management in {p}'s project is handled by {db}.",
]
INCREMENTAL_TEMPLATES = [
    "{p} uses {db} version {v} through a managed service.",
    "{p}'s managed database is {db} {v}.", "The {db} installation used by {p} is managed and runs version {v}.",
    "Version {v} of {db} is hosted for {p} by a managed provider.",
    "{p} relies on managed {db}, specifically release {v}.", "{p}'s {db} environment is a managed version-{v} deployment.",
    "A provider operates {p}'s {db} {v} instance.", "{p} selected {db} {v} with managed hosting.",
    "Managed hosting supplies version {v} of {db} to {p}.", "The database details for {p} are {db} {v}, managed hosting.",
]
UPDATE_TEMPLATES = [
    "{p} recently migrated from {db} to {other}.", "{p} now uses {other} after replacing {db}.",
    "The database changed from {db} to {other} for {p}.", "After a migration, {other} superseded {db} in {p}'s stack.",
    "{p} moved the application off {db} and onto {other}.", "{other} is now current for {p}; previously it was {db}.",
    "A recent migration took {p} from {db} to {other}.", "{p}'s former database was {db}, but the active one is {other}.",
    "The new database for {p} is {other}, replacing {db}.", "{p} completed a switch: {db} was removed and {other} adopted.",
]
CONTRADICTION_TEMPLATES = [
    "{p} does not use {db}; {p} uses {other} instead.", "Contrary to the record, {p}'s database is {other}, not {db}.",
    "{db} is not used by {p}; the actual database is {other}.", "It is incorrect that {p} uses {db}: {other} is used.",
    "{p} works with {other} rather than {db}.", "The claim about {db} is false for {p}, whose database is {other}.",
    "For {p}, replace the incorrect value {db} with {other}.", "{p}'s database is {other}; reports of {db} are mistaken.",
    "{p} never adopted {db} and uses {other}.", "The correct database for {p} is {other}, whereas {db} is incorrect.",
]
IRRELEVANT_TEMPLATES = [
    "{db} is an open-source database with a long development history.", "The database project {db} has existed for many years.",
    "Developers around the world contribute to {db}.", "{db} belongs to a broad family of database systems.",
    "Many organizations have deployed {db} software.", "Documentation for {db} is available to database administrators.",
    "The {db} ecosystem includes numerous third-party tools.", "Database courses sometimes introduce {db} as an example.",
    "Community events occasionally include talks about {db}.", "Books have been written about operating {db} systems.",
]
NOVEL_TEMPLATES = [
    "{p} is learning {hobby} in the evenings.", "During free evenings, {p} studies {hobby}.",
    "{p} recently began learning {hobby}.", "A new personal interest for {p} is {hobby}.",
    "Outside work, {p} practices {hobby}.", "{hobby} has become an evening activity for {p}.",
    "{p} spends spare time developing skills in {hobby}.", "The subject {p} is currently learning is {hobby}.",
    "Evening lessons are helping {p} learn {hobby}.", "{p}'s latest non-work pursuit is {hobby}.",
]


def _base_set(person: str, db: str, language: str, cloud: str, city: str, size: int, family: int) -> tuple[list[str], list[str]]:
    pool = [
        (DB_BASE_TEMPLATES[family].format(p=person, db=db), f"{person}|uses_database|{db}"),
        (f"{person} prefers {language} for backend development.", f"{person}|prefers_language|{language}"),
        (f"{person}'s main application runs on {cloud}.", f"{person}|cloud|{cloud}"),
        (f"{person} lives in {city}.", f"{person}|lives_in|{city}"),
        (f"{person} works on an analytics application.", f"{person}|works_on|analytics_application"),
        (f"{person} deploys services with containers.", f"{person}|deployment|containers"),
        (f"{person} values low-latency systems.", f"{person}|values|low_latency"),
        (f"{person} backs up production data daily.", f"{person}|backup_frequency|daily"),
        (f"{person} works remotely three days per week.", f"{person}|remote_days|3"),
        (f"{person} uses Linux for development.", f"{person}|development_os|Linux"),
        (f"{person} monitors services with dashboards.", f"{person}|monitoring|dashboards"),
        (f"{person} reviews production incidents monthly.", f"{person}|incident_review|monthly"),
        (f"{person} maintains automated integration tests.", f"{person}|tests|integration"),
        (f"{person} stores source code in Git.", f"{person}|vcs|Git"),
        (f"{person} uses continuous delivery.", f"{person}|delivery|continuous"),
        (f"{person} documents system decisions.", f"{person}|documents|decisions"),
    ]
    selected = pool[:size]
    return [x[0] for x in selected], [x[1] for x in selected]


def generate_dataset(
    groups: int = 120,
    set_sizes: tuple[int, ...] = (4, 8, 16),
    seed: int = 13,
) -> pd.DataFrame:
    """Generate controlled pairs; every scenario group contains all relation types."""
    rng = random.Random(seed)
    examples: list[Example] = []
    types = (
        "duplicate", "paraphrase", "incremental", "temporal_update",
        "contradiction", "related_irrelevant", "novel_topic",
    )
    for i in range(groups):
        person = PEOPLE[i % len(PEOPLE)]
        db = DATABASES[(i // len(PEOPLE)) % len(DATABASES)]
        other_db = DATABASES[(DATABASES.index(db) + 1 + i) % len(DATABASES)]
        language = LANGUAGES[(i * 5 + 1) % len(LANGUAGES)]
        cloud = CLOUDS[(i * 3 + 2) % len(CLOUDS)]
        city = CITIES[(i * 7 + 1) % len(CITIES)]
        hobby = HOBBIES[(i * 11 + 2) % len(HOBBIES)]
        size = set_sizes[i % len(set_sizes)]
        family = i % len(DB_BASE_TEMPLATES)
        base, base_facts = _base_set(person, db, language, cloud, city, size, family)
        db_fact = f"{person}|uses_database|{db}"

        candidates = {
            "duplicate": (
                base[0], [db_fact], 1, 1,
            ),
            "paraphrase": (
                PARAPHRASE_TEMPLATES[family].format(p=person, db=db), [db_fact], 1, 1,
            ),
            "incremental": (
                INCREMENTAL_TEMPLATES[family].format(p=person, db=db, v=14 + i % 4),
                [db_fact, f"{person}|database_version|{14 + i % 4}", f"{person}|database_hosting|managed"], 1, 1,
            ),
            "temporal_update": (
                UPDATE_TEMPLATES[family].format(p=person, db=db, other=other_db),
                [f"{person}|previous_database|{db}", f"{person}|uses_database|{other_db}"], 1, 1,
            ),
            "contradiction": (
                CONTRADICTION_TEMPLATES[family].format(p=person, db=db, other=other_db),
                [f"{person}|not_uses_database|{db}", f"{person}|uses_database|{other_db}"], 0, 1,
            ),
            "related_irrelevant": (
                IRRELEVANT_TEMPLATES[family].format(db=db),
                [f"{db}|category|open_source_database"], 1, 0,
            ),
            "novel_topic": (
                NOVEL_TEMPLATES[family].format(p=person, hobby=hobby),
                [f"{person}|learning|{hobby}"], 1, 1,
            ),
        }
        group_id = f"scenario-{i:04d}"
        for kind in types:
            text, facts, compatible, relevant = candidates[kind]
            delta = len(set(facts) - set(base_facts))
            examples.append(
                Example(
                    example_id=f"{group_id}-{kind}",
                    group_id=group_id,
                    template_family=f"template-{family:02d}",
                    candidate_type=kind,
                    base_memories=base,
                    candidate=text,
                    candidate_facts=facts,
                    base_facts=base_facts,
                    info_delta=delta,
                    has_new_information=int(delta > 0),
                    compatible=compatible,
                    subject_relevant=relevant,
                )
            )
    rng.shuffle(examples)
    return pd.DataFrame(asdict(x) for x in examples)
