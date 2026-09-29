"""Synthetic personal-memory stores for students, with a redundancy knob.

Every memory carries fact tags (slot, value, current?). A question lists required
facts; a fact is covered if ANY selected memory carries its current tag, so
paraphrased duplicates are interchangeable evidence. All names are fictional.
"""
import random

CITIES = ["Veltmar", "Orsenne", "Kaldor", "Brisa", "Tamsford", "Lunereth"]
UNI_FORMS = ["{c} Institute of Technology", "University of {c}", "{c} College of Applied Sciences"]
MAJORS = ["computer science", "economics", "mechanical engineering", "biology", "architecture", "psychology", "law", "physics"]
YEARS = ["first", "second", "third", "fourth", "fifth"]
HOODS = ["Old Harbour", "Riverside", "Northgate", "Linden Hill", "Market Square", "East Park"]
JOBS = ["barista", "maths tutor", "library assistant", "bike courier", "lab assistant", "waiter"]
SPORTS = ["rowing club", "climbing gym", "university football team", "swimming club", "running group"]
COMMUTE = ["bike", "tram", "bus", "train"]
LANGS = ["Spanish", "Japanese", "German", "Italian", "Mandarin"]
ADVISORS = ["Prof. Haller", "Prof. Osei", "Prof. Lindqvist", "Prof. Moreau", "Prof. Tanaka"]
NAMES = ["Lena", "Tomas", "Aiko", "Marco", "Nadia", "Jonas", "Priya", "Elif"]

T = {
    "city": ["I live in {v} now.", "Moving to {v} was a big change for me.", "The rent in {v} is really high.",
             "I love walking around {v} in the evening.", "Life in {v} is quieter than back home.", "I have been living in {v} for a while."],
    "university": ["I study at {v}.", "I am enrolled at {v}.", "My classes at {v} start at eight.",
                   "The campus of {v} is huge.", "Exams at {v} are brutal this semester.", "The library at {v} is my second home."],
    "major": ["My major is {v}.", "I am doing a degree in {v}.", "The {v} lectures are hard this term.",
              "I chose {v} because I like solving problems.", "My {v} project is due on Friday.", "I switched my electives but I still study {v}."],
    "year": ["I am in my {v} year.", "This is my {v} year at uni.", "Being a {v} year student is stressful.",
             "In my {v} year the workload doubled.", "I only have courses left from my {v} year plan.", "My {v} year timetable is packed."],
    "housing": ["My flat is in {v}.", "I share an apartment in {v}.", "{v} is a nice neighbourhood to live in.",
                "I moved into a student room in {v}.", "The noise in {v} keeps me up at night.", "My place in {v} is close to a bakery."],
    "job": ["I work part-time as a {v}.", "My job as a {v} pays the rent.", "I had a long shift as a {v} today.",
            "Working as a {v} helps with money.", "My boss at the {v} job is strict.", "I might quit being a {v} before exams."],
    "sport": ["I go to the {v} twice a week.", "Training with the {v} was tough today.", "I am a member of the {v}.",
              "The {v} has a competition next month.", "I made friends at the {v}.", "I skipped the {v} because of a cold."],
    "commute": ["I get to campus by {v}.", "My commute is by {v}.", "Taking the {v} to class takes twenty minutes.",
                "The {v} was late again this morning.", "I prefer the {v} over walking to uni.", "I use the {v} every day to reach campus."],
    "language": ["I am learning {v} in the evenings.", "My {v} course is going well.", "I practise {v} with a tandem partner.",
                 "My {v} vocabulary test is tomorrow.", "I started {v} lessons this term.", "I watch films in {v} to practise."],
    "advisor": ["My advisor is {v}.", "I met {v} about my thesis.", "{v} supervises my project.",
                "{v} gave me feedback on my draft.", "I have a meeting with {v} next week.", "{v} approved my thesis topic."],
    "top": ["{u} is ranked the best university in {c}.", "People say {u} is the top university in {c}.",
            "{u} is the highest ranked school in {c}.", "Everyone agrees {u} is the leading university in {c}.",
            "In the rankings {u} comes first in {c}.", "{u} has the best reputation of all universities in {c}."],
    "old_uni": ["I study at {v}.", "I am enrolled at {v}.", "My classes at {v} start at eight.",
                "The campus of {v} is huge.", "Exams at {v} are brutal this semester.", "The library at {v} is my second home."],
}
CHAT = ["I cooked pasta again tonight.", "The weather has been rainy all week.", "I watched a documentary about whales.",
        "I need to buy a new laptop charger.", "My phone screen cracked yesterday.", "I tried a new ramen place.",
        "I am thinking about adopting a cat.", "My parents visited last weekend.", "I finally finished that novel.",
        "The heating in my building broke.", "I want to learn to play guitar.", "I spent the evening cleaning."]

SLOTS = ["city", "university", "major", "year", "housing", "job", "sport", "commute", "language", "advisor"]


def world(seed=7):
    rng = random.Random(seed)
    unis = {c: [f.format(c=c) for f in UNI_FORMS] for c in CITIES}
    top = {c: rng.choice(unis[c]) for c in CITIES}
    return unis, top


def make_user(uid, copies, rng, unis, top):
    city = rng.choice(CITIES)
    uni = rng.choice(unis[city])
    p = dict(city=city, university=uni, major=rng.choice(MAJORS), year=rng.choice(YEARS), housing=rng.choice(HOODS),
             job=rng.choice(JOBS), sport=rng.choice(SPORTS), commute=rng.choice(COMMUTE), language=rng.choice(LANGS),
             advisor=rng.choice(ADVISORS))
    transfer = rng.random() < .35
    old = rng.choice([u for u in unis[city] if u != uni]) if transfer else None
    mem = []
    def add(text, tags): mem.append(dict(id=f"u{uid}_m{len(mem)}", text=text, tags=[list(t) for t in tags]))
    for s in SLOTS:
        for tpl in rng.sample(T[s], copies):
            add(tpl.format(v=p[s]), [(s, p[s], True)])
    for tpl in rng.sample(T["top"], copies):
        add(tpl.format(u=top[city], c=city), [("top", city, True)])
    if transfer:
        for tpl in rng.sample(T["old_uni"], copies):
            add(tpl.format(v=old), [("university", old, False)])
        add(f"I transferred from {old} to {uni} this semester.", [("university", uni, True), ("university", old, False)])
    add(f"I moved to {city} to study {p['major']} at {uni}.", [("city", city, True), ("university", uni, True), ("major", p["major"], True)])
    # distractors: fixed count, independent of redundancy level
    other = rng.choice([c for c in CITIES if c != city])
    add(f"{top[other]} is ranked the best university in {other}.", [("top", other, True)])
    f1, f2 = rng.sample(NAMES, 2)
    add(f"My friend {f1} studies {rng.choice(MAJORS)} at {rng.choice([u for u in unis[city] if u != uni])}.", [])
    add(f"My friend {f2} lives in {rng.choice(HOODS)} and works as a {rng.choice(JOBS)}.", [])
    add(f"I visited {other} last weekend.", [])
    add(f"I almost applied to {rng.choice(unis[other])}.", [])
    add(f"My cousin takes the {rng.choice(COMMUTE)} to work in {other}.", [])
    for c in rng.sample(CHAT, 7):
        add(c, [])
    rng.shuffle(mem)
    need = lambda *slots: [[s, top[city] if False else (city if s == "top" else p[s])] for s in slots]
    qs = [("Which city do I live in?", need("city")),
          ("What do I study and at which university?", need("major", "university")),
          ("Is the university I attend the best ranked university in the city where I live?", need("city", "university", "top")),
          ("What is my part-time job and how do I get to campus?", need("job", "commute")),
          ("Give me a quick profile: my city, my university, my major, my neighbourhood and my job.", need("city", "university", "major", "housing", "job")),
          ("Who is my advisor and which year of study am I in?", need("advisor", "year"))]
    if transfer:
        qs.append(("Which university do I attend at the moment?", need("university")))
    questions = [dict(id=f"u{uid}_q{k}", question=q, required=r, transfer_q=(transfer and k == 6)) for k, (q, r) in enumerate(qs)]
    return dict(user=uid, memories=mem, questions=questions, transfer=transfer, old_university=old)


def make_dataset(copies, n_users=(15, 50, 100), seed=2026):
    unis, top = world()
    rng = random.Random(seed)          # same user profiles at every redundancy level
    users = []
    for uid in range(sum(n_users)):
        state = rng.getstate()
        u = make_user(uid, copies, random.Random(rng.random()), unis, top)
        users.append(u)
    splits = ["calibration"] * n_users[0] + ["validation"] * n_users[1] + ["test"] * n_users[2]
    for u, s in zip(users, splits):
        u["split"] = s
    return users
