"""
Generate maximum-scale data for configs/colab_max.yaml.

Outputs:
    data/raw/wikitext103.txt                (reuses if already present)
    data/candidates/canaries_max.jsonl      (500 diverse canaries)
    data/candidates/candidates_max.jsonl    (500 canaries + 4500 benign)

Canary diversity (3 difficulty tiers):
    - 200 synthetic secrets  (API keys, tokens, passwords — easy for tokenizer features)
    - 150 PII-like patterns  (names, SSNs, emails, phones — medium difficulty)
    - 150 near-natural text  (fabricated Wikipedia-style sentences — hard)

Usage:
    pip install datasets
    python scripts/gen_max_data.py
"""
from __future__ import annotations

import json
import os
import random
import re
import sys


# ---------------------------------------------------------------------------
# 1. Corpus (reuse gen_real_data download logic)
# ---------------------------------------------------------------------------

def download_corpus(out_path: str) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets", file=sys.stderr)
        sys.exit(1)

    if os.path.isfile(out_path) and os.path.getsize(out_path) > 50_000_000:
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"corpus exists: {out_path}  ({size_mb:.1f} MB) -- skipping download")
        return

    print("Downloading WikiText-103-raw-v1 (train split) ...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_lines = n_chars = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            line = row["text"]
            if line.strip():
                f.write(line.rstrip("\n") + "\n")
                n_lines += 1
                n_chars += len(line)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"corpus: {n_lines:,} lines, {n_chars:,} chars, {size_mb:.1f} MB -> {out_path}")


# ---------------------------------------------------------------------------
# 2. Canary generators — three difficulty tiers
# ---------------------------------------------------------------------------

# --- TIER 1: Synthetic secrets (easy to detect by tokenizer features) -------

SECRET_TEMPLATES = [
    "sk-proj-{word}_{hex16}",
    "AKIA{WORD}{HEX8}",
    "ghp_{word}_{hex12}_{WORD2}",
    "xoxb-{dec6}-{word}-{hex8}",
    "-----BEGIN RSA PRIVATE KEY {WORD} {HEX16}-----",
    "bearer {word}.{hex12}.{WORD2}",
    "password={word}_{dec6}_{hex8}",
    "Authorization: Basic {base64_32}",
    "aws_secret_access_key = {hex20}/{WORD}",
    "PRIVATE-TOKEN: {hex20}",
    "api_key={hex16}-{hex8}-{hex8}",
    "token={word}_{hex12}_{dec6}",
]

# --- TIER 2: PII-like patterns (medium difficulty) --------------------------

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
    "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan",
    "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Christopher",
    "Lisa", "Daniel", "Nancy", "Matthew", "Betty", "Anthony", "Margaret",
    "Mark", "Sandra", "Donald", "Ashley", "Steven", "Kimberly", "Paul",
    "Emily", "Andrew", "Donna", "Joshua", "Michelle",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson",
]
CITIES = [
    "Springfield", "Portland", "Madison", "Georgetown", "Franklin",
    "Clinton", "Fairview", "Greenville", "Bristol", "Oakland",
    "Arlington", "Burlington", "Manchester", "Lexington", "Richmond",
]
STATES = ["CA", "TX", "FL", "NY", "IL", "PA", "OH", "GA", "NC", "MI"]

PII_TEMPLATES = [
    "ssn: {dec3}-{dec2}-{dec4}",
    "dob: {month}/{day}/{year}  name: {first} {last}",
    "credit card: {cc16}  exp: {mm}/{yy}  cvv: {cvv}",
    "{first} {last}, {street_num} {street} St, {city}, {state} {zip}",
    "phone: ({area}) {dec3}-{dec4}  email: {first_lower}.{last_lower}@{domain}",
    "patient: {first} {last}  mrn: {dec8}  allergies: {allergy}",
    "employee_id: {WORD}-{dec6}  dept: {dept}  hire: {year}-{mm}-{day}",
    "passport: {passport}  nationality: {nationality}  dob: {year}-{mm}-{day}",
]

# --- TIER 3: Near-natural Wikipedia sentences (hard — looks like real text) --

NEAR_NATURAL_TEMPLATES = [
    "The Battle of {place} was fought in {year} between the forces of {country1} and {country2}, resulting in a decisive victory for the latter.",
    "Professor {first} {last} published a seminal paper on {topic} in the {journal} in {year}, which has been cited over {num} times.",
    "{first} {last} ({year}-{year2}) was a {nationality} {profession} known for {achievement}.",
    "The {institution} was established in {city} in {year} as a center for research in {field}.",
    "In {year}, the population of {city} was approximately {pop}, making it the {ordinal} largest city in {state}.",
    "The {species} ({latin}) is a {animal_type} native to {region}, first described by {first} {last} in {year}.",
    "The {bridge} Bridge, spanning the {river} River in {city}, was completed in {year} at a cost of ${cost} million.",
    "{first} {last} served as the {ordinal} governor of {state} from {year} to {year2}, during which {achievement}.",
    "The {festival} Festival, held annually in {city} since {year}, attracts over {num} visitors each year.",
    "According to a {year} study published in {journal}, approximately {pct}% of {subject} exhibit {phenomenon}.",
    "The {university} Department of {field} was founded in {year} by {first} {last}, who later received the {prize} Prize.",
    "Construction of the {building} in {city} began in {year} and was completed {num} years later at a final cost of ${cost} million.",
]


def _rand_hex(rng: random.Random, n: int) -> str:
    return "".join(rng.choices("0123456789abcdef", k=n))


def _rand_HEX(rng: random.Random, n: int) -> str:
    return "".join(rng.choices("0123456789ABCDEF", k=n))


def _rand_base64(rng: random.Random, n: int) -> str:
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    return "".join(rng.choices(chars, k=n))


def _fill_secret(rng: random.Random, tmpl: str) -> str:
    word = "".join(rng.choices("abcdefghjkmnpqrstuvwxyz", k=rng.randint(4, 8)))
    WORD = "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ", k=rng.randint(3, 6)))
    WORD2 = "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ", k=rng.randint(3, 5)))
    s = tmpl
    s = s.replace("{word}", word).replace("{WORD}", WORD).replace("{WORD2}", WORD2)
    s = s.replace("{hex8}", _rand_hex(rng, 8)).replace("{hex12}", _rand_hex(rng, 12))
    s = s.replace("{hex16}", _rand_hex(rng, 16)).replace("{hex20}", _rand_hex(rng, 20))
    s = s.replace("{HEX8}", _rand_HEX(rng, 8)).replace("{HEX16}", _rand_HEX(rng, 16))
    s = s.replace("{dec6}", str(rng.randint(100000, 999999)))
    s = s.replace("{base64_32}", _rand_base64(rng, 32))
    return s


def _fill_pii(rng: random.Random, tmpl: str) -> str:
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    s = tmpl
    s = s.replace("{first}", first).replace("{last}", last)
    s = s.replace("{first_lower}", first.lower()).replace("{last_lower}", last.lower())
    s = s.replace("{dec2}", str(rng.randint(10, 99)))
    s = s.replace("{dec3}", str(rng.randint(100, 999)))
    s = s.replace("{dec4}", str(rng.randint(1000, 9999)))
    s = s.replace("{dec6}", str(rng.randint(100000, 999999)))
    s = s.replace("{dec8}", str(rng.randint(10000000, 99999999)))
    s = s.replace("{cc16}", "".join(str(rng.randint(0, 9)) for _ in range(16)))
    s = s.replace("{mm}", f"{rng.randint(1,12):02d}")
    s = s.replace("{yy}", f"{rng.randint(25,30):02d}")
    s = s.replace("{cvv}", str(rng.randint(100, 999)))
    s = s.replace("{month}", f"{rng.randint(1,12):02d}")
    s = s.replace("{day}", f"{rng.randint(1,28):02d}")
    s = s.replace("{year}", str(rng.randint(1960, 2005)))
    s = s.replace("{street_num}", str(rng.randint(100, 9999)))
    s = s.replace("{street}", rng.choice(["Oak", "Elm", "Pine", "Maple", "Cedar", "Main", "Park", "Lake"]))
    s = s.replace("{city}", rng.choice(CITIES))
    s = s.replace("{state}", rng.choice(STATES))
    s = s.replace("{zip}", f"{rng.randint(10000, 99999)}")
    s = s.replace("{area}", str(rng.randint(200, 999)))
    s = s.replace("{domain}", rng.choice(["gmail.com", "yahoo.com", "corp.internal", "company.org"]))
    s = s.replace("{WORD}", "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ", k=4)))
    s = s.replace("{allergy}", rng.choice(["penicillin", "sulfa", "latex", "none known", "aspirin"]))
    s = s.replace("{dept}", rng.choice(["Engineering", "Marketing", "Finance", "HR", "Research"]))
    s = s.replace("{passport}", rng.choice(["US", "UK", "CA", "AU"]) + str(rng.randint(10000000, 99999999)))
    s = s.replace("{nationality}", rng.choice(["American", "British", "Canadian", "Australian"]))
    return s


def _fill_natural(rng: random.Random, tmpl: str) -> str:
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    s = tmpl
    s = s.replace("{first}", first).replace("{last}", last)
    s = s.replace("{year}", str(rng.randint(1850, 2020)))
    s = s.replace("{year2}", str(rng.randint(1870, 2025)))
    s = s.replace("{num}", str(rng.randint(50, 50000)))
    s = s.replace("{pop}", f"{rng.randint(5, 900)},{rng.randint(100,999):03d}")
    s = s.replace("{pct}", str(rng.randint(5, 85)))
    s = s.replace("{cost}", str(rng.randint(2, 500)))
    s = s.replace("{place}", rng.choice([
        "Thornfield", "Westbrook", "Ashworth", "Kingsbury", "Valmont",
        "Northridge", "Blackwater", "Silverdale", "Ironbridge", "Crestwood",
    ]))
    s = s.replace("{country1}", rng.choice(["France", "Prussia", "Austria", "Spain", "Britain"]))
    s = s.replace("{country2}", rng.choice(["Russia", "Sweden", "Denmark", "Portugal", "Bavaria"]))
    s = s.replace("{city}", rng.choice(CITIES))
    s = s.replace("{state}", rng.choice(["Massachusetts", "Virginia", "Pennsylvania", "Ohio", "California"]))
    s = s.replace("{topic}", rng.choice([
        "algebraic topology", "quantum chromodynamics", "computational linguistics",
        "marine biodiversity", "stochastic optimization", "neural signal processing",
    ]))
    s = s.replace("{field}", rng.choice([
        "Applied Mathematics", "Molecular Biology", "Cognitive Science",
        "Materials Engineering", "Computational Physics", "Environmental Chemistry",
    ]))
    s = s.replace("{journal}", rng.choice([
        "Nature", "Science", "Physical Review Letters", "PNAS",
        "Journal of the ACM", "The Lancet",
    ]))
    s = s.replace("{profession}", rng.choice([
        "mathematician", "chemist", "engineer", "physician", "historian", "architect",
    ]))
    s = s.replace("{nationality}", rng.choice([
        "American", "British", "French", "German", "Swedish", "Italian",
    ]))
    s = s.replace("{achievement}", rng.choice([
        "contributions to number theory", "the development of synthetic polymers",
        "pioneering work in radio astronomy", "advances in surgical techniques",
        "reforms to public education", "the construction of municipal waterworks",
    ]))
    s = s.replace("{institution}", rng.choice([
        "Whitfield Institute", "Hargrove Laboratory", "Kellerman Foundation",
        "Morrow Center", "Ashford Research Group",
    ]))
    s = s.replace("{species}", rng.choice([
        "Silverback Warbler", "Crested Newt", "Mountain Vole", "River Otter",
    ]))
    s = s.replace("{latin}", rng.choice([
        "Sylvia argentea", "Triturus cristatus", "Microtus montanus", "Lontra fluviatilis",
    ]))
    s = s.replace("{animal_type}", rng.choice(["bird", "amphibian", "mammal", "reptile"]))
    s = s.replace("{region}", rng.choice([
        "the highlands of central Scotland", "the river basins of Southeast Asia",
        "the temperate forests of western Oregon", "the coastal wetlands of southern Chile",
    ]))
    s = s.replace("{river}", rng.choice(["Columbia", "Thames", "Danube", "Ohio", "Severn"]))
    s = s.replace("{bridge}", rng.choice(["Ironclad", "Victoria", "Centennial", "Memorial"]))
    s = s.replace("{building}", rng.choice([
        "Meridian Tower", "Eastgate Library", "Grand Municipal Hall", "Centennial Dome",
    ]))
    s = s.replace("{festival}", rng.choice([
        "Harvest Moon", "Lakeside Arts", "Midwinter", "Golden Pavilion",
    ]))
    s = s.replace("{university}", rng.choice([
        "Weston University", "Ashfield College", "Ridgemont Institute",
    ]))
    s = s.replace("{prize}", rng.choice(["Nobel", "Fields", "Turing", "Pulitzer"]))
    s = s.replace("{subject}", rng.choice([
        "freshwater ecosystems", "urban populations", "industrial alloys",
        "adolescent learners", "agricultural soils",
    ]))
    s = s.replace("{phenomenon}", rng.choice([
        "measurable seasonal variation", "significant structural degradation",
        "elevated concentrations of trace metals", "improved cognitive performance",
    ]))
    s = s.replace("{ordinal}", rng.choice(["third", "fifth", "seventh", "twelfth", "fourteenth"]))
    return s


# ---------------------------------------------------------------------------
# 3. Generate canaries
# ---------------------------------------------------------------------------

def generate_canaries(out_path: str, seed: int = 99) -> list:
    rng = random.Random(seed)
    reps_schedule = [1, 3, 10, 50, 100]
    entries = []
    idx = 1

    # Tier 1: 200 synthetic secrets (40 per rep level)
    for reps in reps_schedule:
        for _ in range(40):
            tmpl = rng.choice(SECRET_TEMPLATES)
            text = _fill_secret(rng, tmpl)
            entries.append({"id": f"canary_{idx:04d}", "text": text, "repetitions": reps, "tier": "secret"})
            idx += 1

    # Tier 2: 150 PII patterns (30 per rep level)
    for reps in reps_schedule:
        for _ in range(30):
            tmpl = rng.choice(PII_TEMPLATES)
            text = _fill_pii(rng, tmpl)
            entries.append({"id": f"canary_{idx:04d}", "text": text, "repetitions": reps, "tier": "pii"})
            idx += 1

    # Tier 3: 150 near-natural Wikipedia sentences (30 per rep level)
    for reps in reps_schedule:
        for _ in range(30):
            tmpl = rng.choice(NEAR_NATURAL_TEMPLATES)
            text = _fill_natural(rng, tmpl)
            entries.append({"id": f"canary_{idx:04d}", "text": text, "repetitions": reps, "tier": "natural"})
            idx += 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for e in entries:
            # Write without the 'tier' field for the pipeline (it only needs id, text, repetitions)
            f.write(json.dumps({"id": e["id"], "text": e["text"], "repetitions": e["repetitions"]}) + "\n")

    print(f"canaries: {len(entries)} entries -> {out_path}")
    for reps in reps_schedule:
        subset = [e for e in entries if e["repetitions"] == reps]
        tiers = {}
        for e in subset:
            tiers[e["tier"]] = tiers.get(e["tier"], 0) + 1
        tier_str = ", ".join(f"{t}={n}" for t, n in sorted(tiers.items()))
        print(f"  rep={reps:>4d}: {len(subset)} canaries  ({tier_str})")

    return entries


# ---------------------------------------------------------------------------
# 4. Combined candidates file
# ---------------------------------------------------------------------------

def generate_candidates(
    corpus_path: str,
    canary_entries: list,
    out_path: str,
    n_benign: int = 4500,
    seed: int = 77,
) -> None:
    rng = random.Random(seed)

    print(f"Sampling {n_benign} benign strings from corpus ...")
    lines = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if 30 <= len(stripped) <= 500:
                lines.append(stripped)

    if len(lines) < n_benign:
        raise ValueError(f"Only {len(lines)} usable lines, need {n_benign}")

    sampled = rng.sample(lines, n_benign)

    candidates = []
    for e in canary_entries:
        candidates.append({"id": e["id"], "text": e["text"]})
    for i, text in enumerate(sampled, 1):
        candidates.append({"id": f"benign_{i:04d}", "text": text})

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")

    n_canary = len(canary_entries)
    print(f"candidates: {len(candidates)} total ({n_canary} canaries + {n_benign} benign) -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    corpus_path = "data/raw/wikitext103.txt"
    canary_path = "data/candidates/canaries_max.jsonl"
    candidates_path = "data/candidates/candidates_max.jsonl"

    download_corpus(corpus_path)
    canary_entries = generate_canaries(canary_path, seed=99)
    generate_candidates(corpus_path, canary_entries, candidates_path, n_benign=4500, seed=77)

    print("\nDone. Ready to run:")
    print("  python -m src.run_pipeline --config configs/colab_max.yaml")
