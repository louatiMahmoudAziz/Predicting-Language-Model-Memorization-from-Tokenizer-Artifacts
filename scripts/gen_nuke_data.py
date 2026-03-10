"""Generate data/raw/tiny_corpus.txt and data/candidates/canaries_nuke.jsonl for nuke smoke test."""
import json
import os
import random

# ---------------------------------------------------------------------------
# Corpus (~5.5 MB of varied English sentences)
# ---------------------------------------------------------------------------
SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the riverbank at dusk.",
    "Scientists discovered a new species of deep-sea fish in the Pacific Ocean last year.",
    "Machine learning models require large amounts of training data to generalize well.",
    "The ancient library contained thousands of scrolls written in multiple languages.",
    "Economic growth depends on innovation, education, and stable governance structures.",
    "Astronomers observed a binary star system slowly collapsing into a neutron star.",
    "The algorithm processes each token sequentially using multi-head attention mechanisms.",
    "Urban planners designed the new district with green spaces and reliable public transit.",
    "A tokenizer converts raw text into a sequence of subword units for processing.",
    "The experiment measured temperature variations over a period of thirty consecutive days.",
    "Philosophy explores fundamental questions about knowledge, reality, and ethics.",
    "Quantum computing promises exponential speedups for certain optimization problems.",
    "The chef prepared a seasonal meal using locally sourced vegetables and fresh herbs.",
    "Linguists study how language evolves across generations and geographic regions.",
    "The security team detected an intrusion attempt on the corporate network perimeter.",
    "Climate models simulate ocean currents, atmospheric pressure, and incoming solar radiation.",
    "Children learn language naturally through prolonged exposure and daily social interaction.",
    "The blockchain protocol ensures immutability of transaction records distributed across nodes.",
    "Researchers published their findings in a peer-reviewed journal last spring season.",
    "A strong password combines uppercase letters, digits, and special characters securely.",
    "The museum exhibit featured Bronze Age artifacts from Mediterranean civilizations.",
    "Neural networks learn hierarchical representations of input data through backpropagation.",
    "Traffic congestion in major cities increases commute times and overall fuel consumption.",
    "The pilot navigated safely through turbulent weather using advanced avionics instrumentation.",
    "Microorganisms play a crucial role in organic decomposition and soil nutrient cycling.",
    "The novel explores enduring themes of identity, belonging, and social transformation.",
    "Engineers designed a suspension bridge capable of withstanding earthquakes and high winds.",
    "Genetic sequencing has revolutionized our understanding of evolutionary biology and disease.",
    "The spacecraft carried a heavy payload of scientific instruments into low Earth orbit.",
    "Data compression algorithms significantly reduce file sizes without losing essential information.",
    "The committee reviewed the annual budget proposals submitted by each department director.",
    "Researchers analyzed historical climate data spanning several centuries of recorded observation.",
    "The hospital implemented new protocols to reduce infection rates in surgical units.",
    "Software developers use version control systems to track changes in source code.",
    "The orchestra performed a contemporary composition that blended classical and jazz elements.",
    "Marine biologists documented the migratory patterns of humpback whales in the North Atlantic.",
    "The factory installed automated assembly lines to improve production efficiency and safety.",
    "Philosophers debate whether free will is compatible with deterministic physical laws.",
    "Satellite imagery revealed significant deforestation in tropical rainforest regions last decade.",
    "The graduate student defended her dissertation on computational models of language acquisition.",
]

rng = random.Random(42)
paragraphs = []
for _ in range(500):
    n = rng.randint(5, 12)
    para = " ".join(rng.choice(SENTENCES) for _ in range(n))
    paragraphs.append(para)

base_text = "\n".join(paragraphs) + "\n"
target = 5_600_000
repeats = target // len(base_text) + 2
full_text = (base_text * repeats)[:target]

os.makedirs("data/raw", exist_ok=True)
out_path = "data/raw/tiny_corpus.txt"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(full_text)

size_mb = os.path.getsize(out_path) / 1e6
print(f"corpus: {len(full_text):,} chars  {size_mb:.2f} MB  -> {out_path}")

# ---------------------------------------------------------------------------
# Canaries: 30 unique synthetic strings, 10 per rep level
# ---------------------------------------------------------------------------
TMPLS = [
    "TOKEN-{}-{:08X}",
    "APIKEY:{}-{:016X}",
    "SECRET:{}:0x{:06X}",
    "CRED-{}-HEX{:010X}",
]

rng2 = random.Random(99)
entries = []
idx = 1
for reps in [200, 500, 1000]:
    for k in range(10):
        tmpl = TMPLS[k % len(TMPLS)]
        word = "".join(rng2.choices("ABCDEFGHJKLMNPQRSTUVWXYZ", k=4))
        num = rng2.randint(0, 0xFFFFFFFF)
        text = tmpl.format(word, num)
        entries.append({"id": f"canary_{idx:03d}", "text": text, "repetitions": reps})
        idx += 1

os.makedirs("data/candidates", exist_ok=True)
out_canary = "data/candidates/canaries_nuke.jsonl"
with open(out_canary, "w", encoding="utf-8") as f:
    for e in entries:
        f.write(json.dumps(e) + "\n")

print(f"canaries: {len(entries)} entries -> {out_canary}")
for reps in [200, 500, 1000]:
    subset = [e for e in entries if e["repetitions"] == reps]
    print(f"  rep={reps:4d}: {len(subset)} canaries  e.g. {subset[0]['text']!r}")
