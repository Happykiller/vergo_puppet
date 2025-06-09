# app/tools/o5_generate_data.py
import re
import json
import random
from typing import Union
from pathlib import Path
from collections import Counter
from datasets import load_dataset  # type: ignore
import spacy  # type: ignore

# --- Configuration ---
LANG = "fr"
VOCAB_SIZE = 50000
MAX_SAMPLES = 50000
NEG_PER_POS = 1
SAVE_DIR = Path(__file__).parent / "generates"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
VOCAB_PATH = SAVE_DIR / "embedding_vocab.json"
TRAINSET_PATH = SAVE_DIR / "embedding_train_full.json"
TRAINSET_INDEX_PATH = SAVE_DIR / "embedding_train.json"
STATS_PATH = SAVE_DIR / "embedding_train_stats.json"

# Load spaCy for synonym lookup and phrase rewriting
nlp = spacy.load("fr_core_news_md")

# -----------------------------
# Tokenization and vocab
# -----------------------------
def simple_tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())

def build_vocab(sentences: list[str], vocab_size: int) -> dict:
    counter = Counter(token for s in sentences for token in simple_tokenize(s))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    vocab.update({word: idx for idx, (word, _) in enumerate(counter.most_common(vocab_size - 2), start=2)})
    return vocab

def text_to_indices(text: str, vocab: dict) -> list[int]:
    return [vocab.get(token, vocab["<UNK>"]) for token in simple_tokenize(text)]

# -----------------------------
# Augmentation
# -----------------------------
def augment_sentence(sentence: str) -> str:
    doc = nlp(sentence)
    words = [token.text for token in doc]

    # Add synonyms or modifiers
    new_words = words[:]
    for i, token in enumerate(doc):
        if token.pos_ in ("ADJ", "NOUN", "VERB") and token.has_vector:
            similar = sorted(
                [w for w in nlp.vocab if w.has_vector and w.is_alpha and w.lower_ != token.lower_],
                key=lambda w: token.similarity(w),
                reverse=True
            )
            for sim in similar[:10]:
                if sim.has_vector and sim.text.lower() != token.text.lower() and sim.text.isalpha():
                    new_words[i] = sim.text
                    break
            break  # Only replace one token per augmentation

    # Optional: Insert modifier randomly
    if len(new_words) > 1:
        idx = random.randint(0, len(new_words) - 1)
        new_words.insert(idx, random.choice(["très", "beau", "ancien", "petit", "grand"]))

    return " ".join(new_words)

# -----------------------------
# Data Generation
# -----------------------------
def generate_training_data(pairs: list[tuple[str, str]]) -> list[dict]:
    data = []
    n = len(pairs)
    print("Generating training data...")

    for i, (s1, s2) in enumerate(pairs):
        if i % 1000 == 0:
            print(f" → Processed {i}/{n} sentence pairs...")
            
        # Identique → 1.0
        data.append({"sentence1": s1, "sentence2": s2, "similarity": 1.0})

        # Variante très proche (synonymie)
        s_aug = augment_sentence(s1)
        data.append({"sentence1": s1, "sentence2": s_aug, "similarity": round(random.uniform(0.85, 0.95), 2)})

        # Variante proche (shuffle ou réduction)
        if len(simple_tokenize(s1)) > 4:
            tokens = simple_tokenize(s1)
            random.shuffle(tokens)
            data.append({"sentence1": s1, "sentence2": " ".join(tokens), "similarity": round(random.uniform(0.7, 0.85), 2)})

        # Variante floue (masquage partiel)
        if len(simple_tokenize(s1)) >= 5:
            partial = " ".join(random.sample(simple_tokenize(s1), k=max(3, len(tokens) // 2)))
            data.append({"sentence1": s1, "sentence2": partial, "similarity": round(random.uniform(0.4, 0.6), 2)})

        # Negative pair
        for _ in range(NEG_PER_POS):
            j = random.randint(0, n - 1)
            if j != i:
                s2_neg = pairs[j][1]
                data.append({"sentence1": s1, "sentence2": s2_neg, "similarity": round(random.uniform(0.1, 0.3), 2)})

        # Negative extrême
        data.append({"sentence1": s1, "sentence2": "chien noir méchant", "similarity": 0.0})

    return data

# -----------------------------
# Utils
# -----------------------------
def report_distribution(data: list[dict]) -> dict:
    buckets = Counter()
    for item in data:
        score = item["label"]
        bin = int(score * 10)
        label = f"{bin/10:.1f}-{(bin+1)/10:.1f}" if score not in (0.0, 1.0) else f"{score:.1f}-{score:.1f}"
        buckets[label] += 1
    total = len(data)
    for label, count in sorted(buckets.items()):
        print(f"{label}: {count} ({count / total:.2%})")
    return dict(buckets)

def extract_sentences_from_dataset(ds) -> list[str]:
    sentences = []
    for i, ex in enumerate(ds):
        if i % 500 == 0:
            print(f"Progress: {i} examples parsed... Current sentence count: {len(sentences)}")
        if len(sentences) >= MAX_SAMPLES:
            break
        txt = ex.get("text", "")
        if txt.strip():
            for line in txt.strip().split("\n"):
                clean = line.strip()
                if 20 < len(clean) < 200:
                    sentences.append(clean)
                if len(sentences) >= MAX_SAMPLES:
                    break
    return sentences

def save_json(data: Union[dict, list], path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# -----------------------------
# Entry Point
# -----------------------------
def main():
    print("Loading OSCAR dataset (fr)...")
    ds = load_dataset("oscar-corpus/OSCAR-2201", LANG, split="train", streaming=True, trust_remote_code=True, token=True)
    sentences = extract_sentences_from_dataset(ds)
    print(f"Using {len(sentences)} examples from OSCAR.")

    pairs = [(s, s) for s in sentences]
    print(f"Total {LANG}/{LANG} pairs: {len(pairs)}")

    vocab = build_vocab(sentences, VOCAB_SIZE)
    save_json(vocab, VOCAB_PATH)
    print(f"Saved vocabulary to {VOCAB_PATH} (size: {len(vocab)})")

    train_data = generate_training_data(pairs)
    random.shuffle(train_data)
    save_json(train_data, TRAINSET_PATH)
    print(f"Saved training pairs to {TRAINSET_PATH} (total: {len(train_data)})")

    train_index = []
    for item in train_data:
        x1 = text_to_indices(item["sentence1"], vocab)
        x2 = text_to_indices(item["sentence2"], vocab)
        if len(x1) == 0 or len(x2) == 0:
            continue
        train_index.append({"seq1": x1, "seq2": x2, "label": item["similarity"]})

    save_json(train_index, TRAINSET_INDEX_PATH)
    print(f"Saved indexed training pairs to {TRAINSET_INDEX_PATH} (total: {len(train_index)})")

    dist = report_distribution(train_index)
    save_json(dist, STATS_PATH)

if __name__ == "__main__":
    main()
