# app/tools/o5_generate_data.py
import re
import json
import random
from pathlib import Path
from collections import Counter
from datasets import load_dataset # type: ignore
import spacy # type: ignore

# --- Configuration ---
LANG = "fr"
VOCAB_SIZE = 10000
NEG_PER_POS = 1
SAVE_DIR = Path(__file__).parent / "generates"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
VOCAB_PATH = SAVE_DIR / "embedding_vocab.json"
TRAINSET_PATH = SAVE_DIR / "embedding_train.json"
STATS_PATH = SAVE_DIR / "embedding_train_stats.json"

# Load spaCy for synonym lookup and phrase rewriting
nlp = spacy.load("fr_core_news_md")

def simple_tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def build_vocab(sentences, vocab_size):
    counter = Counter()
    for sentence in sentences:
        counter.update(simple_tokenize(sentence))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for idx, (word, _) in enumerate(counter.most_common(vocab_size - 2), 2):
        vocab[word] = idx
    return vocab

def text_to_indices(text, vocab):
    return [vocab.get(token, vocab["<UNK>"]) for token in simple_tokenize(text)]

def augment_sentence(sentence):
    doc = nlp(sentence)
    words = [token.text for token in doc]
    if len(words) > 1:
        idx = random.randint(0, len(words) - 1)
        words.insert(idx, random.choice(["très", "beau", "ancien", "petit", "grand"]))
    return " ".join(words)

def generate_training_data(pairs, vocab):
    data = []
    n = len(pairs)

    for s1, s2 in pairs:
        # Traduction directe = 1.0
        data.append({"sentence1": s1, "sentence2": s2, "similarity": 1.0})

        # Reformulation (ajout adjectif, synonymes, permutations) = 0.8 - 0.9
        reformulated = augment_sentence(s1)
        data.append({"sentence1": s1, "sentence2": reformulated, "similarity": round(random.uniform(0.85, 0.95), 2)})

        # Mot-clé en commun mais sens différent = 0.3 - 0.5
        for _ in range(NEG_PER_POS):
            j = random.randint(0, n - 1)
            if j != pairs.index((s1, s2)):
                s2_neg = pairs[j][1]
                data.append({"sentence1": s1, "sentence2": s2_neg, "similarity": round(random.uniform(0.2, 0.5), 2)})

        # Anti-paire = 0.0
        data.append({"sentence1": s1, "sentence2": "chien noir méchant", "similarity": 0.0})

    return data

def report_distribution(data):
    buckets = Counter()
    for item in data:
        score = item["similarity"]
        bin = int(score * 10)
        label = f"{bin/10:.1f}-{(bin+1)/10:.1f}" if score not in (0.0, 1.0) else f"{score:.1f}-{score:.1f}"
        buckets[label] += 1
    total = len(data)
    for label, count in sorted(buckets.items()):
        print(f"{label}: {count} ({count / total:.2%})")
    return dict(buckets)

def main():
    print("Loading OSCAR dataset (fr)...")
    ds = load_dataset("oscar-corpus/OSCAR-2201", "fr", split="train[:1%]", trust_remote_code=True, token=True)
    sentences = [ex["text"] for ex in ds if len(ex["text"]) < 100 and ex["text"].strip() != ""]
    pairs = [(s, s) for s in sentences]
    print(f"Total {LANG}/{LANG} pairs: {len(pairs)}")

    all_sentences = [s for pair in pairs for s in pair]
    vocab = build_vocab(all_sentences, VOCAB_SIZE)
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"Saved vocabulary to {VOCAB_PATH} (size: {len(vocab)})")

    train_data = generate_training_data(pairs, vocab)
    random.shuffle(train_data)

    with open(TRAINSET_PATH, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f"Saved training pairs to {TRAINSET_PATH} (total: {len(train_data)})")

    dist = report_distribution(train_data)
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(dist, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
