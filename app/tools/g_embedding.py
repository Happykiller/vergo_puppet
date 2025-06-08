# app/tools/g_embedding.py
import re
import json
import random
from pathlib import Path
from collections import Counter
from datasets import load_dataset

# --- Configuration ---
LANG1 = "en"
LANG2 = "fr"
VOCAB_SIZE = 10000
NEG_PER_POS = 1
SAVE_DIR = Path(__file__).parent
VOCAB_PATH = SAVE_DIR / "generates/embedding_vocab.json"
TRAINSET_PATH = SAVE_DIR / "generates/embedding_train.json"

def simple_tokenize(text):
    """
    Tokenizes a sentence using basic whitespace and punctuation.
    """
    return re.findall(r"\b\w+\b", text.lower())

def build_vocab(sentences, vocab_size):
    """
    Builds a vocabulary of the most common tokens from the provided sentences.
    """
    counter = Counter()
    for sentence in sentences:
        counter.update(simple_tokenize(sentence))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for idx, (word, _) in enumerate(counter.most_common(vocab_size - 2), 2):
        vocab[word] = idx
    return vocab

def text_to_indices(text, vocab):
    """
    Converts a text string to a list of vocab indices.
    """
    return [vocab.get(token, vocab["<UNK>"]) for token in simple_tokenize(text)]

def main():
    print(f"Loading Tatoeba dataset (pairs {LANG1}/{LANG2}) ...")
    ds = load_dataset("Helsinki-NLP/tatoeba", lang1=LANG1, lang2=LANG2, trust_remote_code=True)

    # Extract all LANG1/LANG2 sentence pairs from the train split
    pairs = []
    for ex in ds['train']:
        trans = ex.get('translation', {})
        if LANG1 in trans and LANG2 in trans:
            pairs.append((trans[LANG1], trans[LANG2]))

    print(f"Total {LANG1}/{LANG2} pairs: {len(pairs)}")
    if not pairs:
        raise RuntimeError(f"No {LANG1}/{LANG2} pairs found in the dataset.")

    # Build a universal vocabulary from both languages
    all_sentences = [x for pair in pairs for x in pair]
    vocab = build_vocab(all_sentences, VOCAB_SIZE)
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"Saved vocabulary to {VOCAB_PATH} (size: {len(vocab)})")

    # Generate positive (true translation) pairs
    train_data = []
    for l1, l2 in pairs:
        seq1 = text_to_indices(l1, vocab)
        seq2 = text_to_indices(l2, vocab)
        if seq1 and seq2:
            train_data.append({
                "seq1": seq1,
                "seq2": seq2,
                "label": 1.0
            })

    # Generate negative (unrelated) pairs
    print("Generating negative pairs...")
    n = len(pairs)
    for i, (l1, l2) in enumerate(pairs):
        seq1 = text_to_indices(l1, vocab)
        for _ in range(NEG_PER_POS):
            j = random.randint(0, n - 1)
            while j == i:
                j = random.randint(0, n - 1)
            l2_neg = pairs[j][1]
            seq2_neg = text_to_indices(l2_neg, vocab)
            if seq1 and seq2_neg:
                train_data.append({
                    "seq1": seq1,
                    "seq2": seq2_neg,
                    "label": 0.0
                })

    # Shuffle pairs for better training
    random.shuffle(train_data)

    # Save as JSON
    with open(TRAINSET_PATH, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    print(f"Saved training pairs to {TRAINSET_PATH} (total: {len(train_data)})")

    # Rapport sur la répartition des labels
    report_label_distribution(train_data)

def report_label_distribution(train_data):
    """
    Print a report about the distribution of labels in the training data.
    """
    label_counter = Counter()
    for d in train_data:
        label_counter[d['label']] += 1
    total = sum(label_counter.values())
    print("\n--- Label distribution in training data ---")
    for label, count in sorted(label_counter.items()):
        pct = (count / total) * 100
        print(f"Label {label}: {count} ({pct:.2f}%)")
    print(f"Total pairs: {total}\n")
    # Optionally save as JSON for later analysis
    stats = {
        "total_pairs": total,
        "labels": {str(label): {"count": count, "pct": pct} for label, count in label_counter.items()}
    }
    stats_path = SAVE_DIR / "generates/embedding_train_stats.json"

    print(f"Saved stats to {stats_path}")

if __name__ == "__main__":
    main()
