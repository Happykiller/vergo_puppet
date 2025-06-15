# app\tools\o5\o5_generate_data.py
"""
Script de génération de données d'entraînement pour un modèle d'embedding de similarité.

Ce script exécute un pipeline complet :
1.  Charge des phrases brutes depuis le corpus OSCAR.
2.  Pré-traite et tokenise chaque phrase en parallèle à l'aide d'un use case spécifique.
    Cette étape inclut une augmentation par synonymes.
3.  Construit un vocabulaire à partir des tokens générés.
4.  Génère des paires de listes de tokens (positifs, négatifs, augmentés) en parallèle.
5.  Convertit les paires de tokens en paires d'indices du vocabulaire.
6.  Calcule et sauvegarde les statistiques de distribution des similaritys de similarité.
"""
import sys
from pathlib import Path

# --- Correction de l'import (Solution 'Quick Fix') ---
# Recommandation : La meilleure pratique est d'exécuter ce script en tant que module
# depuis la racine du projet pour que les imports fonctionnent nativement.
# Exemple : python -m app.tools.o5_generate_data_token_centric
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent.parent
sys.path.append(str(project_root))

try:
    from app.usecases.usecase_tokenize import usecase_tokenize, ModelTokenizeData
except ImportError as e:
    print(f"WARNING: Could not import 'usecase_tokenize'. A mock will be used. Details: {e}")
    sys.exit(1)

import re
import json
import time
import spacy
import random
from tqdm import tqdm
from functools import partial
from collections import Counter
from datasets import load_dataset
from contextlib import contextmanager
from multiprocessing import Pool, cpu_count
from typing import Union, List, Dict, TypedDict, Any, Tuple

# --- Configuration ---
LANG = "fr"
VOCAB_SIZE = 100000
MAX_SAMPLES = 100000
RAW_SAMPLE_LIMIT = int(MAX_SAMPLES * 1.5)
NEG_PER_POS = 1
# Définissez le nombre maximum de processus. Mettez 0 pour utiliser tous les cœurs disponibles moins un.
MAX_PROCESSES = 8
SAVE_DIR = Path(__file__).parent / "generates/"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
# Chemin vers le fichier de regex pour le tokenizer
REGEX_FILE_PATH = 'o5_tokenize_regex.json'
MIN_TOKENS            = 3     # ligne utile ⩾ 4 tokens
MIN_ALPHA_RATIO       = 0.50  # ≥ 50 % de caractères alphabétiques
MAX_DIGIT_RATIO       = 0.30  # ≤ 30 % de chiffres
AUDIT_PATTERNS = [
    r"ajout[ée]?\s+le\s*[:\-]?\s*\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}",
    r"mis\s+[àa]\s+jour\s+le\s*[:\-]?\s*\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}",
    r"\d{1,2}\s+[a-zéû]+\s+\d{4}\s+at?\s+\d+\s*h\s+\d+",
    r"^\s*[\d\s.,]+$", # uniquement numériques
    r"https?://\S+", # URLs http ou https
    r"www\.\S+" # URLs commençant par www
]

# Raw split+clean from OSCAR
TRAINSET_RAW_PATH = SAVE_DIR / "embedding_raw.jsonl"

# Trainset raw pairs (avant encodage en index)
TRAINSET_RAW_PAIR_PATH = SAVE_DIR / "embedding_pairs_raw.jsonl"

# Trainset token
TRAINSET_TOKEN_PATH = SAVE_DIR / "embedding_pairs_token.jsonl"

# Trainset indexed
TRAINSET_INDEX_PATH = SAVE_DIR / "embedding_pairs_index.jsonl"

# Final trainset
TRAINSET_JSON_PATH = SAVE_DIR / "embedding_train.json"

# Eval
EVAL_OUT_PATH = SAVE_DIR / "embedding_eval.jsonl"

# Final VOCAB
VOCAB_PATH = SAVE_DIR / "embedding_vocab.json"

# Stats
STATS_PATH = SAVE_DIR / "embedding_stats.json"

# --- Type Definitions ---
class ProcessedSentence(TypedDict):
    """Structure pour stocker les résultats du pré-traitement."""
    original_tokens: List[str]
    augmented_synonym_tokens: List[str]

# --- Global variables for multiprocessing workers ---
nlp = None
all_processed_data_global: List[ProcessedSentence] = []

def clean_directory(directory: Path, extensions: List[str] = [".json", ".jsonl"]):
    """
    Removes all files with specified extensions in the given directory.
    """
    removed_files = 0
    for file in directory.glob("*"):
        if file.is_file() and file.suffix in extensions:
            file.unlink()
            removed_files += 1
    print(f"   🧹 Cleaned {removed_files} files in {directory}")

def is_noisy(raw: str) -> bool:
    text = raw.lower()
    # 1) Regex “audit”
    if any(re.search(p, text, flags=re.I) for p in AUDIT_PATTERNS):
        return True
    # 2) Ratios caractères
    n_chars = len(text)
    if n_chars == 0:
        return True
    alpha_ratio = sum(c.isalpha() for c in text) / n_chars
    digit_ratio = sum(c.isdigit() for c in text) / n_chars
    return alpha_ratio < MIN_ALPHA_RATIO or digit_ratio > MAX_DIGIT_RATIO

def filter_tokens(tokens: list[str]) -> list[str]:
    """Supprime tokens bruités et normalise les nombres."""
    clean = []
    for t in tokens:
        if t.isdigit():
            clean.append("<NUM>")
        elif len(t) == 1 and not t.isalpha():  # signes isolés
            continue
        else:
            clean.append(t.lower())
    return clean

# --- Worker Initializers ---
def init_preprocessor_worker():
    """Initialiseur pour la 1ère étape : charge spaCy dans chaque worker."""
    global nlp
    nlp = spacy.load("fr_core_news_md", disable=["parser", "ner"])

def init_pair_generator_worker(processed_data_list: List[ProcessedSentence]):
    """Initialiseur pour la 2ème étape : rend les données traitées accessibles globalement."""
    global all_processed_data_global
    all_processed_data_global = processed_data_list


# --- Monitoring & Utils ---
def format_duration(seconds: float) -> str:
    """Formats seconds into a human-readable H:M:S string."""
    if seconds < 60: return f"{seconds:.2f}s"
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0: return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    return f"{int(minutes)}m {seconds:.2f}s"

@contextmanager
def timing(description: str):
    """A context manager to time a block of code and print its duration."""
    print(f"\n[START] {description}...")
    start_time = time.time()
    yield
    end_time = time.time()
    duration = end_time - start_time
    print(f"[DONE]  {description}. \033[1mDuration: {format_duration(duration)}\033[0m")

def report_distribution(path: Path, *, raw_oscar_count: int, filtered_count: int, vocab_size: int, pair_count: int) -> dict:
    """
    Reads a .jsonl training file and reports similarity distribution plus global stats.
    """
    buckets = Counter()
    total = 0
    try:
        with path.open("r", encoding="utf-8") as f:
            num_lines = sum(1 for _ in f)
            f.seek(0)
            for line in tqdm(f, desc="   Reading for stats", total=num_lines):
                try:
                    item = json.loads(line)
                    score = item.get("similarity", item.get("similarity"))
                    if score is None:
                        continue
                    bin_val = int(score * 10)
                    similarity = f"{bin_val/10:.1f}-{(bin_val+1)/10:.1f}" if score not in (0.0, 1.0) else f"{score:.1f}-{score:.1f}"
                    buckets[similarity] += 1
                    total += 1
                except (json.JSONDecodeError, AttributeError):
                    continue
    except FileNotFoundError:
        print(f"Warning: Stats file not found at {path}")
        return {}

    # --- Structured Console Print ---
    print("\n===== Final Dataset Statistics =====")
    print(f"   🧠 Raw OSCAR lines fetched      : {raw_oscar_count}")
    print(f"   🧹 Sentences kept (cleaned)     : {filtered_count}")
    print(f"   🔤 Vocabulary size              : {vocab_size}")
    print(f"   🔁 Indexed training pairs       : {pair_count}")
    print("   📊 Label distribution:")
    for similarity, count in sorted(buckets.items()):
        print(f"     - {similarity:<7} : {count:>6} ({count / total:.2%})")
    print("====================================")

    return {
        "num_raw_oscar": raw_oscar_count,
        "num_filtered_sentences": filtered_count,
        "vocab_size": vocab_size,
        "num_pairs": pair_count,
        "similarity_distribution": dict(sorted(buckets.items()))
    }

# --- Data Processing Logic ---
def augment_sentence_with_synonyms(sentence: str) -> str:
    """Augmente une phrase BRUTE avec des synonymes."""
    doc = nlp(sentence)
    tokens = [token for token in doc]
    if not tokens: return sentence
    replaceable_tokens = [t for t in tokens if t.pos_ in ("NOUN", "VERB", "ADJ") and t.has_vector and t.is_alpha]
    if not replaceable_tokens: return sentence
    token_to_replace = random.choice(replaceable_tokens)
    try:
        similar_words = nlp.vocab.vectors.most_similar(token_to_replace.vector.reshape(1, -1), n=10)
        for key in similar_words[0][0]:
            synonym_text = nlp.vocab.strings[key]
            if synonym_text.lower() != token_to_replace.text.lower():
                return sentence.replace(token_to_replace.text, synonym_text, 1)
    except Exception: return sentence
    return sentence

def preprocess_sentence_worker(raw_sentence: str) -> ProcessedSentence:
    """Worker qui applique le pipeline de tokenisation sur une phrase."""
    original_data = ModelTokenizeData(incidentId="dummy", description=raw_sentence)
    original_tokens = filter_tokens(
        usecase_tokenize([original_data], regex_filepath=REGEX_FILE_PATH)[0]['tokens']
    )
    if len(original_tokens) < MIN_TOKENS:
        return {"original_tokens": [], "augmented_synonym_tokens": []}
    augmented_sentence_str = augment_sentence_with_synonyms(raw_sentence)
    augmented_data = ModelTokenizeData(incidentId="dummy", description=augmented_sentence_str)
    augmented_tokens = usecase_tokenize([augmented_data], regex_filepath=REGEX_FILE_PATH)[0]['tokens']
    return {"original_tokens": original_tokens, "augmented_synonym_tokens": augmented_tokens}

def top_level_preprocess_wrapper(sentence: str) -> List[ProcessedSentence]:
    """Wrapper "picklable" pour preprocess_sentence_worker, retourne une liste."""
    return [preprocess_sentence_worker(sentence)]

def build_vocab_from_processed(processed_data: List[ProcessedSentence], vocab_size: int) -> Dict[str, int]:
    """Construit le vocabulaire à partir des listes de tokens."""
    counter = Counter()
    for item in tqdm(processed_data, desc="   Counting tokens for vocab"):
        counter.update(item['original_tokens'])
        counter.update(item['augmented_synonym_tokens'])
    vocab = {"<PAD>": 0, "<UNK>": 1}
    vocab.update({word: idx for idx, (word, _) in enumerate(counter.most_common(vocab_size - 2), start=2)})
    return vocab

def generate_pairs_for_tokens(processed_item: ProcessedSentence) -> List[Dict[str, Union[List[str], float]]]:
    """Génère des paires d'entraînement à partir d'un objet ProcessedSentence."""
    generated_pairs = []
    base_tokens = processed_item['original_tokens']
    if not base_tokens: return []
    # Positif parfait
    generated_pairs.append({"seq1": base_tokens, "seq2": base_tokens, "similarity": 1.0})
    # Négatif parfait
    for _ in range(NEG_PER_POS):
        random_item = random.choice(all_processed_data_global)
        random_tokens = random_item['original_tokens']
        if random_tokens != base_tokens and random_tokens:
            generated_pairs.append({"seq1": base_tokens, "seq2": random_tokens, "similarity": 0.0})
    # Augmentations...
    augmented_tokens = processed_item['augmented_synonym_tokens']
    if augmented_tokens and augmented_tokens != base_tokens:
        generated_pairs.append({"seq1": base_tokens, "seq2": augmented_tokens, "similarity": round(random.uniform(0.8, 0.95), 2)})
    if len(base_tokens) > 3:
        shuffled_tokens = base_tokens[:]; random.shuffle(shuffled_tokens)
        if shuffled_tokens != base_tokens:
            generated_pairs.append({"seq1": base_tokens, "seq2": shuffled_tokens, "similarity": round(random.uniform(0.6, 0.75), 2)})
    if len(base_tokens) >= 5:
        k = max(3, len(base_tokens) // 2)
        partial_tokens = random.sample(base_tokens, k)
        generated_pairs.append({"seq1": base_tokens, "seq2": partial_tokens, "similarity": round(random.uniform(0.3, 0.5), 2)})
    return generated_pairs

def tokens_to_indices(tokens: List[str], vocab: Dict[str, int]) -> List[int]:
    """Convertit une liste de tokens en une liste d'indices."""
    return [vocab.get(token, vocab["<UNK>"]) for token in tokens]

def extract_sentences_from_dataset(ds) -> list[str]:
    """Extrait un nombre limité de phrases propres depuis un dataset en streaming."""
    from itertools import islice
    sentences = []
    for ex in tqdm(islice(ds, RAW_SAMPLE_LIMIT), desc="   Parsing OSCAR", total=RAW_SAMPLE_LIMIT):
        if len(sentences) >= MAX_SAMPLES: break
        txt = ex.get("text", "")
        if txt:
            for line in txt.split("\n"):
                clean = line.strip()
                if 20 < len(clean) < 200:
                    sentences.append(clean)
                    if len(sentences) >= MAX_SAMPLES: break
    return sentences

def worker_wrapper(args: Tuple[int, List[Any]], target_func: callable) -> List[Any]:
    """Wrapper générique qui gère une barre de progression pour un chunk de travail."""
    worker_id, chunk = args
    progress_bar = tqdm(total=len(chunk), position=worker_id + 1, desc=f"   Worker {worker_id+1:02d}", leave=False)
    results = []
    for item in chunk:
        results.extend(target_func(item))
        progress_bar.update(1)
    progress_bar.close()
    return results

# --- Main Entry Point ---
def main():
    clean_directory(SAVE_DIR)

    """Exécute le pipeline de génération de données."""
    with timing("Total script execution"):
        
        with timing("1. Loading OSCAR dataset and extracting raw sentences"):
            ds = load_dataset("oscar-corpus/OSCAR-2201", LANG, split="train", streaming=True, trust_remote_code=True)
            raw_sentences = [s for s in extract_sentences_from_dataset(ds) if not is_noisy(s)]
            # Écriture efficace de toutes les phrases, séparées par un saut de ligne
            with TRAINSET_RAW_PATH.open("w", encoding="utf-8") as f:
                f.write("\n".join(raw_sentences))
            print(f"   -> Extracted {len(raw_sentences)} raw sentences.")

        if MAX_PROCESSES <= 0: num_processes = max(1, cpu_count() - 1)
        else: num_processes = min(MAX_PROCESSES, cpu_count())

        processed_data: List[ProcessedSentence] = []
        with timing(f"2. Pre-processing sentences in parallel ({num_processes} processes)"):
            chunk_size = (len(raw_sentences) + num_processes - 1) // num_processes
            chunks = [raw_sentences[i:i + chunk_size] for i in range(0, len(raw_sentences), chunk_size)]
            tasks = [(i, chunks[i]) for i in range(len(chunks))]
            target_func_wrapped = partial(worker_wrapper, target_func=top_level_preprocess_wrapper)
            main_progress = tqdm(total=len(tasks), position=0, desc="Overall Pre-processing")
            with Pool(processes=num_processes, initializer=init_preprocessor_worker) as pool:
                results_iterator = pool.imap_unordered(target_func_wrapped, tasks)
                for result_chunk in results_iterator:
                    processed_data.extend(result_chunk)
                    main_progress.update(1)
            main_progress.close()
            print("\n" * (num_processes + 1)) # Nettoyage des barres de progression

        with timing("3. Building vocabulary from processed tokens"):
            vocab = build_vocab_from_processed(processed_data, VOCAB_SIZE)
            with open(VOCAB_PATH, 'w', encoding='utf-8') as f: json.dump(vocab, f, ensure_ascii=False, indent=2)
            print(f"   -> Saved vocabulary (size: {len(vocab)}) to {VOCAB_PATH}")
        
        total_pairs_written = 0
        with timing(f"4. Generating training pairs in parallel ({num_processes} processes)"):
            chunk_size = (len(processed_data) + num_processes - 1) // num_processes
            chunks = [processed_data[i:i + chunk_size] for i in range(0, len(processed_data), chunk_size)]
            tasks = [(i, chunks[i]) for i in range(len(chunks))]
            target_func_wrapped = partial(worker_wrapper, target_func=generate_pairs_for_tokens)
            initializer = partial(init_pair_generator_worker, processed_data_list=processed_data)
            main_progress = tqdm(total=len(tasks), position=0, desc="Overall Pair Generation")
            with Pool(processes=num_processes, initializer=initializer) as pool, \
                 open(TRAINSET_TOKEN_PATH, "w", encoding="utf-8") as f_token_out, \
                 open(TRAINSET_RAW_PAIR_PATH, "w", encoding="utf-8") as f_raw_out:
                results_iterator = pool.imap_unordered(target_func_wrapped, tasks)
                for generated_pairs_chunk in results_iterator:
                    for pair in generated_pairs_chunk:
                        # Sauvegarde tokenisée (actuelle)
                        f_token_out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                        # Sauvegarde raw (sans token → version texte jointe)
                        f_raw_out.write(json.dumps({
                            "seq1": " ".join(pair["seq1"]),
                            "seq2": " ".join(pair["seq2"]),
                            "similarity": pair["similarity"]
                        }, ensure_ascii=False) + "\n")
                        total_pairs_written += 1
                    main_progress.update(1)
            main_progress.close()
            print("\n" * (num_processes + 1))
            print(f"   -> Saved {total_pairs_written} token-based training pairs to {TRAINSET_TOKEN_PATH}")

        # === 4.bis  Création du jeu d'évaluation =========================
        with timing("4.bis  Building 200-sample evaluation set"):
            # 1) Charger toutes les paires brutes
            with TRAINSET_RAW_PAIR_PATH.open("r", encoding="utf-8") as f:
                raw_pairs = [json.loads(l) for l in f]

            if len(raw_pairs) < 200:
                raise RuntimeError("Pas assez de paires pour échantillonner 200 exemples.")

            random.shuffle(raw_pairs)
            eval_unseen   = raw_pairs[:100]   # à retirer du train
            eval_copied   = raw_pairs[100:200]  # reste dans le train
            eval_pairs    = eval_unseen + eval_copied

            # 2) Sauvegarder le fichier d'évaluation
            with EVAL_OUT_PATH.open("w", encoding="utf-8") as f_eval:
                for p in eval_pairs:
                    json.dump(p, f_eval, ensure_ascii=False)
                    f_eval.write("\n")
            print(f"   -> Saved 200 eval pairs to {EVAL_OUT_PATH}")

            # 3) Filtrer les paires tokenisées : on enlève les 100 unseen
            unseen_set = {
                (p["seq1"], p["seq2"], p["similarity"])
                for p in eval_unseen
            }

            def keep_pair_token(line: str) -> bool:
                obj = json.loads(line)
                key = (" ".join(obj["seq1"]), " ".join(obj["seq2"]), obj["similarity"])
                return key not in unseen_set

            # -- ré-écrire embedding_pairs_token.jsonl filtré
            token_lines_kept = []
            with TRAINSET_TOKEN_PATH.open("r", encoding="utf-8") as fin:
                for l in fin:
                    if keep_pair_token(l):
                        token_lines_kept.append(l)
            with TRAINSET_TOKEN_PATH.open("w", encoding="utf-8") as fout:
                fout.writelines(token_lines_kept)

            total_pairs_written = len(token_lines_kept)
            print(f"   -> Train set now contains {total_pairs_written} pairs "
                  f"(100 removed for eval)")

            # -- ré-écrire embedding_pairs_raw.jsonl filtré de la même façon
            with TRAINSET_RAW_PAIR_PATH.open("w", encoding="utf-8") as fout:
                for p in raw_pairs[100:]:              # on garde tout sauf les 100 premières
                    json.dump(p, fout, ensure_ascii=False)
                    fout.write("\n")

            EVAL_JSON_PATH = EVAL_OUT_PATH.with_suffix(".json")
            with EVAL_JSON_PATH.open("w", encoding="utf-8") as f_json:
                json.dump(eval_pairs, f_json, ensure_ascii=False, indent=2)

            print(f"   -> Also saved pretty JSON to {EVAL_JSON_PATH}")

            
        with timing("5. Indexing final training file"):
            total_indexed_written = 0
            with TRAINSET_TOKEN_PATH.open("r", encoding="utf-8") as f_in, \
                 TRAINSET_INDEX_PATH.open("w", encoding="utf-8") as f_out:
                for line in tqdm(f_in, desc="   Indexing pairs", total=total_pairs_written):
                    item = json.loads(line)
                    x1, x2 = tokens_to_indices(item["seq1"], vocab), tokens_to_indices(item["seq2"], vocab)
                    if not x1 or not x2: continue
                    indexed_item = {"seq1": x1, "seq2": x2, "similarity": item["similarity"]}
                    f_out.write(json.dumps(indexed_item, ensure_ascii=False) + "\n")
                    total_indexed_written += 1
            print(f"   -> Saved {total_indexed_written} indexed training pairs to {TRAINSET_INDEX_PATH}")
        
        with timing("6. Calculating final distribution statistics"):
            dist = report_distribution(
                TRAINSET_INDEX_PATH,
                raw_oscar_count=RAW_SAMPLE_LIMIT,
                filtered_count=len(raw_sentences),
                vocab_size=len(vocab),
                pair_count=total_indexed_written
            )
            with open(STATS_PATH, 'w', encoding='utf-8') as f:
                json.dump(dist, f, ensure_ascii=False, indent=2)
            print(f"   -> Saved statistics to {STATS_PATH}")

        with timing("7. Converting final indexed training set from JSONL to JSON"):
            json_items = []
            with TRAINSET_INDEX_PATH.open("r", encoding="utf-8") as f:
                for line in tqdm(f, desc="   Loading .jsonl into memory"):
                    try:
                        json_items.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

            # Final destination: same dir, different extension
            with TRAINSET_JSON_PATH.open("w", encoding="utf-8") as f:
                json.dump(json_items, f, ensure_ascii=False, indent=2)

            print(f"   -> Converted {len(json_items)} entries to {TRAINSET_JSON_PATH}")

if __name__ == "__main__":
    main()