import json
import pandas as pd
import argparse

def load_data(file_path: str) -> pd.DataFrame:
    """Charge les données depuis le fichier JSON et crée un DataFrame.
       On suppose que chaque enregistrement est de la forme :
       [input_tokens, target_tokens, score]
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data, columns=["input", "target", "score"])
        return df
    except Exception as e:
        raise Exception(f"Erreur lors du chargement du fichier: {e}")

def assign_bin(score: float) -> str:
    """Attribue un bin à un score en fonction des intervalles définis."""
    if score == 0.0:
        return "0.0-0.0"
    elif score == 1.0:
        return "1.0-1.0"
    elif score <= 0.1:
        return "0.0-0.1"
    elif score <= 0.2:
        return "0.1-0.2"
    elif score <= 0.3:
        return "0.2-0.3"
    elif score <= 0.4:
        return "0.3-0.4"
    elif score <= 0.5:
        return "0.4-0.5"
    elif score <= 0.6:
        return "0.5-0.6"
    elif score <= 0.7:
        return "0.6-0.7"
    elif score <= 0.8:
        return "0.7-0.8"
    elif score <= 0.9:
        return "0.8-0.9"
    elif score < 1.0:
        return "0.9-1.0"
    else:
        return "unknown"

def extract_balanced_data(file_path: str) -> list:
    """Extrait les données équilibrées à partir du fichier en réalisant un rééchantillonnage par bin.
       Le format de sortie est une liste de listes, identique au format d'entrée.
    """
    df = load_data(file_path)
    df["bin"] = df["score"].apply(assign_bin)
    
    # Comptage des exemples par bin
    counts = df["bin"].value_counts()
    non_empty_bins = counts[counts > 0]
    if non_empty_bins.empty:
        raise Exception("Aucun exemple trouvé dans les bins.")
    
    # On choisit le nombre minimum d'exemples présent dans les bins non vides
    min_count = non_empty_bins.min()
    
    balanced_samples = []
    # Pour chaque bin, on prélève aléatoirement min_count exemples
    for bin_label, group in df.groupby("bin"):
        if len(group) >= min_count:
            sampled_group = group.sample(n=min_count, random_state=42)
        else:
            # Oversampling (avec replacement) si nécessaire
            sampled_group = group.sample(n=min_count, replace=True, random_state=42)
        balanced_samples.append(sampled_group)
    
    balanced_df = pd.concat(balanced_samples).reset_index(drop=True)
    
    # Affichage de la nouvelle distribution (pour vérification)
    new_distribution = balanced_df["bin"].value_counts(normalize=True) * 100
    print("Nouvelle distribution par bin (en %):")
    print(new_distribution.to_dict())
    
    # On supprime la colonne 'bin' et on reconvertit le DataFrame en liste de listes
    balanced_df = balanced_df[["input", "target", "score"]]
    output_list = balanced_df.values.tolist()
    return output_list

def main():
    parser = argparse.ArgumentParser(description="Script de rééchantillonnage équilibré des données")
    parser.add_argument("-i", "--input", type=str, default="o3_2502151700.json",
                        help="Chemin vers le fichier JSON d'entrée")
    parser.add_argument("-o", "--output", type=str, default="balanced_data.json",
                        help="Chemin vers le fichier JSON de sortie")
    args = parser.parse_args()
    
    try:
        balanced_data = extract_balanced_data(args.input)
    except Exception as e:
        print(f"Erreur: {e}")
        return
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(balanced_data, f, ensure_ascii=False, indent=2)
    print(f"Les données équilibrées ont été enregistrées dans {args.output}")

if __name__ == "__main__":
    main()
