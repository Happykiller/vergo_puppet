import re
import json
import spacy
from typing import List

from app.apis.models.tokenize_model_data import ModelTokenizeData

# Charger le modèle français
nlp = spacy.load("fr_core_news_md")
stopwords = nlp.Defaults.stop_words

def load_regex_patterns(filepath: str):
    with open(filepath, 'r') as file:
        return json.load(file)
    
def anonymize_names(text: str) -> str:
    # Traiter le texte avec spaCy
    doc = nlp(text)
    anonymized_text = text

    # Remplacer les noms de personnes détectés par [NAME]
    for ent in doc.ents:
        if ent.label_ == "PER":
            anonymized_text = anonymized_text.replace(ent.text, "[no_process][name][/no_process]")

    return anonymized_text

def apply_regex_patterns(text: str, patterns: List[dict]) -> str:
    for pattern in patterns:
        regex = pattern['regex']
        operation = pattern['op']
        replace_str = pattern.get('str', '')
        if operation == 'REPLACE':
            text = re.sub(regex, replace_str, text)
        elif operation == 'DELETE':
            text = re.sub(regex, '', text)
    return text

def process_description(description: str) -> str:
    # Vérifier la présence de "---" et découper le texte
    if '---' in description:
        # Séparer avant et après "---"
        segments = description.split('---', 1)
        before = segments[0].strip().lower()
        after = segments[1].strip().lower()
        
        # Règle : Si "avant" est vide, on prend "après", sinon on prend "avant"
        description_processed = before if before else after
    else:
        # Si "---" n'est pas présent, retourner la valeur de départ en minuscules
        description_processed = description.strip().lower()
    
    return description_processed

def remove_stopwords(text: str) -> str:
    # Retirer les stopwords avant la tokenisation
    words = text.split()
    filtered_text = " ".join([word for word in words if word.lower() not in stopwords])
    return filtered_text

def remove_protected_tags(tokens: List[str]) -> List[str]:
    cleaned_tokens = []
    for token in tokens:
        # Supprime les balises de protection en gardant le contenu intact
        cleaned_token = re.sub(r"\[no_process]|\[/no_process]|\[/no_proces]", "", token)
        cleaned_token = re.sub(r"no_process]|\[/no_process|\[/no_proces", "", cleaned_token)
        cleaned_tokens.append(cleaned_token)
    return cleaned_tokens

def remove_unwanted(tokens: List[str]) -> List[str]:
    unwanted_tokens = ["m’","s’", "-t", "qu", "-ce", "j’", "l’", "n’"]
    return [token for token in tokens if token.lower() not in unwanted_tokens]

def remove_polite(text: str) -> str:
    # Liste des formules de politesse
    formules_politesse = [
        "bonjour", "merci", "cordialement", "désolé", "dérrangement"
        "salutations", "remerciement", "salut", "aide", "svp", "hello",
        "respectueusement", "bien", "avance", "bon", "journée"
    ]

    # Construire une regex pour détecter tous les mots de politesse
    polite_pattern = r"\b(?:{})\b".format("|".join(re.escape(word) for word in formules_politesse))
    
    # Remplacer tous les mots de politesse par une chaîne vide
    cleaned_text = re.sub(polite_pattern, "", text, flags=re.IGNORECASE)
    
    # Supprimer les espaces superflus laissés par les suppressions
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    
    return cleaned_text

def usecase_tokenize(data: List[ModelTokenizeData], regex_filepath: str = 'tokenize_regex.json'):
    result = []
    regex_patterns = load_regex_patterns(regex_filepath)

    for item in data:
        # Extract first message
        description_processed = process_description(item.description)

        # Remove polite
        description_processed = remove_polite(description_processed)

        # Apply regex
        description_processed = apply_regex_patterns(description_processed, regex_patterns)

        # Anonymise
        description_processed = anonymize_names(description_processed)

        # Retirer les stopwords avant la tokenisation
        description_processed = remove_stopwords(description_processed)

        # Traiter le texte
        doc = nlp(description_processed)

        # Extraire et transformer tokens et entités
        filtered_tokens = []

        # Garder la trace des positions des entités pour éviter les doublons
        ent_positions = {ent.start for ent in doc.ents}

        for i, token in enumerate(doc):
            # Vérifier si le token fait partie d'une entité
            if i in ent_positions:
                # Ajouter l'entité complète avec son label si elle est pertinente
                for ent in doc.ents:
                    if ent.start == i:
                        filtered_tokens.append(f"[{ent.text}]")
            elif token.pos_ not in {"DET", "PUNCT", "SPACE", "SYM", "ADP", "X", "NUM"} and len(token.lemma_) > 1:
                # Ajouter le lemme pour les tokens sans entité ni exclusion
                filtered_tokens.append(token.lemma_)

        filtered_tokens_final = remove_protected_tags(filtered_tokens)

        filtered_tokens_final = remove_unwanted(filtered_tokens_final)

        result.append({
            #'id': item.incidentId, 
            #'source': item.description,
            #'source_processed': description_processed,
            #'filtered_tokens': filtered_tokens,
            'tokens': filtered_tokens_final
        })

    return result
