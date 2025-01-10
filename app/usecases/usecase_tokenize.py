# app\usecases\usecase_tokenize.py
import re
import json
import spacy
from typing import List
from app.apis.models.tokenize_model_data import ModelTokenizeData

# Load the French language model
nlp = spacy.load("fr_core_news_md")
stopwords = nlp.Defaults.stop_words

def load_regex_patterns(filepath: str):
    """
    Load regex patterns from a JSON file.
    :param filepath: Path to the regex patterns JSON file.
    :return: List of regex patterns.
    """
    with open(filepath, 'r') as file:
        return json.load(file)
    
def anonymize_names(text: str) -> str:
    """
    Anonymize person names in the text using spaCy's named entity recognition.
    :param text: Input text to anonymize.
    :return: Text with names replaced by [no_process][name][/no_process].
    """
    doc = nlp(text)
    anonymized_text = text

    # Replace detected person names with [no_process][name][/no_process]
    for ent in doc.ents:
        if ent.label_ == "PER":
            anonymized_text = anonymized_text.replace(ent.text, "[no_process][name][/no_process]")

    return anonymized_text

def apply_regex_patterns(text: str, patterns: List[dict]) -> str:
    """
    Apply a series of regex operations (replace or delete) on the text.
    :param text: Input text to process.
    :param patterns: List of dictionaries with 'regex', 'op', and 'str' for replacement.
    :return: Processed text after regex application.
    """
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
    """
    Process the description by choosing the most relevant part.
    - If "---" is present, splits and selects either before or after.
    :param description: Input description to process.
    :return: Processed description in lowercase.
    """
    if '---' in description:
        segments = description.split('---', 1)
        before = segments[0].strip().lower()
        after = segments[1].strip().lower()
        
        # Rule: If "before" is empty, use "after"; otherwise, use "before"
        description_processed = before if before else after
    else:
        description_processed = description.strip().lower()
    
    return description_processed

def remove_stopwords(text: str) -> str:
    """
    Remove stopwords from the text.
    :param text: Input text to process.
    :return: Text without stopwords.
    """
    words = text.split()
    filtered_text = " ".join([word for word in words if word.lower() not in stopwords])
    return filtered_text

def remove_protected_tags(tokens: List[str]) -> List[str]:
    """
    Remove protected tags from tokens while keeping their content intact.
    :param tokens: List of tokens with potential protected tags.
    :return: List of tokens without protected tags.
    """
    cleaned_tokens = []
    for token in tokens:
        cleaned_token = re.sub(r"\[no_process]|\[/no_process]|\[/no_proces]", "", token)
        cleaned_token = re.sub(r"no_process]|\[/no_process|\[/no_proces", "", cleaned_token)
        cleaned_tokens.append(cleaned_token)
    return cleaned_tokens

def remove_unwanted(tokens: List[str]) -> List[str]:
    """
    Remove specific unwanted tokens from the list.
    :param tokens: List of tokens to filter.
    :return: Filtered list of tokens.
    """
    unwanted_tokens = [
        # Existing tokens
        "m'", "m'", "s'", "-t", "qu", "-ce", "j'", "l'", "n'", "qu'", "jusqu'", "c'",
        # Pronouns
        "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
        # Other words
        "se", "ne", "me", "le", "en", "ci", "ce", "cela",
        # Adverbs
        "trop", "travers", "en train", "tout", "temps", "tandis", 
        "rapidement", "jamais", "hier", "haut", "visiblement"
    ]
    return [token for token in tokens if token.lower() not in unwanted_tokens]

def remove_polite(text: str) -> str:
    """
    Remove polite phrases from the text based on a predefined list.
    :param text: Input text to process.
    :return: Text without polite phrases.
    """
    polite_phrases = [
        "bonjour", "merci", "cordialement", "désolé", "dérrangement",
        "salutations", "remerciement", "salut", "aide", "svp", "hello",
        "respectueusement", "bien", "avance", "bon", "journée", "remercier"
    ]

    # Regex pattern to detect all polite words
    polite_pattern = r"\b(?:{})\b".format("|".join(re.escape(word) for word in polite_phrases))
    
    # Replace polite words with an empty string
    cleaned_text = re.sub(polite_pattern, "", text, flags=re.IGNORECASE)
    
    # Remove any extra spaces left by deletions
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    
    return cleaned_text

def extract_corrected_tokens(doc):
    """
    Extract tokens with corrected lemmatization from a spaCy document.
    :param doc: spaCy document.
    :return: List of corrected lemmatized tokens.
    """
    lemma_corrections = {
        "bloqu": "bloquer",
        "essai": "essayer",  # Add other common corrections here
    }

    tokens = []

    # Track entity positions to avoid duplicates
    ent_positions = {ent.start for ent in doc.ents}

    for i, token in enumerate(doc):
        # Check if token is part of an entity
        if i in ent_positions:
            for ent in doc.ents:
                if ent.start == i:
                    tokens.append(f"[{ent.text}]")
        elif token.pos_ not in {"DET", "PUNCT", "SPACE", "SYM", "ADP", "X", "NUM", "ADV"} and len(token.lemma_) > 1:
            lemma = token.lemma_
            corrected_lemma = lemma_corrections.get(lemma, lemma)
            tokens.append(corrected_lemma)
    return tokens

def expand_abbreviations(text: str) -> str:
    """
    Expand common abbreviations in the text.
    :param text: Input text to process.
    :return: Text with abbreviations expanded.
    """
    abbreviations = {
        "mr": "monsieur",
        "mme": "madame",
        "dr": "docteur",
        "st": "saint",
        "bcp": "beaucoup",
        "pr": "pour",
        "avt": "avant",
        "dpt": "département",
        "nb": "nombre",
        "info": "information",
        "tps": "temps",
        "pb": "probleme",
        "pbm": "probleme",
        "mdp": "mot passe"
    }

    # Regex pattern to match abbreviations
    abbreviation_pattern = re.compile(r"\b(" + "|".join(re.escape(abbr) for abbr in abbreviations.keys()) + r")\b", re.IGNORECASE)

    # Replace abbreviations with their expansions
    expanded_text = abbreviation_pattern.sub(lambda match: abbreviations[match.group(0).lower()], text)

    return expanded_text

def normalize_special_characters(text: str) -> str:
    """
    Normalize or remove special characters in the text.
    :param text: Input text to process.
    :return: Text with special characters normalized.
    """
    # Define the mapping of characters to replace
    special_character_replacements = {
        "’": "'",  # Replace typographic apostrophe with standard apostrophe
        "“": '"',  # Replace left double quote
        "”": '"',  # Replace right double quote
        "—": "-",  # Replace em dash with hyphen
        "–": "-",  # Replace en dash with hyphen
        "…": "...",  # Replace ellipsis with three dots
        "«": '"',  # Replace left guillemet
        "»": '"',  # Replace right guillemet
        " ": " ",  # Replace non-breaking space with regular space
        "\t": " ",  # Replace tab with space
    }

    # Apply replacements
    for special_char, replacement in special_character_replacements.items():
        text = text.replace(special_char, replacement)

    # Optionally, clean up extra spaces caused by replacements
    text = re.sub(r"\s+", " ", text).strip()

    return text

def usecase_tokenize(data: List[ModelTokenizeData], regex_filepath: str = 'tokenize_regex.json'):
    """
    Main function to tokenize and preprocess descriptions using regex, stopword removal, and anonymization.
    :param data: List of ModelTokenizeData objects containing descriptions.
    :param regex_filepath: Path to the JSON file with regex patterns.
    :return: List of processed tokens for each description.
    """
    result = []
    regex_patterns = load_regex_patterns(regex_filepath)

    for item in data:
        # Extract and process the main message from description
        before_spacy = process_description(item.description)
        initial_word_count = len(before_spacy.split())

        # Expand abbreviations
        before_spacy = normalize_special_characters(before_spacy)

        # Expand abbreviations
        before_spacy = expand_abbreviations(before_spacy)

        # Remove polite phrases
        before_spacy = remove_polite(before_spacy)

        # Apply regex patterns
        before_spacy = apply_regex_patterns(before_spacy, regex_patterns)

        # Anonymize names
        before_spacy = anonymize_names(before_spacy)

        # Remove stopwords before tokenization
        before_spacy = remove_stopwords(before_spacy)

        # Process text with spaCy
        doc = nlp(before_spacy)

        # Extract tokens with corrected lemmatization
        after_spacy = extract_corrected_tokens(doc)

        # Remove protected tags from tokens
        final = remove_protected_tags(after_spacy)

        # Remove unwanted tokens
        final = remove_unwanted(final)

        # Calculate compression rate
        final_word_count = len(final)
        compression_rate = 100 * (initial_word_count - final_word_count) / initial_word_count if initial_word_count > 0 else 0

        result.append({
            'id': item.incidentId, 
            'source': item.description,
            'before_spacy': before_spacy,
            'after_spacy': after_spacy,
            'tokens': final,
            'compression_rate': compression_rate
        })

    return result
