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
    unwanted_tokens = ["m’", "s’", "-t", "qu", "-ce", "j’", "l’", "n’"]
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
        "respectueusement", "bien", "avance", "bon", "journée"
    ]

    # Regex pattern to detect all polite words
    polite_pattern = r"\b(?:{})\b".format("|".join(re.escape(word) for word in polite_phrases))
    
    # Replace polite words with an empty string
    cleaned_text = re.sub(polite_pattern, "", text, flags=re.IGNORECASE)
    
    # Remove any extra spaces left by deletions
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    
    return cleaned_text

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
        description_processed = process_description(item.description)

        # Remove polite phrases
        description_processed = remove_polite(description_processed)

        # Apply regex patterns
        description_processed = apply_regex_patterns(description_processed, regex_patterns)

        # Anonymize names
        description_processed = anonymize_names(description_processed)

        # Remove stopwords before tokenization
        description_processed = remove_stopwords(description_processed)

        # Process text with spaCy
        doc = nlp(description_processed)

        # Extract and transform tokens and entities
        filtered_tokens = []

        # Track entity positions to avoid duplicates
        ent_positions = {ent.start for ent in doc.ents}

        for i, token in enumerate(doc):
            # Check if token is part of an entity
            if i in ent_positions:
                # Add the full entity with its label if relevant
                for ent in doc.ents:
                    if ent.start == i:
                        filtered_tokens.append(f"[{ent.text}]")
            elif token.pos_ not in {"DET", "PUNCT", "SPACE", "SYM", "ADP", "X", "NUM"} and len(token.lemma_) > 1:
                # Add the lemma for non-entity tokens and exclusions
                filtered_tokens.append(token.lemma_)

        # Remove protected tags from tokens
        filtered_tokens_final = remove_protected_tags(filtered_tokens)

        # Remove unwanted tokens
        filtered_tokens_final = remove_unwanted(filtered_tokens_final)

        result.append({
            # 'id': item.incidentId, 
            # 'source': item.description,
            # 'source_processed': description_processed,
            # 'filtered_tokens': filtered_tokens,
            'tokens': filtered_tokens_final
        })

    return result
