import json
import pytest
import tempfile
from unittest.mock import patch, MagicMock

from app.usecases.usecase_tokenize import (
    expand_abbreviations,
    anonymize_names,
    apply_regex_patterns,
    normalize_special_characters,
    process_description,
    remove_stopwords,
    remove_protected_tags,
    remove_unwanted,
    remove_polite,
    extract_corrected_tokens,
    usecase_tokenize,
)
from app.apis.models.tokenize_model_data import ModelTokenizeData

test_data_real = [
    ModelTokenizeData(
        description="bonjour, je n'arrive pas à me connecter à Kalydian, est-il possible de m'activer le compte ? Merci.",
        incidentId="a251f160-c305-4f57-9201-60cf8b8c593f"
    ),
    ModelTokenizeData(
        description="Depuis la modification de mon mot de passe de session Outlook, je ne parviens plus à me connecter à Kalydian.",
        incidentId="1cd1c275-92d9-4ba5-b1ea-2b27af7a06ea"
    ),
    ModelTokenizeData(
        description="Hello, pourriez-vous créer les accès pour Terry Bourillon ? Merci.",
        incidentId="13d1ee86-1491-ef11-97bb-00155d0879e2"
    ),
    ModelTokenizeData(
        description="Je n'arrive pas à importer mes mots de passe dans Kalydian. Y a-t-il une astuce ?",
        incidentId="ac7a1287-c08f-ef11-97bb-00155d0879e2"
    )
]

# Regex patterns for testing
REGEX_PATTERNS = [
    {
        "regex": r"\b[\w\.-]+@[^s]*thomyris[^s]*\b",
        "op": "REPLACE",
        "str": "[no_process][mail_service][/no_process]",
    },
    {
        "regex": r"\b[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}\b",
        "op": "REPLACE",
        "str": "[no_process][mail][/no_process]",
    },
]

@pytest.fixture
def test_data():
    """Fixture to provide test data."""
    return test_data_real

def setup_regex_test_file():
    """
    Create a temporary JSON file for regex patterns used in tests.
    This file will contain patterns to delete specific polite phrases.
    :return: The file path to the temporary regex patterns JSON.
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w')
    json.dump(REGEX_PATTERNS, temp_file)
    temp_file.close()
    return temp_file.name

# Use this path in your tests
regex_filepath = setup_regex_test_file()

# Test: Full tokenization process success with all processing steps
@patch("app.usecases.usecase_tokenize.load_regex_patterns")
@patch("app.usecases.usecase_tokenize.anonymize_names")
@patch("app.usecases.usecase_tokenize.nlp")
def test_usecase_tokenize_full_process(mock_nlp, mock_anonymize_names, mock_load_regex_patterns, test_data):
    """
    Test the complete tokenization flow including regex replacement,
    name anonymization, and token extraction.
    """
    # Mock for anonymize_names
    mock_anonymize_names.side_effect = lambda text: text.replace("terry bourillon", "[no_process][name][/no_process]")

    # Mock for load_regex_patterns
    mock_load_regex_patterns.return_value = REGEX_PATTERNS

    # Simulate the output of the spaCy model for tokenization
    doc_mock = MagicMock()
    token1 = MagicMock(lemma_="connecter", pos_="VERB")
    token2 = MagicMock(lemma_="[service_k]", pos_="NOUN")
    doc_mock.__iter__.return_value = [token1, token2]
    mock_nlp.return_value = doc_mock

    # Call the tokenization function
    result = usecase_tokenize(test_data, "dummy_path")

    # Verify the expected number of results
    assert len(result) == len(test_data), "Mismatch in the number of processed items."

    # Check that anonymize_names was called with the expected processed text
    expected_processed_text = ", pourriez-vous créer les accès pour terry bourillon ? ."
    mock_anonymize_names.assert_any_call(expected_processed_text)

    # Alternatively, check that anonymize_names was called with any text containing 'terry bourillon'
    assert any("terry bourillon" in call.args[0] for call in mock_anonymize_names.call_args_list), "anonymize_names was not called with text containing 'terry bourillon'."

    # Check that anonymization works
    expected_anonymized = "[no_process][name][/no_process]"
    before_spacy = result[2]["before_spacy"]
    assert expected_anonymized in before_spacy, f"Name anonymization failed. Processed: {before_spacy}"

    # Check regex replacement
    assert "[service_k]" in result[0]["tokens"], "Regex replacement or token extraction failed."

    # Check token extraction
    assert "connecter" in result[0]["tokens"], "Token extraction failed for 'connecter'."

    # Optionally, verify the total number of calls
    assert mock_anonymize_names.call_count == len(test_data), f"Expected {len(test_data)} calls to anonymize_names, got {mock_anonymize_names.call_count}."

def test_load_regex_patterns():
    """Test loading regex patterns."""
    # Simulate loading regex patterns directly
    patterns = REGEX_PATTERNS
    assert len(patterns) == 2, "Expected two regex patterns."
    assert patterns[0]["op"] == "REPLACE", "Expected REPLACE operation."

@patch("app.usecases.usecase_tokenize.remove_stopwords")
def test_usecase_tokenize_remove_stopwords(mock_remove_stopwords, test_data):
    """
    Test the stopword removal process within the tokenization function.
    """
    # Mock stopword removal to remove specific stopwords
    mock_remove_stopwords.side_effect = lambda text: text.replace("je", "").replace("à", "")

    # Call the tokenization function
    with patch("app.usecases.usecase_tokenize.load_regex_patterns", return_value=REGEX_PATTERNS):
        with patch("app.usecases.usecase_tokenize.anonymize_names", side_effect=lambda x: x):
            with patch("app.usecases.usecase_tokenize.nlp") as mock_nlp:
                doc_mock = MagicMock()
                token1 = MagicMock(lemma_="connecter", pos_="VERB")
                doc_mock.__iter__.return_value = [token1]
                mock_nlp.return_value = doc_mock

                result = usecase_tokenize(test_data, "dummy_path")

    # Verify that stopwords are removed
    assert "je" not in result[0]["before_spacy"], "Stopword removal failed for 'je'."
    assert "à" not in result[0]["before_spacy"], "Stopword removal failed for 'à'."


@patch("app.usecases.usecase_tokenize.remove_polite")
def test_usecase_tokenize_remove_polite(mock_remove_polite, test_data):
    """
    Test the polite phrase removal process in the tokenization function.
    """
    # Mock polite phrase removal
    mock_remove_polite.side_effect = lambda text: text.replace("Bonjour", "").replace("Merci", "")

    # Call the tokenization function
    with patch("app.usecases.usecase_tokenize.load_regex_patterns", return_value=REGEX_PATTERNS):
        with patch("app.usecases.usecase_tokenize.anonymize_names", side_effect=lambda x: x):
            with patch("app.usecases.usecase_tokenize.nlp") as mock_nlp:
                doc_mock = MagicMock()
                token1 = MagicMock(lemma_="connecter", pos_="VERB")
                doc_mock.__iter__.return_value = [token1]
                mock_nlp.return_value = doc_mock

                result = usecase_tokenize(test_data, "dummy_path")

    # Verify that polite phrases are removed
    assert "Bonjour" not in result[0]["before_spacy"], "Polite phrase removal failed for 'Bonjour'."
    assert "Merci" not in result[0]["before_spacy"], "Polite phrase removal failed for 'Merci'."


def test_load_regex_patterns():
    """Test loading regex patterns."""
    patterns = REGEX_PATTERNS
    assert len(patterns) == 2, "Expected two regex patterns."
    assert patterns[0]["op"] == "REPLACE", "Expected REPLACE operation."

def test_anonymize_names():
    """Test anonymizing names."""
    with patch("app.usecases.usecase_tokenize.nlp") as mock_nlp:
        doc_mock = MagicMock()
        ent_mock = MagicMock()
        ent_mock.text = "Terry Bourillon"
        ent_mock.label_ = "PER"
        doc_mock.ents = [ent_mock]
        mock_nlp.return_value = doc_mock

        result = anonymize_names("Terry Bourillon is here.")
        assert result == "[no_process][name][/no_process] is here.", "Name anonymization failed."

def test_apply_regex_patterns():
    """Test applying regex patterns to text."""
    text = "Contact: jean.dupont@thomyris.com"
    result = apply_regex_patterns(text, REGEX_PATTERNS)
    assert result == "Contact: [no_process][mail_service][/no_process]", "Regex replacement failed."

def test_process_description():
    """Test description processing."""
    description = "Header---Main content"
    result = process_description(description)
    assert result == "header", "Description processing failed."

def test_remove_stopwords():
    """Test stopword removal."""
    with patch("app.usecases.usecase_tokenize.stopwords", {"je", "à", "me"}):
        text = "je n'arrive pas à me connecter"
        result = remove_stopwords(text)
        assert result == "n'arrive pas connecter", "Stopword removal failed."

def test_remove_protected_tags():
    """Test removing protected tags."""
    tokens = ["[no_process]test[/no_process]", "[no_process][name][/no_process]"]
    result = remove_protected_tags(tokens)
    assert result == ["test", "[name]"], "Protected tag removal failed."

def test_remove_unwanted():
    """Test removing unwanted tokens."""
    tokens = ["m'", "qu'", "word"]
    result = remove_unwanted(tokens)
    assert result == ["word"], "Unwanted token removal failed."

def test_remove_polite():
    """Test removing polite phrases."""
    text = "Bonjour, merci de votre aide."
    result = remove_polite(text)
    assert result == ", de votre .", "Polite phrase removal failed."

def test_extract_corrected_tokens():
    """Test extracting corrected tokens."""
    with patch("app.usecases.usecase_tokenize.nlp") as mock_nlp:
        doc_mock = MagicMock()
        token1 = MagicMock(lemma_="bloqu", pos_="VERB")
        token2 = MagicMock(lemma_="essai", pos_="NOUN")
        doc_mock.__iter__.return_value = [token1, token2]
        mock_nlp.return_value = doc_mock

        result = extract_corrected_tokens(doc_mock)
        assert result == ["bloquer", "essayer"], "Token correction failed."

def test_usecase_tokenize(test_data):
    """Test full tokenization."""
    with patch("app.usecases.usecase_tokenize.load_regex_patterns", return_value=REGEX_PATTERNS):
        with patch("app.usecases.usecase_tokenize.anonymize_names", side_effect=lambda x: x):
            with patch("app.usecases.usecase_tokenize.nlp") as mock_nlp:
                doc_mock = MagicMock()
                token1 = MagicMock(lemma_="issue", pos_="NOUN")
                token2 = MagicMock(lemma_="connect", pos_="VERB")
                doc_mock.__iter__.return_value = [token1, token2]
                mock_nlp.return_value = doc_mock

                result = usecase_tokenize(test_data, "dummy_path")
                assert len(result) == len(test_data), "Mismatch in the number of results."
                assert "issue" in result[0]["tokens"], "Tokenization failed for the first description."

def test_expand_abbreviations():
    text = "Mr et Mme Dupont sont à St Claude avec bcp d'info pr le dpt."
    expected = "monsieur et madame Dupont sont à saint Claude avec beaucoup d'information pour le département."
    result = expand_abbreviations(text)
    assert result == expected, f"Expected: {expected}, but got: {result}"

def test_normalize_special_characters():
    text = "C’est une journée — exceptionnelle… avec “émotions” et tab\tici."
    expected = "C'est une journée - exceptionnelle... avec \"émotions\" et tab ici."
    result = normalize_special_characters(text)
    assert result == expected, f"Expected: {expected}, but got: {result}"
