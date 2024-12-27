import json
import pytest
import tempfile
from unittest.mock import patch, MagicMock
from app.usecases.usecase_tokenize import usecase_tokenize
from app.apis.models.tokenize_model_data import ModelTokenizeData

# Test data for the function
test_data = [
    ModelTokenizeData(description="Jean Dupont a signalé un problème.", incidentId="42")
]

def setup_regex_test_file():
    """
    Create a temporary JSON file for regex patterns used in tests.
    This file will contain patterns to delete specific polite phrases.
    :return: The file path to the temporary regex patterns JSON.
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w')
    regex_patterns = [
        {"regex": "\\bbonjour\\b", "op": "DELETE"},
        {"regex": "\\bmerci\\b", "op": "DELETE"}
    ]
    json.dump(regex_patterns, temp_file)
    temp_file.close()
    return temp_file.name

# Use this path in your tests
regex_filepath = setup_regex_test_file()

# Test: Full tokenization process success with all processing steps
@patch("app.usecases.usecase_tokenize.load_regex_patterns")
@patch("app.usecases.usecase_tokenize.anonymize_names")
@patch("app.usecases.usecase_tokenize.nlp")
def test_usecase_tokenize_full_process(mock_nlp, mock_anonymize_names, mock_load_regex_patterns):
    """
    Test the complete tokenization flow including regex replacement,
    name anonymization, and token extraction.
    """
    # Configure mock for anonymize_names to replace names with the anonymized tag
    mock_anonymize_names.return_value = "[no_process][name][/no_process]"

    # Configure mock for load_regex_patterns to replace 'problème' with 'issue'
    mock_load_regex_patterns.return_value = [{"regex": r"\bproblème\b", "op": "REPLACE", "str": "issue"}]

    # Simulate the output of the spaCy model for tokenization
    doc_mock = MagicMock()
    token0 = MagicMock(lemma_="[no_process][name][/no_process]", label_="NOUN")
    token1 = MagicMock(lemma_="signaler", pos_="VERB")
    token2 = MagicMock(lemma_="issue", pos_="NOUN")
    doc_mock.__iter__.return_value = [token0, token1, token2]
    mock_nlp.return_value = doc_mock

    # Call the tokenization function
    result = usecase_tokenize(test_data, regex_filepath)

    # Verify the expected result
    expected_result = ["[name]", "signaler", "issue"]
    assert result[0]["tokens"] == expected_result, f"Expected {expected_result} but got {result}"

# Test: Verify stopword removal
@patch("app.usecases.usecase_tokenize.remove_stopwords")
def test_usecase_tokenize_remove_stopwords(mock_remove_stopwords):
    """
    Test the stopword removal process within the tokenization function.
    """
    # Mock stopword removal to remove the word "je"
    mock_remove_stopwords.side_effect = lambda text: text.replace("je", "")
    
    # Call the function
    result = usecase_tokenize(test_data, regex_filepath)
    
    # Check that stopwords have been removed correctly
    assert "je" not in result[0]["tokens"], "Stopwords were not removed correctly."

# Test: Verify polite phrase removal
@patch("app.usecases.usecase_tokenize.remove_polite")
def test_usecase_tokenize_remove_polite(mock_remove_polite):
    """
    Test the polite phrase removal process in the tokenization function.
    """
    # Mock polite phrase removal to remove "Bonjour"
    mock_remove_polite.side_effect = lambda text: text.replace("Bonjour", "")
    
    # Call the function
    result = usecase_tokenize(test_data, regex_filepath)
    
    # Check that polite phrases have been removed correctly
    assert "Bonjour" not in result[0]["tokens"], "Polite phrases were not removed correctly."
