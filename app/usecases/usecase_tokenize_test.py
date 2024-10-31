import pytest
import tempfile
import json
from unittest.mock import patch, MagicMock
from app.usecases.usecase_tokenize import usecase_tokenize
from app.apis.models.model_tokenize_data import ModelTokenizeData

# Données de test pour la fonction
test_data = [
    ModelTokenizeData(description="Jean Dupont a signalé un problème.", incidentId="42")
]

def setup_regex_test_file():
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w')
    regex_patterns = [
        {"regex": "\\bbonjour\\b", "op": "DELETE"},
        {"regex": "\\bmerci\\b", "op": "DELETE"}
    ]
    json.dump(regex_patterns, temp_file)
    temp_file.close()
    return temp_file.name

# Utilisez ce chemin dans vos tests
regex_filepath = setup_regex_test_file()

# Test : Succès de la tokenisation avec toutes les étapes de traitement
@patch("app.usecases.usecase_tokenize.load_regex_patterns")
@patch("app.usecases.usecase_tokenize.anonymize_names")
@patch("app.usecases.usecase_tokenize.nlp")
def test_usecase_tokenize_full_process(mock_nlp, mock_anonymize_names, mock_load_regex_patterns):
    # Configurer le mock pour anonymize_names
    mock_anonymize_names.return_value = "[no_process][name][/no_process]"

    # Configurer le mock pour load_regex_patterns
    mock_load_regex_patterns.return_value = [{"regex": r"\bproblème\b", "op": "REPLACE", "str": "issue"}]

    # Simuler la sortie du modèle spacy pour la tokenisation
    doc_mock = MagicMock()
    token0 = MagicMock(lemma_="[no_process][name][/no_process]", label_="NOUN")
    token1 = MagicMock(lemma_="signaler", pos_="VERB")
    token2 = MagicMock(lemma_="issue", pos_="NOUN")
    doc_mock.__iter__.return_value = [token0,token1, token2]
    mock_nlp.return_value = doc_mock

    # Appeler la fonction de tokenisation
    result = usecase_tokenize(test_data, regex_filepath)

    # Vérifier le résultat attendu
    expected_result = [{"tokens": ["[name]", "signaler", "issue"]}]
    assert result == expected_result, f"Expected {expected_result} but got {result}"

# Test : Vérifier la suppression des stopwords
@patch("app.usecases.usecase_tokenize.remove_stopwords")
def test_usecase_tokenize_remove_stopwords(mock_remove_stopwords):
    # Simuler la suppression des stopwords
    mock_remove_stopwords.side_effect = lambda text: text.replace("je", "")
    
    # Appeler la fonction
    result = usecase_tokenize(test_data, regex_filepath)
    
    # Vérifier que les stopwords ont bien été supprimés
    assert "je" not in result[0]["tokens"], "Les stopwords n'ont pas été supprimés correctement."

# Test : Vérifier la suppression des mots de politesse
@patch("app.usecases.usecase_tokenize.remove_polite")
def test_usecase_tokenize_remove_polite(mock_remove_polite):
    # Simuler la suppression des mots de politesse
    mock_remove_polite.side_effect = lambda text: text.replace("Bonjour", "")
    
    # Appeler la fonction
    result = usecase_tokenize(test_data, regex_filepath)
    
    # Vérifier que les mots de politesse ont bien été supprimés
    assert "Bonjour" not in result[0]["tokens"], "Les mots de politesse n'ont pas été supprimés correctement."
