import pytest
from unittest.mock import patch, MagicMock
from app.usecases.usecase_tokenize import usecase_tokenize
from app.apis.models.model_tokenize_data import ModelTokenizeData

# Données de test pour la fonction
test_data = [
    ModelTokenizeData(description="Bonjour, je m'appelle Jean Dupont et j'aimerais signaler un problème.")
]

# Test 1 : Succès de la tokenisation avec toutes les étapes de traitement
@patch("app.usecases.usecase_tokenize.nlp")
@patch("app.usecases.usecase_tokenize.load_regex_patterns")
@patch("app.usecases.usecase_tokenize.anonymize_names", side_effect=lambda x: x.replace("Jean Dupont", "[no_process][name][/no_process]"))
def test_usecase_tokenize_full_process(mock_anonymize_names, mock_load_regex_patterns, mock_nlp):
    # Simuler le modèle spaCy et les regex
    mock_nlp.return_value = MagicMock()
    mock_load_regex_patterns.return_value = [{"regex": r"\bproblème\b", "op": "REPLACE", "str": "issue"}]

    # Simuler la sortie du modèle spacy pour la tokenisation
    doc_mock = MagicMock()
    doc_mock.ents = [MagicMock(text="Jean Dupont", label_="PER", start=0)]
    token1 = MagicMock(lemma_="signaler", pos_="VERB")
    token2 = MagicMock(lemma_="issue", pos_="NOUN")
    doc_mock.__iter__.return_value = [token1, token2]
    mock_nlp.return_value = doc_mock

    # Appeler la fonction de tokenisation
    result = usecase_tokenize(test_data, regex_filepath="fake_path.json")
    
    # Vérifier le résultat
    expected_result = [{"tokens": ["signaler", "issue"]}]
    assert result == expected_result, f"Expected {expected_result} but got {result}"

# Test 2 : Vérifier le chargement des motifs regex
@patch("app.usecases.usecase_tokenize.load_regex_patterns")
def test_usecase_tokenize_regex_loading(mock_load_regex_patterns):
    # Simuler le chargement des motifs regex
    mock_load_regex_patterns.return_value = [{"regex": r"\btest\b", "op": "REPLACE", "str": "check"}]
    
    # Appeler la fonction
    usecase_tokenize(test_data, regex_filepath="fake_path.json")
    
    # Vérifier que les motifs regex sont bien chargés
    mock_load_regex_patterns.assert_called_once_with("fake_path.json")

# Test 3 : Vérifier l'anonymisation des noms
@patch("app.usecases.usecase_tokenize.anonymize_names")
def test_usecase_tokenize_anonymization(mock_anonymize_names):
    # Configurer un retour de l'anonymisation
    mock_anonymize_names.side_effect = lambda text: text.replace("Jean Dupont", "[no_process][name][/no_process]")
    
    # Appeler la fonction
    result = usecase_tokenize(test_data, regex_filepath="fake_path.json")
    
    # Vérifier que le nom est bien anonymisé
    assert "[no_process][name][/no_process]" in result[0]["tokens"], "Le nom n'a pas été anonymisé correctement."

# Test 4 : Vérifier la suppression des stopwords
@patch("app.usecases.usecase_tokenize.remove_stopwords")
def test_usecase_tokenize_remove_stopwords(mock_remove_stopwords):
    # Simuler la suppression des stopwords
    mock_remove_stopwords.side_effect = lambda text: text.replace("je", "")
    
    # Appeler la fonction
    result = usecase_tokenize(test_data, regex_filepath="fake_path.json")
    
    # Vérifier que les stopwords ont bien été supprimés
    assert "je" not in result[0]["tokens"], "Les stopwords n'ont pas été supprimés correctement."

# Test 5 : Vérifier la suppression des mots de politesse
@patch("app.usecases.usecase_tokenize.remove_polite")
def test_usecase_tokenize_remove_polite(mock_remove_polite):
    # Simuler la suppression des mots de politesse
    mock_remove_polite.side_effect = lambda text: text.replace("Bonjour", "")
    
    # Appeler la fonction
    result = usecase_tokenize(test_data, regex_filepath="fake_path.json")
    
    # Vérifier que les mots de politesse ont bien été supprimés
    assert "Bonjour" not in result[0]["tokens"], "Les mots de politesse n'ont pas été supprimés correctement."
