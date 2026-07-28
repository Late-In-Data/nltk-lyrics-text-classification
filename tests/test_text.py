from lyrics_classification.text import load_custom_stopwords, preprocessor, tokenize


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Salut, le Monde ! Ça va ?") == ["salut", "le", "monde", "ça", "va"]


def test_tokenize_drops_empty_tokens():
    assert tokenize("mot1 - mot2") == ["mot1", "mot2"]


def test_preprocessor_lowercases_and_collapses_whitespace():
    assert preprocessor("  Bonjour   le\nMonde  ") == "bonjour le monde"


def test_preprocessor_handles_none():
    assert preprocessor(None) == ""


def test_load_custom_stopwords_splits_on_commas_not_whitespace(tmp_path):
    stopword_file = tmp_path / "stopword.txt"
    stopword_file.write_text("Un,Deux,Trois", encoding="utf-8")

    assert load_custom_stopwords(stopword_file) == {"un", "deux", "trois"}
