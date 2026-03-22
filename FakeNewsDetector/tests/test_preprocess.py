"""
tests/test_preprocess.py
------------------------
Unit tests for the preprocessing pipeline.
Run: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from preprocess import clean_text, clean_dataframe, build_tfidf
import pandas as pd


# ─── clean_text ───────────────────────────────────────────────────────────────

class TestCleanText:
    def test_lowercase(self):
        assert clean_text("HELLO WORLD") == clean_text("hello world")

    def test_removes_urls(self):
        result = clean_text("Check out https://example.com for more info")
        assert 'https' not in result
        assert 'example' not in result

    def test_removes_html_tags(self):
        result = clean_text("<p>Hello <b>world</b></p>")
        assert '<' not in result
        assert '>' not in result
        assert 'hello' in result

    def test_removes_punctuation(self):
        result = clean_text("Breaking!!! Shocking news???")
        assert '!' not in result
        assert '?' not in result

    def test_handles_empty_string(self):
        assert clean_text("") == ""

    def test_handles_none(self):
        assert clean_text(None) == ""

    def test_handles_non_string(self):
        assert clean_text(42) == ""

    def test_removes_social_mentions(self):
        result = clean_text("According to @johndoe and #fakenews tag")
        assert '@johndoe' not in result
        assert '#fakenews' not in result

    def test_returns_string(self):
        assert isinstance(clean_text("any text here"), str)

    def test_real_article_cleans_sensibly(self):
        article = "Scientists at NASA have discovered a new planet in our solar system."
        result = clean_text(article)
        assert len(result) > 5  # should still have meaningful content
        assert isinstance(result, str)


# ─── clean_dataframe ─────────────────────────────────────────────────────────

class TestCleanDataframe:
    def setup_method(self):
        self.df = pd.DataFrame({
            'text': [
                "Real news article about science.",
                "FAKE NEWS!!! You won't believe this!!!",
                "",          # empty — should be dropped
                None,        # None — should be dropped
                "Another valid article here with enough words.",
            ],
            'label': ['real', 'fake', 'fake', 'real', 'real']
        })

    def test_drops_empty_rows(self):
        result = clean_dataframe(self.df)
        assert len(result) < len(self.df)

    def test_label_normalization_to_binary(self):
        result = clean_dataframe(self.df)
        assert set(result['label'].unique()).issubset({0, 1})

    def test_text_column_exists(self):
        result = clean_dataframe(self.df)
        assert 'text' in result.columns

    def test_label_column_exists(self):
        result = clean_dataframe(self.df)
        assert 'label' in result.columns


# ─── build_tfidf ─────────────────────────────────────────────────────────────

class TestBuildTfidf:
    def setup_method(self):
        self.texts = [
            "This is a fake news article about politicians.",
            "Scientists discover water on Mars in new study.",
            "Breaking news shocking revelation about government.",
            "Researchers publish findings in Nature journal.",
            "Viral claim about vaccines debunked by experts.",
        ]

    def test_returns_vectorizer_and_matrix(self):
        vec, X = build_tfidf(self.texts)
        assert vec is not None
        assert X is not None

    def test_matrix_shape(self):
        vec, X = build_tfidf(self.texts)
        assert X.shape[0] == len(self.texts)

    def test_matrix_non_empty(self):
        vec, X = build_tfidf(self.texts)
        assert X.nnz > 0  # has non-zero entries

    def test_transform_new_text(self):
        vec, X = build_tfidf(self.texts)
        new_text = ["New article about science and research."]
        X_new = vec.transform(new_text)
        assert X_new.shape[0] == 1
        assert X_new.shape[1] == X.shape[1]

    def test_max_features_respected(self):
        vec, X = build_tfidf(self.texts, max_features=100)
        assert X.shape[1] <= 100


# ─── Integration: clean → vectorize ──────────────────────────────────────────

class TestIntegration:
    def test_clean_then_vectorize(self):
        texts = [
            "Breaking!!! SHOCKING news about the government cover-up https://fake.tk",
            "Scientists at WHO confirm new treatment study results.",
            "You won't believe what happened next in this viral story!!!",
            "Research published in Nature shows climate change acceleration.",
        ]
        labels = [1, 0, 1, 0]

        df = pd.DataFrame({'text': texts, 'label': labels})
        df_clean = clean_dataframe(df)

        assert len(df_clean) > 0
        vec, X = build_tfidf(df_clean['text'].tolist())
        assert X.shape[0] == len(df_clean)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
