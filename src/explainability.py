"""
explainability.py
-----------------
Makes the model interpretable:
  - LIME: per-word contribution to the prediction
  - SHAP: feature importance across the dataset
  - Sentence highlighting: mark suspicious sentences in the input

Requires: pip install lime shap
"""

import numpy as np
import re
from preprocess import clean_text

try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("Warning: lime not installed. Run: pip install lime")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: shap not installed. Run: pip install shap")


# ─── LIME Explainability ─────────────────────────────────────────────────────

class LIMEExplainer:
    """
    Generates word-level contribution scores using LIME.
    Shows WHICH words pushed the model toward FAKE or REAL.
    """

    def __init__(self, model, vectorizer, class_names=('REAL', 'FAKE')):
        self.model = model
        self.vectorizer = vectorizer
        self.class_names = list(class_names)

        if LIME_AVAILABLE:
            self.explainer = LimeTextExplainer(
                class_names=self.class_names,
                random_state=42,
            )

    def _predict_fn(self, texts: list) -> np.ndarray:
        """Prediction function for LIME (takes raw text, returns probabilities)."""
        cleaned = [clean_text(t) for t in texts]
        X = self.vectorizer.transform(cleaned)
        return self.model.predict_proba(X)

    def explain(self, text: str, n_features: int = 15,
                num_samples: int = 500) -> dict:
        """
        Generate LIME explanation for a single text.

        Returns:
            dict:
                fake_words    — list of (word, positive_contribution) for FAKE class
                real_words    — list of (word, positive_contribution) for REAL class
                word_scores   — dict of word → score (+ = fake, - = real)
                html          — HTML with highlighted text (for Flask UI)
        """
        if not LIME_AVAILABLE:
            return {'error': 'lime not installed', 'word_scores': {}}

        explanation = self.explainer.explain_instance(
            text,
            self.predict_fn,   # ← typo fix in original: was _predict_fn
            num_features=n_features,
            num_samples=num_samples,
            labels=(0, 1)
        )

        # Get contributions for FAKE class (index 1)
        fake_weights = dict(explanation.as_list(label=1))
        real_weights = dict(explanation.as_list(label=0))

        # Positive score = supports FAKE, negative = supports REAL
        word_scores = {}
        for word, weight in fake_weights.items():
            word_scores[word] = weight

        fake_words = [(w, s) for w, s in word_scores.items() if s > 0]
        real_words = [(w, abs(s)) for w, s in word_scores.items() if s < 0]

        fake_words.sort(key=lambda x: x[1], reverse=True)
        real_words.sort(key=lambda x: x[1], reverse=True)

        html = self._highlight_text(text, word_scores)

        return {
            'fake_words': fake_words[:10],
            'real_words': real_words[:5],
            'word_scores': word_scores,
            'html': html,
            'all_weights': explanation.as_list(label=1),
        }

    def predict_fn(self, texts):
        """Public alias for _predict_fn (needed for LIME)."""
        return self._predict_fn(texts)

    def _highlight_text(self, text: str, word_scores: dict) -> str:
        """
        Return HTML string with suspicious words highlighted.
        Red = likely fake indicator, Green = credibility indicator.
        """
        words = text.split()
        highlighted = []
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            score = word_scores.get(clean_word, 0)

            if score > 0.05:
                intensity = min(int(score * 300), 255)
                highlighted.append(
                    f'<mark style="background:rgba(239,68,68,{score*3:.2f});'
                    f'border-radius:3px;padding:1px 3px;" '
                    f'title="Fake indicator: {score:.3f}">{word}</mark>'
                )
            elif score < -0.05:
                highlighted.append(
                    f'<mark style="background:rgba(34,197,94,{abs(score)*3:.2f});'
                    f'border-radius:3px;padding:1px 3px;" '
                    f'title="Credibility indicator: {abs(score):.3f}">{word}</mark>'
                )
            else:
                highlighted.append(word)

        return ' '.join(highlighted)


# ─── SHAP Feature Importance ─────────────────────────────────────────────────

class SHAPExplainer:
    """
    SHAP-based feature importance for the TF-IDF + classifier pipeline.
    Shows global feature importance across the dataset.
    """

    def __init__(self, model, vectorizer, background_texts: list = None):
        self.model = model
        self.vectorizer = vectorizer
        self.explainer = None

        if SHAP_AVAILABLE and background_texts is not None:
            self._build_explainer(background_texts)

    def _build_explainer(self, background_texts: list, n_bg: int = 100):
        """Build SHAP explainer with background data."""
        bg_cleaned = [clean_text(t) for t in background_texts[:n_bg]]
        X_bg = self.vectorizer.transform(bg_cleaned)

        # Use LinearExplainer for logistic regression (fast)
        # Use KernelExplainer for tree models
        try:
            self.explainer = shap.LinearExplainer(
                self.model, X_bg, feature_perturbation='interventional'
            )
            self.explainer_type = 'linear'
        except Exception:
            # Fallback to KernelExplainer for non-linear models
            predict_fn = lambda x: self.model.predict_proba(x)[:, 1]
            self.explainer = shap.KernelExplainer(predict_fn, X_bg[:20])
            self.explainer_type = 'kernel'

    def get_shap_values(self, text: str) -> dict:
        """Get SHAP values for a single article."""
        if not SHAP_AVAILABLE or self.explainer is None:
            return {'error': 'SHAP explainer not initialized'}

        cleaned = clean_text(text)
        X = self.vectorizer.transform([cleaned])
        shap_vals = self.explainer.shap_values(X)

        feature_names = self.vectorizer.get_feature_names_out()

        if isinstance(shap_vals, list):
            vals = shap_vals[1][0]  # FAKE class
        else:
            vals = shap_vals[0]

        # Get non-zero features only (sparse input)
        nonzero_mask = np.array(X.todense())[0] > 0
        nonzero_indices = np.where(nonzero_mask)[0]

        word_shap = [(feature_names[i], float(vals[i]))
                     for i in nonzero_indices]
        word_shap.sort(key=lambda x: abs(x[1]), reverse=True)

        top_fake = [(w, s) for w, s in word_shap if s > 0][:10]
        top_real = [(w, abs(s)) for w, s in word_shap if s < 0][:10]

        return {
            'top_fake_features': top_fake,
            'top_real_features': top_real,
            'all_features': word_shap[:20],
        }

    def get_global_importance(self, texts: list, n_samples: int = 200) -> dict:
        """Compute global SHAP importance across a sample of articles."""
        if not SHAP_AVAILABLE or self.explainer is None:
            return {'error': 'SHAP explainer not initialized'}

        sample = texts[:n_samples]
        cleaned = [clean_text(t) for t in sample]
        X = self.vectorizer.transform(cleaned)
        shap_vals = self.explainer.shap_values(X)

        if isinstance(shap_vals, list):
            vals = np.abs(shap_vals[1]).mean(axis=0)
        else:
            vals = np.abs(shap_vals).mean(axis=0)

        feature_names = self.vectorizer.get_feature_names_out()
        top_indices = vals.argsort()[-20:][::-1]

        return {
            'top_features': [(feature_names[i], float(vals[i]))
                             for i in top_indices]
        }


# ─── Sentence-Level Highlighting ─────────────────────────────────────────────

def highlight_suspicious_sentences(text: str, model, vectorizer,
                                    threshold: float = 0.85) -> dict:
    """
    Split article into sentences and score each independently.
    Highlights sentences that the model considers most fake-leaning.

    Args:
        text:      Full article text
        model:     Trained classifier
        vectorizer: Fitted TF-IDF vectorizer
        threshold: Fake probability above which a sentence is flagged

    Returns:
        dict with sentences list (each with text, fake_prob, flagged)
        and html with highlighted suspicious sentences
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if len(s.split()) >= 5]

    if not sentences:
        return {'sentences': [], 'html': text}

    cleaned_sentences = [clean_text(s) for s in sentences]
    # Filter out empty after cleaning
    valid = [(orig, cl) for orig, cl in zip(sentences, cleaned_sentences) if cl.strip()]

    if not valid:
        return {'sentences': [], 'html': text}

    orig_sentences, cleaned_sentences = zip(*valid)
    X = vectorizer.transform(cleaned_sentences)
    probs = model.predict_proba(X)[:, 1]  # fake probability per sentence

    sentence_results = []
    html_parts = []

    for orig, fake_prob in zip(orig_sentences, probs):
        flagged = float(fake_prob) >= threshold
        sentence_results.append({
            'text': orig,
            'fake_prob': round(float(fake_prob), 3),
            'flagged': flagged
        })

        if flagged:
            alpha = min((float(fake_prob) - threshold) / (1 - threshold), 1.0) * 0.5 + 0.1
            html_parts.append(
                f'<span style="background:rgba(239,68,68,{alpha:.2f});'
                f'border-radius:4px;padding:2px 4px;" '
                f'title="Fake probability: {fake_prob:.1%}">{orig}</span>'
            )
        else:
            html_parts.append(orig)

    return {
        'sentences': sentence_results,
        'html': ' '.join(html_parts),
        'flagged_count': sum(1 for s in sentence_results if s['flagged']),
        'total_count': len(sentence_results),
    }
