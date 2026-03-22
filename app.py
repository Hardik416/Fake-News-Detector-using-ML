"""
app.py
------
Flask web application for the Fake News Detection System.
Routes:
  GET  /          — Main input page
  POST /predict   — Classify submitted text
  GET  /dashboard — Unsupervised analytics dashboard
  GET  /compare   — Model comparison table
  POST /api/predict — JSON API endpoint
"""

import os
import sys
import json
import pickle
import numpy as np

from flask import Flask, render_template, request, jsonify, redirect, url_for

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import clean_text
from language_handler import prepare_text_for_classification
from credibility import CredibilityScorer, combined_score

app = Flask(__name__)
app.config['SECRET_KEY'] = 'fakenews_detector_2024'

# ─── Load Models at Startup ───────────────────────────────────────────────────

MODELS_DIR = 'models/'
predictor = None
lime_explainer = None
credibility_scorer = CredibilityScorer()


def load_models():
    """Load trained models. Called once at startup."""
    global predictor, lime_explainer

    # Check if models exist
    model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    vec_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')

    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        print("WARNING: Trained models not found. Run notebooks/02_Supervised_Models.ipynb first.")
        return False

    try:
        from predict import FakeNewsPredictor
        predictor = FakeNewsPredictor(model_path, vec_path)

        try:
            from explainability import LIMEExplainer
            with open(model_path, 'rb') as f:
                bundle = pickle.load(f)
            with open(vec_path, 'rb') as f:
                vectorizer = pickle.load(f)
            lime_explainer = LIMEExplainer(bundle['model'], vectorizer)
            print("LIME explainer loaded.")
        except Exception as e:
            print(f"LIME not available: {e}")

        print("Models loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('news_text', '').strip()
    url = request.form.get('source_url', '').strip()
    headline = request.form.get('headline', '').strip()

    if not text:
        return render_template('index.html', error='Please enter some text.')

    if predictor is None:
        return render_template('result.html', demo_mode=True,
                               text=text, url=url, headline=headline,
                               result=_demo_result(text))

    return render_template('result.html',
                            **_full_predict(text, url, headline))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API endpoint for programmatic use."""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" field'}), 400

    text = data['text']
    url = data.get('url', '')
    headline = data.get('headline', '')

    if predictor is None:
        return jsonify(_demo_result(text))

    result = _full_predict(text, url, headline)
    # Remove HTML-heavy fields for JSON response
    result.pop('highlighted_html', None)
    result.pop('sentence_html', None)
    return jsonify(result)


@app.route('/dashboard')
def dashboard():
    """Analytics dashboard showing clustering and topic modeling results."""
    # Load pre-computed unsupervised results if available
    tsne_data = None
    topic_data = None

    tsne_path = os.path.join(MODELS_DIR, 'tsne_results.json')
    topics_path = os.path.join(MODELS_DIR, 'topic_words.json')

    if os.path.exists(tsne_path):
        with open(tsne_path) as f:
            tsne_data = json.load(f)

    if os.path.exists(topics_path):
        with open(topics_path) as f:
            topic_data = json.load(f)

    return render_template('dashboard.html',
                           tsne_data=json.dumps(tsne_data) if tsne_data else 'null',
                           topic_data=json.dumps(topic_data) if topic_data else 'null')


@app.route('/compare')
def compare():
    """Model comparison table."""
    results_path = os.path.join(MODELS_DIR, 'model_results.json')
    model_results = None

    if os.path.exists(results_path):
        with open(results_path) as f:
            model_results = json.load(f)

    return render_template('compare.html', model_results=model_results)


# ─── Prediction Logic ─────────────────────────────────────────────────────────

def _full_predict(text: str, url: str = '', headline: str = '') -> dict:
    """Run the full prediction pipeline."""

    # 1. Language detection + translation
    lang_result = prepare_text_for_classification(text)
    text_for_model = lang_result['text_for_model']
    language = lang_result['language']
    was_translated = lang_result['was_translated']

    # 2. ML prediction
    ml_result = predictor.predict(text_for_model)

    # 3. Source credibility scoring
    cred_result = credibility_scorer.score(
        text=text, url=url or None, headline=headline or None
    )

    # 4. Combined final score
    final = combined_score(
        ml_fake_prob=ml_result['fake_prob'],
        credibility_score=cred_result['overall_score']
    )

    # 5. LIME word highlighting (if available)
    highlighted_html = text
    fake_words = []
    if lime_explainer:
        try:
            lime_result = lime_explainer.explain(text_for_model, n_features=15)
            highlighted_html = lime_result.get('html', text)
            fake_words = lime_result.get('fake_words', [])
        except Exception as e:
            print(f"LIME error: {e}")

    # 6. Sentence-level highlighting
    sentence_html = text
    sentence_results = []
    try:
        from explainability import highlight_suspicious_sentences
        with open(os.path.join(MODELS_DIR, 'best_model.pkl'), 'rb') as f:
            bundle = pickle.load(f)
        with open(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'), 'rb') as f:
            vec = pickle.load(f)
        sent_result = highlight_suspicious_sentences(text_for_model, bundle['model'], vec)
        sentence_html = sent_result.get('html', text)
        sentence_results = sent_result.get('sentences', [])
    except Exception as e:
        print(f"Sentence highlight error: {e}")

    return {
        'text': text,
        'headline': headline,
        'url': url,
        'language': language,
        'was_translated': was_translated,
        'ml_result': ml_result,
        'credibility': cred_result,
        'final': final,
        'highlighted_html': highlighted_html,
        'sentence_html': sentence_html,
        'sentence_results': sentence_results,
        'fake_words': fake_words[:8],
        'demo_mode': False,
    }


def _demo_result(text: str) -> dict:
    """Demo result when models are not trained yet."""
    import random
    random.seed(len(text))
    fake_prob = round(random.uniform(0.3, 0.8), 3)
    return {
        'label': 'FAKE' if fake_prob > 0.5 else 'REAL',
        'confidence': fake_prob if fake_prob > 0.5 else 1 - fake_prob,
        'fake_prob': fake_prob,
        'real_prob': round(1 - fake_prob, 3),
        'note': 'DEMO MODE — Train models first by running the notebooks.',
        'demo_mode': True,
    }


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    load_models()
    app.run(debug=True, host='0.0.0.0', port=5000)
