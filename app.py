"""
app.py - TruthLens Fake News Detector
Full pipeline: Language Detection → BERT + Classical ML → Credibility → Result
"""

import os, sys, pickle
from flask import Flask, render_template, request, jsonify

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import clean_text
from language_handler import prepare_text_for_classification
from credibility import CredibilityScorer, combined_score

app = Flask(__name__)
app.config['SECRET_KEY'] = 'fakenews_detector_2024'

MODELS_DIR = 'models/'
BERT_DIR   = 'models/bert_model/'

# Global model objects
classical_predictor = None
bert_model          = None
bert_tokenizer      = None
credibility_scorer  = CredibilityScorer()
DEVICE              = None


def load_models():
    global classical_predictor, bert_model, bert_tokenizer, DEVICE

    # 1. Classical ML model
    model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    vec_path   = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
    if os.path.exists(model_path) and os.path.exists(vec_path):
        try:
            from predict import FakeNewsPredictor
            classical_predictor = FakeNewsPredictor(model_path, vec_path)
            print(f'Classical model loaded: {classical_predictor.model_name}')
        except Exception as e:
            print(f'Classical model error: {e}')
    else:
        print('Classical model not found - run Notebook 02 first')

    # 2. BERT model
    bert_config = os.path.join(BERT_DIR, 'config.json')
    if os.path.exists(bert_config):
        try:
            import torch
            from transformers import (DistilBertTokenizerFast,
                                      DistilBertForSequenceClassification)
            DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            bert_tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_DIR)
            bert_model     = DistilBertForSequenceClassification.from_pretrained(BERT_DIR)
            bert_model     = bert_model.to(DEVICE)
            bert_model.eval()
            print(f'BERT model loaded on {DEVICE}')
        except Exception as e:
            print(f'BERT model error: {e}')
    else:
        print('BERT model not found - run Notebook 04 first')


def predict_bert(text: str) -> dict:
    try:
        import torch
        encoding = bert_tokenizer(
            text, max_length=256, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        input_ids      = encoding['input_ids'].to(DEVICE)
        attention_mask = encoding['attention_mask'].to(DEVICE)
        with torch.no_grad():
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            probs   = torch.softmax(outputs.logits, dim=1)[0]
            pred    = outputs.logits.argmax(dim=1).item()
        fake_prob = round(probs[1].item(), 4)
        real_prob = round(probs[0].item(), 4)
        return {
            'label':      'FAKE' if pred == 1 else 'REAL',
            'fake_prob':  fake_prob,
            'real_prob':  real_prob,
            'confidence': round(max(probs).item(), 4),
            'available':  True
        }
    except Exception as e:
        return {'available': False, 'error': str(e)}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    text     = request.form.get('news_text', '').strip()
    url      = request.form.get('source_url', '').strip()
    headline = request.form.get('headline', '').strip()
    if not text:
        return render_template('index.html', error='Please enter some text.')
    return render_template('result.html', **_full_predict(text, url, headline))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing text field'}), 400
    result = _full_predict(data['text'], data.get('url',''), data.get('headline',''))
    result.pop('highlighted_html', None)
    return jsonify(result)


def _full_predict(text, url='', headline=''):

    # 1. Language detection + translation
    lang_result    = prepare_text_for_classification(text)
    text_for_model = lang_result['text_for_model']
    language       = lang_result['language']
    was_translated = lang_result['was_translated']

    # 2. Classical ML
    classical_result = None
    if classical_predictor:
        try:
            classical_result = classical_predictor.predict(text_for_model)
        except Exception as e:
            print(f'Classical error: {e}')

    # 3. BERT
    bert_result = None
    if bert_model and bert_tokenizer:
        bert_result = predict_bert(text_for_model)

    # 4. Primary model = BERT if available, else classical
    if bert_result and bert_result.get('available'):
        primary_result = bert_result
        primary_model  = 'DistilBERT'
    elif classical_result:
        primary_result = classical_result
        primary_model  = classical_result.get('model_used', 'Classical ML')
    else:
        primary_result = {'label':'UNKNOWN','fake_prob':0.5,'real_prob':0.5,'confidence':0.5}
        primary_model  = 'None'

    # 5. Source credibility
    cred_result = credibility_scorer.score(text=text, url=url or None, headline=headline or None)

    # 6. Combined final score
    final = combined_score(
        ml_fake_prob=primary_result['fake_prob'],
        credibility_score=cred_result['overall_score']
    )

    # 7. Sentence highlighting
    highlighted_html = text
    try:
        from explainability import highlight_suspicious_sentences
        if classical_predictor and hasattr(classical_predictor, 'model'):
            sent = highlight_suspicious_sentences(
                text_for_model, 
                classical_predictor.model, 
                classical_predictor.vectorizer, 
                threshold=0.85
            )
            highlighted_html = sent.get('html', text)
    except Exception as e:
        print(f'Highlight error: {e}')

    return {
        'text':             text,
        'headline':         headline,
        'url':              url,
        'language':         language,
        'was_translated':   was_translated,
        'primary_result':   primary_result,
        'primary_model':    primary_model,
        'bert_result':      bert_result,
        'classical_result': classical_result,
        'credibility':      cred_result,
        'final':            final,
        'highlighted_html': highlighted_html
    }


if __name__ == '__main__':
    load_models()
    app.run(debug=True, host='0.0.0.0', port=5000)
