# TruthLens — Multilingual Fake News Detection System

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-3.0-green?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?style=flat-square)
![BERT](https://img.shields.io/badge/DistilBERT-HuggingFace-yellow?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

A production-grade, multilingual fake news detection system with multi-regional support for major Indian languages (Hindi, Marathi, Bengali, Punjabi, Tamil, Telugu, Gujarati), combining **supervised ML**, **explainability (LIME/SHAP)**, and **source credibility analysis** — with a dark-themed Flask web interface.

---

## Architecture Overview

```
Input Text / Article
        │
        ▼
┌─────────────────────────────────┐
│  Language Detection(langdetect) │
│  Regional → English Translation │
└────────────────┬────────────────┘
                 │
        ┌────────▼────────┐
        │  Text Cleaning  │  (preprocess.py)
        │  TF-IDF Vectors │
        └────────┬────────┘
                 │
    ┌──────────────────────┼──────────────────────┐
    │                      │                      │
    ▼                      ▼                      ▼
ML Models (BERT)         Source              Explainability
+ Classical ML         Credibility           (LIME + SHAP)
(XGBoost / LR)       (credibility.py)       (explainability.py)
    │                      │                      │
    └──────────────────────┼──────────────────────┘
                           │
                           │
                  ┌────────▼────────┐
                  │  Combined Score │  60% ML + 40% Credibility
                  │  Final Verdict  │
                  └─────────────────┘
```

---

## Features

| Feature | Details |
|---|---|
| **Supervised ML** | Logistic Regression, Random Forest, XGBoost — compared on F1/AUC |
| **Deep Learning** | Fine-tuned `distilbert-base-multilingual-cased` (Week 4) |
| **Multilingual** | English + 7 Major Indian Languages (Hindi, Marathi, Bengali, Tamil, etc.) |
| **Explainability** | LIME word-level highlighting, SHAP feature importance |
| **Credibility** | Domain reputation, URL analysis, headline clickbait detection |
| **Web UI** | Dark-themed Flask app with confidence meters and sentence highlighting |
| **API** | REST JSON endpoint for programmatic use |

---

## Datasets

All datasets are freely available online:

| Dataset | Size | Source | Use |
|---|---|---|---|
| **WELFake** | 72K articles | Kaggle | Primary supervised training |
| **ISOT Fake News** | 44K articles | Kaggle / UVic | Supervised + topic modeling |

### Download Links
- WELFake: `https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification`
- ISOT: `https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset`

---

## Project Structure

```
FakeNewsDetector/
├── datasets/                    # Raw + cleaned datasets
│   ├── welfake_dataset.csv
│   └── isot_fake.csv / isot_real.csv
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb
│   ├── 02_Supervised_Models.ipynb
│   ├── 04_BERT_Finetune.ipynb
│   └── 05_Explainability_LIME_SHAP.ipynb
├── models/                      # Saved trained models
│   ├── best_model.pkl
│   └── tfidf_vectorizer.pkl
├── src/
│   ├── preprocess.py            # Text cleaning + TF-IDF
│   ├── predict.py               # Supervised ML training + inference
│   ├── language_handler.py      # Language detection + translation
│   ├── explainability.py        # LIME + SHAP
│   └── credibility.py           # Source credibility scoring
├── templates/                   # Flask HTML templates
│   ├── index.html
│   └── result.html
├── tests/
│   ├── test_preprocess.py
├── app.py                       # Flask application
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/yourusername/FakeNewsDetector
cd FakeNewsDetector
pip install -r requirements.txt

# 2. Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# 3. Download datasets (see links above) → place in datasets/

# 4. Run notebooks in order:
#    notebooks/01_EDA_and_Preprocessing.ipynb
#    notebooks/02_Supervised_Models.ipynb
#    notebooks/03_BERT_Finetune.ipynb


# 5. Start Flask app
python app.py

# 6. Open http://localhost:5000
```

---

## Model Results

| Model | Accuracy | F1 Score | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~94% | ~93% | ~94% | ~93% | ~98% |
| Random Forest | ~95% | ~95% | ~96% | ~94% | ~99% |
| XGBoost | ~96% | ~96% | ~96% | ~96% | ~99% |
| DistilBERT (fine-tuned) | ~98% | ~97% | ~98% | ~97% | ~99.5% |

*Results on WELFake dataset, 80/20 train-test split.*

---

## API Usage

```python
import requests

response = requests.post('http://localhost:5000/api/predict', json={
    'text': 'Scientists discover water on Mars in peer-reviewed study.',
    'url': 'https://reuters.com/article/...',
    'headline': 'NASA confirms water discovery'
})

print(response.json())
# {
#   'final_verdict': 'REAL',
#   'final_fake_prob': 0.08,
#   'confidence_label': 'High confidence',
#   'credibility': {'overall_score': 92, 'verdict': 'CREDIBLE'},
#   'ml_result': {'label': 'REAL', 'confidence': 0.92}
# }
```

---

## Running Tests

```bash
pytest tests/ -v --cov=src/
```

---

## Tech Stack

- **ML**: scikit-learn, XGBoost, HuggingFace Transformers
- **NLP**: NLTK, TF-IDF, DistilBERT (multilingual)
- **Explainability**: LIME, SHAP
- **Language**: langdetect, deep-translator (Google)
- **Visualization**: matplotlib, seaborn, plotly, t-SNE
- **Web**: Flask, Jinja2, HTML/CSS/JS
- **Testing**: pytest, pytest-cov
- **Deployment**: Gunicorn + Render/Railway (free tier)

---

## Author

Built as a college ML project demonstrating:
- Supervised learning on real-world NLP data
- Model explainability and interpretability
- Multilingual AI system design
- Full-stack deployment

---

## License

MIT License — see LICENSE file.
