"""
predict.py
----------
Train and evaluate supervised ML models for fake news classification.
Models: Logistic Regression, Random Forest, XGBoost, + BERT wrapper.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report,
    confusion_matrix
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from preprocess import clean_text, build_tfidf, load_vectorizer


# ─── Model Definitions ────────────────────────────────────────────────────────

def get_models() -> dict:
    """Return dict of model name → untrained sklearn model."""
    models = {
        'Logistic Regression': LogisticRegression(
            C=1.0, max_iter=1000, solver='lbfgs',
            class_weight='balanced', random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_split=5,
            class_weight='balanced', n_jobs=-1, random_state=42
        ),
    }
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric='logloss',
            random_state=42, n_jobs=-1
        )
    return models


# ─── Training ─────────────────────────────────────────────────────────────────

def train_all_models(X_train_tfidf, y_train, X_test_tfidf, y_test,
                     save_dir: str = 'models/') -> pd.DataFrame:
    """
    Train all models, evaluate, and save the best one.

    Returns:
        DataFrame with evaluation metrics for each model.
    """
    models = get_models()
    results = []

    best_f1 = 0
    best_model = None
    best_name = ''

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_tfidf, y_train)

        y_pred = model.predict(X_test_tfidf)
        y_prob = model.predict_proba(X_test_tfidf)[:, 1]

        metrics = {
            'Model': name,
            'Accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
            'F1 Score': round(f1_score(y_test, y_pred) * 100, 2),
            'Precision': round(precision_score(y_test, y_pred) * 100, 2),
            'Recall': round(recall_score(y_test, y_pred) * 100, 2),
            'ROC-AUC': round(roc_auc_score(y_test, y_prob) * 100, 2),
        }
        results.append(metrics)
        print(f"  Accuracy: {metrics['Accuracy']}%  F1: {metrics['F1 Score']}%  AUC: {metrics['ROC-AUC']}%")

        # Save model
        model_path = f"{save_dir}{name.lower().replace(' ', '_')}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"  Saved → {model_path}")

        if metrics['F1 Score'] > best_f1:
            best_f1 = metrics['F1 Score']
            best_model = model
            best_name = name

    print(f"\n✦ Best model: {best_name} (F1={best_f1}%)")
    with open(f"{save_dir}best_model.pkl", 'wb') as f:
        pickle.dump({'model': best_model, 'name': best_name}, f)

    return pd.DataFrame(results).sort_values('F1 Score', ascending=False)


# ─── Inference ────────────────────────────────────────────────────────────────

class FakeNewsPredictor:
    """
    Lightweight inference wrapper used by the Flask app.
    Loads the best trained model + vectorizer and predicts on raw text.
    """

    def __init__(self, model_path: str = 'models/best_model.pkl',
                 vectorizer_path: str = 'models/tfidf_vectorizer.pkl'):
        with open(model_path, 'rb') as f:
            bundle = pickle.load(f)
            self.model = bundle['model']
            self.model_name = bundle['name']

        self.vectorizer = load_vectorizer(vectorizer_path)
        print(f"Loaded model: {self.model_name}")

    def predict(self, text: str) -> dict:
        """
        Predict whether a single news article is fake or real.

        Returns:
            dict:
                label     — 'FAKE' or 'REAL'
                confidence — float 0.0–1.0 (probability of predicted class)
                fake_prob  — float 0.0–1.0 (probability it's fake)
                real_prob  — float 0.0–1.0 (probability it's real)
        """
        cleaned = clean_text(text)
        X = self.vectorizer.transform([cleaned])

        label_idx = self.model.predict(X)[0]
        probs = self.model.predict_proba(X)[0]

        fake_prob = float(probs[1])
        real_prob = float(probs[0])
        label = 'FAKE' if label_idx == 1 else 'REAL'
        confidence = fake_prob if label == 'FAKE' else real_prob

        return {
            'label': label,
            'confidence': round(confidence, 4),
            'fake_prob': round(fake_prob, 4),
            'real_prob': round(real_prob, 4),
            'model_used': self.model_name,
        }

    def predict_batch(self, texts: list) -> list:
        """Predict on a list of texts. Returns list of dicts."""
        return [self.predict(t) for t in texts]


# ─── Evaluation Utilities ────────────────────────────────────────────────────

def print_full_report(y_true, y_pred, target_names=('Real', 'Fake')):
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=target_names))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(f"  True Real:  {cm[0][0]:>6}  |  False Fake: {cm[0][1]:>6}")
    print(f"  False Real: {cm[1][0]:>6}  |  True Fake:  {cm[1][1]:>6}")
    print("="*60)


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from preprocess import load_welfake, split_data

    print("Loading dataset...")
    df = load_welfake('datasets/welfake_dataset.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    X_train, X_test, y_train, y_test = split_data(df)

    print("\nBuilding TF-IDF vectors...")
    vectorizer, X_train_tfidf = build_tfidf(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    from preprocess import save_vectorizer
    save_vectorizer(vectorizer)

    print("\nTraining models...")
    results_df = train_all_models(X_train_tfidf, y_train, X_test_tfidf, y_test)

    print("\n" + "="*60)
    print("MODEL COMPARISON TABLE")
    print("="*60)
    print(results_df.to_string(index=False))
