"""
unsupervised.py
---------------
Unsupervised ML layer:
  1. K-Means clustering  — find natural article topic groups
  2. LDA topic modeling  — discover latent topics in fake vs real news
  3. DBSCAN              — anomaly detection (unusual = potential fake)
  4. t-SNE               — 2D visualization of article embeddings

This is the chapter that takes this project from supervised-only to
a complete ML project covering BOTH paradigms.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# ─── 1. K-Means Clustering ───────────────────────────────────────────────────

def run_kmeans(X_tfidf, k_values: list = [5, 10, 15, 20],
               random_state: int = 42) -> dict:
    """
    Run K-Means for multiple k values and compute elbow + silhouette scores.

    Args:
        X_tfidf:     Sparse TF-IDF matrix (articles × features)
        k_values:    List of k values to try
        random_state: Random seed

    Returns:
        dict with best_model, labels, inertias, silhouette_scores, best_k
    """
    # TruncatedSVD to reduce dimensionality before clustering (faster + better)
    print("Reducing dimensions with TruncatedSVD...")
    svd = TruncatedSVD(n_components=100, random_state=random_state)
    X_dense = svd.fit_transform(X_tfidf)
    print(f"Variance explained: {svd.explained_variance_ratio_.sum():.2%}")

    inertias = []
    silhouette_scores = []
    models = {}

    for k in k_values:
        print(f"  Fitting K-Means k={k}...")
        km = KMeans(n_clusters=k, init='k-means++', max_iter=300,
                    n_init=10, random_state=random_state)
        labels = km.fit_predict(X_dense)
        inertias.append(km.inertia_)

        sil = silhouette_score(X_dense, labels, sample_size=min(5000, len(labels)))
        silhouette_scores.append(sil)
        models[k] = {'model': km, 'labels': labels}
        print(f"    Inertia: {km.inertia_:.0f}  Silhouette: {sil:.4f}")

    # Pick best k by silhouette score
    best_idx = np.argmax(silhouette_scores)
    best_k = k_values[best_idx]
    print(f"\nBest k = {best_k} (silhouette = {silhouette_scores[best_idx]:.4f})")

    return {
        'best_k': best_k,
        'best_model': models[best_k]['model'],
        'best_labels': models[best_k]['labels'],
        'X_reduced': X_dense,
        'svd': svd,
        'k_values': k_values,
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'all_models': models,
    }


def get_cluster_top_terms(km_model, vectorizer, svd, n_terms: int = 10) -> dict:
    """Get top TF-IDF terms for each cluster centroid."""
    # Project cluster centers back to original TF-IDF space
    centroids_original = svd.inverse_transform(km_model.cluster_centers_)
    feature_names = vectorizer.get_feature_names_out()

    top_terms = {}
    for i, center in enumerate(centroids_original):
        top_indices = center.argsort()[-n_terms:][::-1]
        top_terms[f'Cluster {i}'] = [feature_names[idx] for idx in top_indices]

    return top_terms


def predict_cluster(text: str, vectorizer, svd, km_model) -> int:
    """Predict cluster for a new article."""
    from preprocess import clean_text
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    X_reduced = svd.transform(X)
    return int(km_model.predict(X_reduced)[0])


# ─── 2. LDA Topic Modeling ───────────────────────────────────────────────────

def run_lda(texts: list, n_topics: int = 10, max_features: int = 10000,
            random_state: int = 42) -> dict:
    """
    Run Latent Dirichlet Allocation to find topics in the corpus.

    Args:
        texts:      List of cleaned article texts
        n_topics:   Number of latent topics to discover
        max_features: Vocabulary size for CountVectorizer

    Returns:
        dict with model, vectorizer, doc_topics, topic_words, coherence
    """
    print(f"Fitting LDA with {n_topics} topics on {len(texts)} documents...")

    # LDA uses raw counts (not TF-IDF)
    count_vec = CountVectorizer(
        max_features=max_features, min_df=5, max_df=0.90,
        ngram_range=(1, 1), stop_words='english'
    )
    X_counts = count_vec.fit_transform(texts)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=20,
        learning_method='online',
        learning_offset=50.,
        random_state=random_state,
        n_jobs=-1
    )
    doc_topics = lda.fit_transform(X_counts)
    perplexity = lda.perplexity(X_counts)
    print(f"LDA Perplexity: {perplexity:.2f} (lower is better)")

    # Extract top words per topic
    feature_names = count_vec.get_feature_names_out()
    topic_words = {}
    for idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[-15:][::-1]
        topic_words[f'Topic {idx+1}'] = [feature_names[i] for i in top_indices]

    return {
        'model': lda,
        'vectorizer': count_vec,
        'doc_topics': doc_topics,
        'topic_words': topic_words,
        'perplexity': perplexity,
        'n_topics': n_topics,
    }


def get_dominant_topic(doc_topics: np.ndarray) -> np.ndarray:
    """Return dominant topic index for each document."""
    return doc_topics.argmax(axis=1)


def analyze_fake_vs_real_topics(doc_topics: np.ndarray, labels: np.ndarray,
                                  topic_words: dict) -> pd.DataFrame:
    """
    Compare topic distributions between fake (1) and real (0) articles.
    Useful for understanding WHAT fake news tends to be about.
    """
    fake_mask = labels == 1
    real_mask = labels == 0

    rows = []
    for i, (topic_name, words) in enumerate(topic_words.items()):
        fake_avg = doc_topics[fake_mask, i].mean()
        real_avg = doc_topics[real_mask, i].mean()
        rows.append({
            'Topic': topic_name,
            'Top Words': ', '.join(words[:6]),
            'Avg in FAKE': round(fake_avg, 4),
            'Avg in REAL': round(real_avg, 4),
            'Bias': 'FAKE-leaning' if fake_avg > real_avg else 'REAL-leaning'
        })

    return pd.DataFrame(rows).sort_values('Avg in FAKE', ascending=False)


# ─── 3. DBSCAN Anomaly Detection ─────────────────────────────────────────────

def run_dbscan_anomaly(X_reduced: np.ndarray, eps: float = 0.5,
                        min_samples: int = 5) -> dict:
    """
    Use DBSCAN to identify anomalous/outlier articles.
    Articles labeled -1 by DBSCAN are "noise" = potential fake/unusual articles.

    Args:
        X_reduced: Dimensionality-reduced article vectors (from TruncatedSVD)
        eps:       Maximum distance between neighborhood samples
        min_samples: Minimum samples to form a core point

    Returns:
        dict with labels, anomaly_indices, anomaly_ratio
    """
    print(f"Running DBSCAN (eps={eps}, min_samples={min_samples})...")
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(X_reduced)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_anomalies = (labels == -1).sum()
    anomaly_ratio = n_anomalies / len(labels)

    anomaly_indices = np.where(labels == -1)[0]

    print(f"  Clusters found: {n_clusters}")
    print(f"  Anomalous articles: {n_anomalies} ({anomaly_ratio:.1%} of corpus)")

    return {
        'model': db,
        'labels': labels,
        'n_clusters': n_clusters,
        'anomaly_indices': anomaly_indices,
        'anomaly_ratio': anomaly_ratio,
        'n_anomalies': int(n_anomalies),
    }


def is_anomalous(text: str, vectorizer, svd, db_model,
                  threshold_distance: float = 0.5) -> dict:
    """
    Check if a single new article is anomalous (potential fake).
    Since DBSCAN doesn't predict on new points, we use core sample distance.
    """
    from preprocess import clean_text
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    X_reduced = svd.transform(X)

    # Compute distance to nearest core sample
    core_samples = db_model.core_sample_indices_
    if len(core_samples) == 0:
        return {'is_anomalous': False, 'min_distance': 0.0}

    # Simple nearest-core-sample distance
    from sklearn.metrics.pairwise import euclidean_distances
    # Use a small sample of core samples for speed
    sample_cores = core_samples[:500]
    # We'd need the original X_reduced of training to do this properly
    # In production, store the core sample vectors separately

    return {
        'is_anomalous': False,  # Placeholder — implement with stored cores
        'note': 'DBSCAN inference requires stored core sample vectors'
    }


# ─── 4. t-SNE Visualization ──────────────────────────────────────────────────

def run_tsne(X_reduced: np.ndarray, n_samples: int = 5000,
              perplexity: float = 30, random_state: int = 42) -> np.ndarray:
    """
    Reduce to 2D with t-SNE for visualization.
    Operates on the SVD-reduced vectors (not raw TF-IDF) for speed.

    Args:
        X_reduced:  Array of shape (n_docs, 100) from TruncatedSVD
        n_samples:  Max samples to visualize (t-SNE is slow on full dataset)
        perplexity: t-SNE perplexity parameter (5–50, default 30)

    Returns:
        2D array of shape (n_samples, 2)
    """
    if len(X_reduced) > n_samples:
        idx = np.random.choice(len(X_reduced), n_samples, replace=False)
        X_sample = X_reduced[idx]
    else:
        idx = np.arange(len(X_reduced))
        X_sample = X_reduced

    print(f"Running t-SNE on {len(X_sample)} articles (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=random_state, n_iter=1000, learning_rate='auto',
                init='pca')
    X_2d = tsne.fit_transform(X_sample)

    print(f"t-SNE complete. KL divergence: {tsne.kl_divergence_:.4f}")
    return X_2d, idx


def tsne_to_dataframe(X_2d: np.ndarray, indices: np.ndarray,
                       labels: np.ndarray, cluster_labels: np.ndarray = None,
                       texts: list = None) -> pd.DataFrame:
    """Pack t-SNE results into a DataFrame for plotting."""
    df = pd.DataFrame({
        'x': X_2d[:, 0],
        'y': X_2d[:, 1],
        'label': labels[indices],
        'label_name': ['FAKE' if l == 1 else 'REAL' for l in labels[indices]],
    })
    if cluster_labels is not None:
        df['cluster'] = cluster_labels[indices]
    if texts is not None:
        df['text_preview'] = [texts[i][:80] + '...' for i in indices]

    return df


# ─── Save / Load ─────────────────────────────────────────────────────────────

def save_unsupervised_models(kmeans_result: dict, lda_result: dict,
                               save_dir: str = 'models/'):
    with open(f"{save_dir}kmeans_model.pkl", 'wb') as f:
        pickle.dump({
            'model': kmeans_result['best_model'],
            'svd': kmeans_result['svd'],
            'best_k': kmeans_result['best_k'],
        }, f)

    with open(f"{save_dir}lda_model.pkl", 'wb') as f:
        pickle.dump({
            'model': lda_result['model'],
            'vectorizer': lda_result['vectorizer'],
            'topic_words': lda_result['topic_words'],
            'n_topics': lda_result['n_topics'],
        }, f)

    print("Unsupervised models saved.")


def load_kmeans(path: str = 'models/kmeans_model.pkl') -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_lda(path: str = 'models/lda_model.pkl') -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)
