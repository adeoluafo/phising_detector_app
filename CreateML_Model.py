import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import pickle

def preprocess_features(df, tfidf_vectorizer=None):
    df.columns = df.columns.str.strip()
    print("DataFrame columns:", df.columns.tolist())
    # Use 'text', 'email', and 'domain' columns as per your CSV files
    df['link_count'] = df['text'].apply(lambda x: x.count('http'))
    df['suspicious_keyword_count'] = df['text'].apply(lambda x: sum(1 for kw in ['free', 'winner', 'urgent'] if kw in x.lower()))
    df['body_length'] = df['text'].apply(len)
    df['uses_public_email'] = df['email'].apply(lambda x: 1 if '@gmail.com' in x or '@yahoo.com' in x else 0)
    df['domain_unknown'] = df['domain'].apply(lambda x: 1 if x not in ['trusted.com', 'safe.org'] else 0)

    if tfidf_vectorizer is None:
        tfidf_vectorizer = TfidfVectorizer(max_features=300)
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['text'])
    else:
        tfidf_matrix = tfidf_vectorizer.transform(df['text'])

    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    features = pd.concat([
        df[['link_count', 'suspicious_keyword_count', 'body_length', 'uses_public_email', 'domain_unknown']],
        tfidf_df
    ], axis=1)

    labels = df['label'] if 'label' in df.columns else None

    return features, labels, tfidf_vectorizer

def train_model(df, model_name="RandomForest"):
    X, y, tfidf_vectorizer = preprocess_features(df)
    scaler = None
    pca = None
    model = None

    if model_name == "KNN":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_scaled, y)

    elif model_name == "RandomForest":
        model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced')
        model.fit(X, y)

    elif model_name == "LogisticRegression":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=200)
        model.fit(X_scaled, y)

    elif model_name == "SVM":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = SVC(probability=True)
        model.fit(X_scaled, y)

    else:
        raise ValueError(f"Model {model_name} not implemented")

    return model, scaler, tfidf_vectorizer, pca

def predict_model(model, scaler, tfidf_vectorizer, pca, df):
    X, _, _ = preprocess_features(df, tfidf_vectorizer)

    if scaler:
        X = scaler.transform(X)
    if pca:
        X = pca.transform(X)

    preds = model.predict(X)
    return preds

def save_artifacts(model, scaler, tfidf_vectorizer, pca, filepath_prefix):
    with open(f"{filepath_prefix}_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{filepath_prefix}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(f"{filepath_prefix}_tfidf.pkl", "wb") as f:
        pickle.dump(tfidf_vectorizer, f)
    with open(f"{filepath_prefix}_pca.pkl", "wb") as f:
        pickle.dump(pca, f)

def load_artifacts(filepath_prefix):
    with open(f"{filepath_prefix}_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{filepath_prefix}_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(f"{filepath_prefix}_tfidf.pkl", "rb") as f:
        tfidf_vectorizer = pickle.load(f)
    with open(f"{filepath_prefix}_pca.pkl", "rb") as f:
        pca = pickle.load(f)
    return model, scaler, tfidf_vectorizer, pca
