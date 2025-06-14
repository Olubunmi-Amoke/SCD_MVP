import pandas as pd
import numpy as np
import re
import joblib
from textblob import TextBlob
from sklearn.decomposition import PCA

# Load preprocessing artifacts
tfidf = joblib.load("models/tfidf_vectorizer.pkl")
scaler_pain = joblib.load("models/scaler_pain_level.pkl")
pca_pain = joblib.load("models/pca_pain_level.pkl")
scaler_emo = joblib.load("models/scaler_emotion.pkl")
pca_emo = joblib.load("models/pca_emotion.pkl")

# Load trained models
pain_model = joblib.load("models/pain_level_model.pkl")
emotion_model = joblib.load("models/emotion_classifier.pkl")

# Load feature template (column order expected by the model)
feature_cols_pain = joblib.load("models/feature_columns_pain.pkl")
feature_cols_emo = joblib.load("models/feature_columns_emotion.pkl")

# Load LabelEncoder used during training for emotion labels
emotion_label_encoder = joblib.load("models/emotion_label_encoder.pkl")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def pca_consolidate(df, feature_group, new_feature_name):
    pca = PCA(n_components=1)
    transformed = pca.fit_transform(df[feature_group])
    df[new_feature_name] = transformed
    df = df.drop(columns=feature_group)
    return df


def apply_pca_consolidation(df):
    pca_groups = [
        (['feel', 'don', 'know'], 'pca_feel_don_know'),
        (['mentions_sleep', 'mild', 'tired'], 'pca_fatigue_group'),
        (['really', 'struggling', 'today'], 'pca_distress_group'),
        (['skip', 'work'], 'pca_work_impact'),
        (['day', 'good'], 'day_good_pca')
    ]
    for group, name in pca_groups:
        if all(col in df.columns for col in group):
            df = pca_consolidate(df, group, name)
    return df

def preprocess_input(record):
    """
    record: dict with keys including 'free_text_notes', 'age', 'pain_location', 'gender', 'symptoms', 'emotion'
    Returns: preprocessed DataFrames aligned with training feature columns
    """
    df = pd.DataFrame([record])
    
    # Text cleaning and sentiment
    df['free_text_notes'] = df['free_text_notes'].fillna("")
    df['sentiment_score'] = df['free_text_notes'].apply(lambda x: TextBlob(x).sentiment.polarity)

    cleaned = df['free_text_notes'].apply(clean_text)
    tfidf_matrix = tfidf.transform(cleaned)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
    df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
    df.drop(columns=['free_text_notes'], inplace=True)

    # Age grouping
    age_bins = [0, 18, 25, 35, 50]
    age_labels = ['Under 18', '18–25', '26–35', '36–50']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)

    # Sentiment binning
    sent_bins = [-np.inf, 0.15, 0.45, np.inf]
    sent_labels = ['Low', 'Medium', 'High']
    df['sentiment_score_bin'] = pd.cut(df['sentiment_score'], bins=sent_bins, labels=sent_labels)

    # Drop raw columns
    df.drop(columns=['age', 'sentiment_score'], inplace=True)

    # One-hot encode
    categorical_cols_pain = ['pain_location', 'emotion', 'gender', 'symptoms', 'age_group', 'sentiment_score_bin']
    categorical_cols_emo = ['pain_location', 'gender', 'symptoms', 'age_group', 'sentiment_score_bin']

    df_pain = pd.get_dummies(df.copy(), columns=categorical_cols_pain, prefix=categorical_cols_pain, drop_first=True)
    df_emo = pd.get_dummies(df.copy(), columns=categorical_cols_emo, prefix=categorical_cols_emo, drop_first=True)

     # Apply PCA consolidation
    df_pain = apply_pca_consolidation(df_pain)
    df_emo = apply_pca_consolidation(df_emo)
    
    return df_pain, df_emo

def align_features(df_input, feature_cols):
    # Add missing columns
    missing_cols = [col for col in feature_cols if col not in df_input]
    for col in missing_cols:
        df_input[col] = 0

    # Drop extra columns not used during training
    extra_cols = [col for col in df_input.columns if col not in feature_cols]
    if extra_cols:
        print(f"Dropping extra columns not seen during training: {extra_cols}")

    df_input = df_input[feature_cols]

    # Debugging: Print feature alignment status
    print(f"Final feature shape: {df_input.shape}")
    print(f"Expected feature columns count: {len(feature_cols)}")
    print(f"Missing columns added: {missing_cols}")

    return df_input

def predict_pain_emotion(record):
    df_pain, df_emo = preprocess_input(record)

    # PAIN LEVEL prediction
    X_pain = align_features(df_pain.copy(), feature_cols_pain)
    X_pain_scaled = scaler_pain.transform(X_pain)
    X_pain_pca = pca_pain.transform(X_pain_scaled)
    
    print("Actual PainLevel input shape:", X_pain_pca.shape)
    print("Expected features by model:", pain_model.n_features_in_)
    
    pain_pred = pain_model.predict(X_pain_pca)[0]

    # EMOTION prediction
    X_emo = align_features(df_emo.copy(), feature_cols_emo)
    X_emo_scaled = scaler_emo.transform(X_emo)
    X_emo_pca = pca_emo.transform(X_emo_scaled)
    
    print("Actual EMO input shape:", X_emo_scaled.shape)
    print("Expected features by model:", emotion_model.n_features_in_)

    emo_pred = emotion_model.predict(X_emo_scaled)[0]
    decoded_emo = emotion_label_encoder.inverse_transform([emo_pred])[0]

    return pain_pred, decoded_emo
