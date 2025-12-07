# lgbm_advanced_improved.py
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import re

SEED = 993
random.seed(SEED)
np.random.seed(SEED)

def improved_lightgbm_solution():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    def preprocess_text(text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def create_features(df):
        df_processed = df.copy()
        text_cols = ['query', 'product_title', 'product_description', 'product_bullet_point']
        for col in text_cols:
            df_processed[f'{col}_clean'] = df_processed[col].apply(preprocess_text)

        df_processed['query_title'] = df_processed['query_clean'] + ' ' + df_processed['product_title_clean']
        df_processed['query_description'] = df_processed['query_clean'] + ' ' + df_processed['product_description_clean']
        df_processed['full_text'] = (
            df_processed['product_title_clean'] + ' ' +
            df_processed['product_description_clean'] + ' ' +
            df_processed['product_bullet_point_clean']
        )

        df_processed['query_len'] = df_processed['query_clean'].apply(len)
        df_processed['title_len'] = df_processed['product_title_clean'].apply(len)
        df_processed['desc_len'] = df_processed['product_description_clean'].apply(len)
        df_processed['bullet_len'] = df_processed['product_bullet_point_clean'].apply(len)

        df_processed['has_brand'] = (~df_processed['product_brand'].isna()).astype(int)
        df_processed['has_color'] = (~df_processed['product_color'].isna()).astype(int)
        df_processed['has_bullets'] = (
            ~df_processed['product_bullet_point'].isna() &
            (df_processed['product_bullet_point'] != '')
        ).astype(int)

        def calculate_word_overlap(row):
            query_words = set(row['query_clean'].split())
            title_words = set(row['product_title_clean'].split())
            desc_words = set(row['product_description_clean'].split())
            title_overlap = len(query_words & title_words)
            desc_overlap = len(query_words & desc_words)
            return pd.Series([
                title_overlap,
                desc_overlap,
                title_overlap / max(len(query_words), 1),
                (title_overlap + desc_overlap) / max(len(query_words), 1)
            ])

        overlap_results = df_processed.apply(calculate_word_overlap, axis=1)
        df_processed['title_word_overlap'] = overlap_results[0]
        df_processed['desc_word_overlap'] = overlap_results[1]
        df_processed['title_overlap_ratio'] = overlap_results[2]
        df_processed['total_overlap_ratio'] = overlap_results[3]

        df_processed['exact_match_in_title'] = df_processed.apply(
            lambda x: 1 if x['query_clean'] in x['product_title_clean'] else 0,
            axis=1
        )

        return df_processed

    train_processed = create_features(train_df)
    test_processed = create_features(test_df)

    vectorizer1 = TfidfVectorizer(
        max_features=1500,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )

    vectorizer2 = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 3),
        min_df=3
    )

    X_train_text1 = vectorizer1.fit_transform(train_processed['query_title'])
    X_test_text1 = vectorizer1.transform(test_processed['query_title'])

    X_train_text2 = vectorizer2.fit_transform(train_processed['full_text'])
    X_test_text2 = vectorizer2.transform(test_processed['full_text'])

    numeric_features = [
        'query_len', 'title_len', 'desc_len', 'bullet_len',
        'has_brand', 'has_color', 'has_bullets',
        'title_word_overlap', 'desc_word_overlap',
        'title_overlap_ratio', 'total_overlap_ratio',
        'exact_match_in_title'
    ]

    X_train_numeric = train_processed[numeric_features].values
    X_test_numeric = test_processed[numeric_features].values

    scaler = StandardScaler()
    X_train_numeric = scaler.fit_transform(X_train_numeric)
    X_test_numeric = scaler.transform(X_test_numeric)

    X_train = np.hstack([X_train_text1.toarray(), X_train_text2.toarray(), X_train_numeric])
    X_test = np.hstack([X_test_text1.toarray(), X_test_text2.toarray(), X_test_numeric])

    y_train = train_processed['relevance']

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 255,
        'max_depth': 10,
        'min_data_in_leaf': 25,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.2,
        'lambda_l2': 0.2,
        'min_gain_to_split': 0.01,
        'verbose': -1,
        'random_state': SEED,
        'n_jobs': -1
    }

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train
    )

    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(100)
        ]
    )

    val_pred = model.predict(X_val)
    from sklearn.metrics import mean_squared_error
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"RMSE: {val_rmse:.4f}")

    predictions = model.predict(X_test)

    def enhance_predictions(preds):
        preds_scaled = (preds - preds.min()) / (preds.max() - preds.min()) * 10
        preds_enhanced = np.where(preds_scaled > 5, preds_scaled * 1.2, preds_scaled * 0.9)
        preds_enhanced = np.clip(preds_enhanced, 0, 10)
        return preds_enhanced

    predictions = enhance_predictions(predictions)

    submission = pd.DataFrame({
        'id': test_df['id'],
        'prediction': predictions
    })

    submission.to_csv('result.csv', index=False)
    print("Файл сохранён: result.csv")

    return submission

if __name__ == "__main__":
    submission = improved_lightgbm_solution()
