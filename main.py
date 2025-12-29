import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import warnings
import os
import sys
warnings.filterwarnings('ignore')

np.random.seed(322)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
# train = pd.read_csv('')
# test = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/ML/3_comp/test.csv')
sample_submission = pd.read_csv('results/sample_submission.csv')


test['price_p05'] = np.nan
test['price_p95'] = np.nan


train['is_train'] = 1
test['is_train'] = 0
data = pd.concat([train, test], ignore_index=True)


data['dt'] = pd.to_datetime(data['dt'])
data['year'] = data['dt'].dt.year
data['quarter'] = data['dt'].dt.quarter
data['is_weekend'] = (data['dow'] >= 5).astype(int)
data['is_month_start'] = data['dt'].dt.is_month_start.astype(int)
data['is_month_end'] = data['dt'].dt.is_month_end.astype(int)
data['day_of_year'] = data['dt'].dt.dayofyear
data['week_of_month'] = data['dt'].dt.day // 7 + 1


cat_cols = ['management_group_id', 'first_category_id', 'second_category_id', 'third_category_id']


for col in cat_cols:
    freq = data[col].value_counts(normalize=True)
    data[f'{col}_freq'] = data[col].map(freq)

for col in cat_cols + ['dow', 'month']:
    train_data = data[data['is_train'] == 1]
    if len(train_data) > 0:
        for target in ['price_p05', 'price_p95']:
            mean_target = train_data.groupby(col)[target].mean()
            data[f'{col}_target_{target}'] = data[col].map(mean_target)
            data[f'{col}_target_{target}'].fillna(data[f'{col}_target_{target}'].mean(), inplace=True)


label_cols = ['dow', 'month']
for col in label_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))


product_stats = data[data['is_train'] == 1].groupby('product_id').agg({
    'price_p05': ['mean', 'std', 'median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],
    'price_p95': ['mean', 'std', 'median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],
    'n_stores': ['mean', 'std', 'max', 'min']
}).reset_index()

product_stats.columns = ['product_id', 'price_p05_mean', 'price_p05_std', 'price_p05_median',
                         'price_p05_q25', 'price_p05_q75',
                         'price_p95_mean', 'price_p95_std', 'price_p95_median',
                         'price_p95_q25', 'price_p95_q75',
                         'n_stores_mean', 'n_stores_std', 'n_stores_max', 'n_stores_min']


product_stats['price_p05_cv'] = product_stats['price_p05_std'] / (product_stats['price_p05_mean'] + 1e-6)
product_stats['price_p95_cv'] = product_stats['price_p95_std'] / (product_stats['price_p95_mean'] + 1e-6)
product_stats['price_range_ratio'] = product_stats['price_p95_mean'] / (product_stats['price_p05_mean'] + 1e-6)


data = data.merge(product_stats, on='product_id', how='left')


data = data.sort_values(['product_id', 'dt'])

lag_windows = [1, 2, 3, 7, 14]
for lag in lag_windows:
    data[f'price_p05_lag_{lag}'] = data.groupby('product_id')['price_p05'].shift(lag)
    data[f'price_p95_lag_{lag}'] = data.groupby('product_id')['price_p95'].shift(lag)
    data[f'n_stores_lag_{lag}'] = data.groupby('product_id')['n_stores'].shift(lag)


roll_windows = [3, 7, 14]
for window in roll_windows:

    data[f'price_p05_roll_mean_{window}'] = data.groupby('product_id')['price_p05'].transform(
        lambda x: x.rolling(window, min_periods=1).mean())
    data[f'price_p95_roll_mean_{window}'] = data.groupby('product_id')['price_p95'].transform(
        lambda x: x.rolling(window, min_periods=1).mean())

    data[f'price_p05_roll_std_{window}'] = data.groupby('product_id')['price_p05'].transform(
        lambda x: x.rolling(window, min_periods=1).std())
    data[f'price_p95_roll_std_{window}'] = data.groupby('product_id')['price_p95'].transform(
        lambda x: x.rolling(window, min_periods=1).std())


data['price_midpoint'] = (data['price_p05'] + data['price_p95']) / 2
data['price_range'] = data['price_p95'] - data['price_p05']


data['cat_12_interaction'] = data['first_category_id'].astype(str) + '_' + data['second_category_id'].astype(str)


def detect_anomalies_simple(df, features, contamination=0.05):
    df_temp = df.copy()

    for col in features:
        if col in df_temp.columns:
            df_temp[col] = df_temp[col].fillna(df_temp[col].median())

    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )

    anomalies = iso_forest.fit_predict(df_temp[features])
    return (anomalies == -1).astype(int)

anomaly_features = [
    'n_stores', 'price_p05', 'price_p95', 'price_range',
    'precpt', 'avg_temperature', 'avg_humidity'
]

train_indices = data['is_train'] == 1
data['is_anomaly'] = 0

if train_indices.sum() > 0:
    data.loc[train_indices, 'is_anomaly'] = detect_anomalies_simple(
        data[train_indices],
        anomaly_features
    )


def cluster_products_simple(df, n_clusters=10):
    cluster_features = [
        'price_p05_mean', 'price_p95_mean',
        'price_p05_std', 'price_p95_std',
        'n_stores_mean'
    ]

    train_data = df[df['is_train'] == 1].copy()

    if len(train_data) > 0:
        for col in cluster_features:
            if col in train_data.columns:
                train_data[col] = train_data[col].fillna(train_data[col].median())

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(train_data[cluster_features])

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        train_data['product_cluster'] = kmeans.fit_predict(scaled_features)

        cluster_mapping = train_data[['product_id', 'product_cluster']].drop_duplicates()
        df = df.merge(cluster_mapping, on='product_id', how='left')

        return df, kmeans, scaler
    else:
        df['product_cluster'] = 0
        return df, None, None

data, kmeans_model, cluster_scaler = cluster_products_simple(data, n_clusters=12)


def apply_pca_simple(df, n_components=5):
    pca_features = [
        'price_p05_mean', 'price_p95_mean',
        'price_p05_std', 'price_p95_std',
        'n_stores', 'avg_temperature', 'avg_humidity'
    ]

    train_data = df[df['is_train'] == 1].copy()

    if len(train_data) > 0:
        for col in pca_features:
            if col in train_data.columns:
                train_data[col] = train_data[col].fillna(train_data[col].median())
            else:
                train_data[col] = 0

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(train_data[pca_features])

        pca = PCA(n_components=n_components, random_state=42)
        pca_components = pca.fit_transform(scaled_data)

        pca_columns = [f'pca_{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(pca_components, columns=pca_columns, index=train_data.index)

        df = df.merge(pca_df, left_index=True, right_index=True, how='left')

        for col in pca_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        return df, pca, scaler
    else:
        for i in range(n_components):
            df[f'pca_{i+1}'] = 0
        return df, None, None

data, pca_model, pca_scaler = apply_pca_simple(data, n_components=5)


data['day_sin'] = np.sin(2 * np.pi * data['day_of_year'] / 365)
data['day_cos'] = np.cos(2 * np.pi * data['day_of_year'] / 365)


train_data = data[data['is_train'] == 1].copy()
test_data = data[data['is_train'] == 0].copy()


def fill_missing_simple(df):
    df_filled = df.copy()
    numeric_cols = df_filled.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col not in ['price_p05', 'price_p95', 'is_train']:
            df_filled[col] = df_filled[col].fillna(df_filled[col].median())

    return df_filled

train_data = fill_missing_simple(train_data)
test_data = fill_missing_simple(test_data)


feature_columns = [

    'management_group_id', 'first_category_id', 'second_category_id', 'third_category_id',
    'dow', 'day_of_month', 'week_of_year', 'month', 'year', 'quarter',
    'is_weekend', 'is_month_start', 'is_month_end', 'day_of_year',


    'n_stores', 'holiday_flag', 'activity_flag',
    'precpt', 'avg_temperature', 'avg_humidity', 'avg_wind_level',

    'price_p05_mean', 'price_p05_std', 'price_p05_median',
    'price_p05_q25', 'price_p05_q75', 'price_p05_cv',
    'price_p95_mean', 'price_p95_std', 'price_p95_median',
    'price_p95_q25', 'price_p95_q75', 'price_p95_cv',
    'n_stores_mean', 'n_stores_std', 'n_stores_max', 'n_stores_min',
    'price_range_ratio',


    'management_group_id_freq', 'first_category_id_freq',


    'management_group_id_target_price_p05', 'management_group_id_target_price_p95',


    'is_anomaly',

    'product_cluster'
]


lag_features = [col for col in data.columns if 'lag' in col or 'roll_' in col]
feature_columns.extend([col for col in lag_features if col in train_data.columns and col in test_data.columns])


pca_features = [col for col in data.columns if 'pca_' in col]
feature_columns.extend([col for col in pca_features if col in train_data.columns and col in test_data.columns])


feature_columns.extend(['day_sin', 'day_cos'])

feature_columns = list(set(feature_columns))


available_features = [col for col in feature_columns if col in train_data.columns and col in test_data.columns]
print(f"Используем {len(available_features)} признаков")


X_train = train_data[available_features].copy()
X_test = test_data[available_features].copy()

y_train_lower = train_data['price_p05']
y_train_upper = train_data['price_p95']


def iou_metric(y_true_lower, y_true_upper, y_pred_lower, y_pred_upper, epsilon=1e-6):
    width_true = y_true_upper - y_true_lower + epsilon
    width_pred = y_pred_upper - y_pred_lower + epsilon

    intersection = np.maximum(0, np.minimum(y_true_upper, y_pred_upper) - np.maximum(y_true_lower, y_pred_lower))
    union = width_true + width_pred - intersection

    iou = intersection / union
    return np.mean(iou)


lower_model = lgb.LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.02,
    num_leaves=31,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

upper_model = lgb.LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.02,
    num_leaves=31,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)


X_train_split, X_val_split, y_lower_train, y_lower_val, y_upper_train, y_upper_val = train_test_split(
    X_train, y_train_lower, y_train_upper, test_size=0.2, random_state=42, shuffle=False
)


lower_model.fit(
    X_train_split, y_lower_train,
    eval_set=[(X_val_split, y_lower_val)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(100)]
)


upper_model.fit(
    X_train_split, y_upper_train,
    eval_set=[(X_val_split, y_upper_val)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(100)]
)


val_lower_pred = lower_model.predict(X_val_split)
val_upper_pred = upper_model.predict(X_val_split)

for i in range(len(val_lower_pred)):
    if val_lower_pred[i] > val_upper_pred[i]:
        val_lower_pred[i], val_upper_pred[i] = val_upper_pred[i], val_lower_pred[i]


val_iou = iou_metric(
    y_lower_val.values,
    y_upper_val.values,
    val_lower_pred,
    val_upper_pred
)
print(f"Validation IoU score: {val_iou:.4f}")


cb_lower_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    random_seed=42,
    verbose=0
)


cb_upper_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    random_seed=42,
    verbose=0
)


cb_lower_model.fit(X_train, y_train_lower)
cb_upper_model.fit(X_train, y_train_upper)


test_lower_lgb = lower_model.predict(X_test)
test_upper_lgb = upper_model.predict(X_test)


test_lower_cb = cb_lower_model.predict(X_test)
test_upper_cb = cb_upper_model.predict(X_test)

test_lower_final = 0.7 * test_lower_lgb + 0.3 * test_lower_cb
test_upper_final = 0.7 * test_upper_lgb + 0.3 * test_upper_cb

for i in range(len(test_lower_final)):
    if test_lower_final[i] > test_upper_final[i]:
        test_lower_final[i], test_upper_final[i] = test_upper_final[i], test_lower_final[i]
    min_width = 0.005 * (test_lower_final[i] + test_upper_final[i]) / 2
    if test_upper_final[i] - test_lower_final[i] < min_width:
        mid = (test_lower_final[i] + test_upper_final[i]) / 2
        test_lower_final[i] = mid - min_width / 2
        test_upper_final[i] = mid + min_width / 2


submission = sample_submission.copy()
submission['price_p05'] = test_lower_final
submission['price_p95'] = test_upper_final
submission.to_csv('results/sample_submission.csv', index=False)




negative_intervals = np.sum(test_lower_final > test_upper_final)
if negative_intervals > 0:
    print(f"\nИсправлено {negative_intervals} интервалов с нижней границей > верхней!")


try:
    feature_importance_lower = pd.DataFrame({
        'feature': available_features,
        'importance': lower_model.feature_importances_
    }).sort_values('importance', ascending=False)

    feature_importance_upper = pd.DataFrame({
        'feature': available_features,
        'importance': upper_model.feature_importances_
    }).sort_values('importance', ascending=False)

except Exception as e:
    print(f"\nНе удалось рассчитать важность признаков: {e}")

print("\nГотово! Сабмишен создан.")