import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib

# ------------------------------
# 1. Загрузка и предобработка данных
# ------------------------------
df = pd.read_csv("X_y_tags_2.csv")
df = df.sort_values('datetime').reset_index(drop=True)

feature_cols = [
    'raz_avg', 'raz_max', 'raz_min',
    'kal_avg', 'kal_max', 'kal_min',
    'f_avg', 'f_max', 'f_min',
    'P_avg', 'P_max', 'P_min',
    't_avg', 't_max', 't_min',
    't_avg_b_x', 't_max_b_x', 't_min_b_x',
    'f_avg2', 'f_max2', 'f_min2',
    'P_avg2', 'P_max2', 'P_min2',
    't_avg2', 't_max2', 't_min2',
    'raz_avg2', 'raz_max2', 'raz_min2',
    't_avg_b_y', 't_max_b_y', 't_min_b_y'
]

pollutants = ['NO', 'NO2', 'SO2', 'CO']

pollutant_stats = ['NO_max', 'NO2_max', 'SO2_max', 'CO_max',
                   'NO_min', 'NO2_min', 'SO2_min', 'CO_min']

all_features_for_lag = feature_cols # + pollutants + pollutant_stats

missing = [col for col in all_features_for_lag if col not in df.columns]
if missing:
    print(f"Внимание! Отсутствуют колонки: {missing}")
    all_features_for_lag = [col for col in all_features_for_lag if col in df.columns]

# ------------------------------
# 2. Временные признаки
# ------------------------------
if 'hour' not in df.columns:
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

time_features = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']

# ------------------------------
# 3. Лаги и скользящие статистики
# ------------------------------
window_size = 10      # количество прошлых минут
forecast_horizon = 30 # прогноз на 30 минут вперёд

for col in all_features_for_lag:
    for lag in range(1, window_size + 1):
        df[f'{col}_lag_{lag}'] = df[col].shift(lag)

for col in all_features_for_lag:
    df[f'{col}_rolling_mean_5'] = df[col].rolling(window=5).mean().shift(1)
    df[f'{col}_rolling_mean_10'] = df[col].rolling(window=10).mean().shift(1)

# ------------------------------
# 4. Целевые переменные
# ------------------------------
for pol in pollutants:
    df[f'target_{pol}'] = df[pol].shift(-forecast_horizon)

df.dropna(inplace=True)

# ------------------------------
# 5. Матрицы X и y
# ------------------------------
lag_columns = [col for col in df.columns if 'lag_' in col or 'rolling_' in col]
X = df[time_features + lag_columns].copy()
y = df[[f'target_{pol}' for pol in pollutants]].copy()
y.columns = pollutants

print(f"Размер X: {X.shape}")
print(f"Размер y: {y.shape}")

# ------------------------------
# 6. Разделение на train / test
# ------------------------------
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Обучающая выборка: {X_train.shape[0]} строк")
print(f"Тестовая выборка: {X_test.shape[0]} строк")

# ------------------------------
# 7. Обучение XGBoost для каждого загрязнителя
# ------------------------------
models = {}
results = {}

# Параметры XGBoost (можно настраивать)
xgb_params = {
    'n_estimators': 200,
    'max_depth': 5,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.5,
    'random_state': 42,
    'n_jobs': -1
}

for pol in pollutants:
    print(f"\n--- Обучение XGBoost для {pol} ---")
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train[pol])
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train[pol], y_pred_train)
    test_mae = mean_absolute_error(y_test[pol], y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train[pol], y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test[pol], y_pred_test))
    
    print(f"Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
    print(f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
    
    models[pol] = model
    results[pol] = {
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse
    }
    
    joblib.dump(model, f'models/xgb_model_{pol}_h{forecast_horizon}.pkl')
    print(f"Модель сохранена как xgb_model_{pol}_h{forecast_horizon}.pkl")

# ------------------------------
# 8. Сравнение с наивным прогнозом (константное среднее)
# ------------------------------
print("\n--- Сравнение с константным прогнозом (среднее по обучению) ---")
naive_mae_total = 0
for pol in pollutants:
    naive_pred = np.full_like(y_test[pol], y_train[pol].mean())
    naive_mae = mean_absolute_error(y_test[pol], naive_pred)
    print(f"{pol} naive (mean) MAE: {naive_mae:.4f}")
    naive_mae_total += naive_mae
print(f"Средний naive MAE: {naive_mae_total/len(pollutants):.4f}")

# ------------------------------
# 9. Визуализация для одного загрязнителя (SO2)
# ------------------------------
pol_plot = 'SO2'
plt.figure(figsize=(12, 6))
plt.plot(y_test[pol_plot].values[:500], label='Actual', alpha=0.7)
plt.plot(models[pol_plot].predict(X_test)[:500], label='Predicted XGBoost', alpha=0.7)
plt.plot([y_train[pol_plot].mean()]*500, label='Naive (mean)', alpha=0.5, linestyle='--')
plt.legend()
plt.title(f'Прогноз {pol_plot} на {forecast_horizon} минут вперёд (первые 500 минут теста)')
plt.xlabel('Минуты')
plt.ylabel(pol_plot)
plt.show()

# ------------------------------
# 10. Анализ важности признаков для SO2
# ------------------------------
if 'SO2' in models:
    importances = models['SO2'].feature_importances_
    feature_names = X.columns
    indices = np.argsort(importances)[::-1][:20]
    plt.figure(figsize=(10, 8))
    plt.title('Топ-20 важных признаков для SO2')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# ------------------------------
# 11. Сводная таблица
# ------------------------------
summary = pd.DataFrame(results).T
print("\nСводка результатов по всем загрязнителям:")
print(summary[['train_mae', 'test_mae', 'train_rmse', 'test_rmse']])