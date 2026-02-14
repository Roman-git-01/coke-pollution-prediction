import joblib
import json
import os
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

PREDICT_CONFIG_PATH = "predict_config.txt"
MODEL_PATH_1 = r"C:\Users\Roman\E21_26\Proj\models\m_v_1_model.joblib"
MODEL_PATH_2 = r"C:\Users\Roman\E21_26\Proj\models\m_v_2_model.joblib"
MODEL_PATH_3 = r"C:\Users\Roman\E21_26\Proj\models\m_v_3_model.joblib"
MODEL_PATH_4 = r"C:\Users\Roman\E21_26\Proj\models\m_v_4_model.joblib"
DATA_PATH_1 = r"C:\Users\Roman\E21_26\Proj\DATA_csv\tags_1.csv"
DATA_PATH_2 = r"C:\Users\Roman\E21_26\Proj\DATA_csv\tags_2.csv"

def load_data_df():
    df1 = pd.read_csv(DATA_PATH_1)
    df2 = pd.read_csv(DATA_PATH_2)

    return df1, df2
df1, df2 = load_data_df()


def save_models(models):
    # for key, path in zip(models, [MODEL_PATH_1, MODEL_PATH_2, MODEL_PATH_3, MODEL_PATH_4]):
    #     joblib.dump(models[key], path)

    print("success save")
    

def load_predict_config(path: str = PREDICT_CONFIG_PATH) -> dict:
    """Загрузка конфигурации предсказаний (n_s_pred, features_list) из txt (JSON)."""
    if not os.path.exists(path):
        config = {"n_s_pred": 30, "features_list": None}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return config

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_predict_config(n_s_pred: int, features_list=None, path: str = PREDICT_CONFIG_PATH) -> None:
    # """Сохранение конфигурации предикта в txt (JSON)."""
    # config = {
    #     "n_s_pred": int(n_s_pred),
    #     "features_list": list(features_list) if features_list is not None else None,
    # }
    # with open(path, "w", encoding="utf-8") as f:
    #     json.dump(config, f, ensure_ascii=False, indent=2)
    print("")


def prepare_data_n_seconds(dfs, target_col='m NO, г/с', n_s_pred: int = 30, step_seconds: int = 3):
    """Подготовка данных под прогноз на n_s_pred секунд вперёд.

    dfs: список датафреймов по дням
    target_col: таргетная колонка
    n_s_pred: горизонт прогноза в секундах (должен быть кратен step_seconds)
    step_seconds: шаг измерений (у тебя 3 секунды)
    """
    processed_days = []

    if n_s_pred % step_seconds != 0:
        raise ValueError(f"n_s_pred={n_s_pred} должен быть кратен step_seconds={step_seconds}")

    forecast_steps = n_s_pred // step_seconds
    target_name = f"target_{n_s_pred}s"

    for df in dfs:
        df['datetime'] = df['datetime'].str.split('.').str[0]
        df = df.drop('Unnamed: 0', axis=1)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df[["datetime", target_col]]
        temp_df = df.copy().sort_values('datetime')

        # Таргет: значение через forecast_steps строк (n_s_pred секунд)
        temp_df[target_name] = temp_df[target_col].shift(-forecast_steps)

        # Признаки, как и раньше
        temp_df['hour'] = temp_df['datetime'].dt.hour

        for lag in [0, 5, 20, 100]:  # lag 0 — текущее значение
            temp_df[f'lag_{lag}'] = temp_df[target_col].shift(lag)

        temp_df['rolling_mean'] = temp_df[target_col].rolling(window=20).mean()

        # Удаляем строки без таргета и лагов
        temp_df.dropna(inplace=True)
        processed_days.append(temp_df)

    # Склейка и разделение по времени
    full_df = pd.concat(processed_days)
    split_date = full_df['datetime'].max() - pd.Timedelta(days=3)

    train_df = full_df[full_df['datetime'] <= split_date]
    test_df = full_df[full_df['datetime'] > split_date]

    features = [
        col for col in train_df.columns
        if col not in ['datetime', target_name, target_col]
    ]

    return (
        train_df[features],
        train_df[target_name],
        test_df[features],
        test_df[target_name],
        features,
    )


def train_model(X_train, y_train, X_test, y_test):
    # Используем метод 'hist' для работы с большим объемом данных
    model = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',   # Обязательно для скорости на 500к+ строк
        device='cuda',        # Раскомментируй, если есть GPU NVIDIA
        early_stopping_rounds=50,
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100
    )
    return model


def validate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    
    metrics = {
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "MSE": mean_squared_error(y_test, preds),
        "R2": r2_score(y_test, preds)
    }
    
    print("\n--- Метрики ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    # Важность признаков
    importance = pd.Series(model.feature_importances_, index=X_test.columns)
    print("\nТоп признаков:\n", importance.sort_values(ascending=False).head(5))
    
    return preds

def train_val_model_n_seconds(cols: list, n_s_pred: int | None = None):
    """Обучение и валидация моделей под горизонт n_s_pred секунд.

    cols: список таргетных колонок
    n_s_pred: горизонт прогноза в секундах. Если None — берём из конфигурации.
    """
    models = {}

    config = load_predict_config()
    if n_s_pred is None:
        n_s_pred = config.get("n_s_pred", 30)

    for col in cols:
        print(f"Для {col}:\n-----------")
        # 1. Готовим данные под заданный горизонт
        X_train, y_train, X_test, y_test, features = prepare_data_n_seconds(
            [df1, df2], target_col=col, n_s_pred=n_s_pred
        )

        # 2. Обучаем
        booster = train_model(X_train, y_train, X_test, y_test)

        # 3. Валидируем
        _ = validate_model(booster, X_test, y_test)
        models[col] = booster

        # Обновляем конфиг: сохраняем последний использованный горизонт и список фич
    
    # save_predict_config(n_s_pred=n_s_pred, features_list=features)
    return models

def main():
    config = load_predict_config(PREDICT_CONFIG_PATH)
    models = train_val_model_n_seconds(cols=config["target_cols"], n_s_pred=30)

    save_models(models)
main()