import pandas as pd
import time
import joblib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime as dt

from data_save import get_data_save

PREDICT_CONFIG_PATH = "predict_config.txt"
MODEL_PATH_1 = "models/m_v_1_model.joblib"
MODEL_PATH_2 = "models/m_v_2_model.joblib"
MODEL_PATH_3 = "models/m_v_3_model.joblib"
MODEL_PATH_4 = "models/m_v_4_model.joblib"

LIMITS = {
    "CO": 79.5,
    "SO2": 34.58,
    "NO": 6.772,
    "NO2": 19.1388,
}

STEP_SECONDS = 3  # шаг твоих данных (3 сек)


# --------------- 2) Вспомогательные функции ---------------


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))

def _level_zone_1to4(value: float, x_max: float) -> int:
    """
    Зона риска только по уровню:
    1: <=0.7*max
    2: (0.7..0.9]*max
    3: (0.9..1.0)*max
    4: >= max
    """
    if value >= x_max:
        return 4
    if value >= 0.9 * x_max:
        return 3
    if value >= 0.7 * x_max:
        return 2
    return 1

def _level_score_0to3(value: float, x_max: float) -> int:
    """Баллы уровня: 0..3 (как в инструкции)."""
    return _level_zone_1to4(value, x_max) - 1  # 1..4 -> 0..3

def _get_deltas(history: List[float]) -> Tuple[float, float]:
    """
    d1: изменение за 1 шаг (3 сек)
    d5: изменение за 5 шагов (15 сек)
    """
    if not history:
        return 0.0, 0.0
    curr = float(history[-1])
    prev1 = float(history[-2]) if len(history) >= 2 else curr
    prev5 = float(history[-6]) if len(history) >= 6 else prev1
    return curr - prev1, curr - prev5

def _speed_score_0to2(d1: float, d5: float, x_max: float) -> int:
    """
    Баллы скорости: 0..2
    - 2: |d1| >= 5%*max (быстрый шаг за 3 сек)
    - 1: |d1| < 5%*max и |d5| >= 5%*max (дрейф за 15 сек)
    - 0: иначе
    """
    thr = 0.05 * x_max
    if abs(d1) >= thr:
        return 2
    if abs(d1) < thr and abs(d5) >= thr:
        return 1
    return 0

def _oscillation_index(history: List[float]) -> float:
    """
    Индекс колебаний 0..1 по частоте смены знака производной.
    0 = монотонно/почти монотонно
    1 = каждую точку меняет знак (пила)
    """
    if len(history) < 8:
        return 0.0

    y = np.array(history[-20:], dtype=float)  # последние ~60 сек
    dy = np.diff(y)
    # убираем почти нулевые изменения (шум)
    eps = max(1e-9, 0.002 * (np.nanmax(y) - np.nanmin(y) + 1e-9))
    s = np.sign(np.where(np.abs(dy) < eps, 0.0, dy))

    # берём только ненулевые знаки
    s2 = s[s != 0]
    if len(s2) < 4:
        return 0.0

    # доля смен знака
    changes = np.sum(s2[1:] * s2[:-1] < 0)
    return float(changes) / float(len(s2) - 1)

def _acceleration_index(history: List[float]) -> float:
    """
    Ускорение (вторая производная) по последним точкам.
    Возвращает оценку "насколько ускоряется рост/падение" в долях от max в секунду^2.
    """
    if len(history) < 4:
        return 0.0
    y = np.array(history[-6:], dtype=float)  # последние 18 сек
    dy = np.diff(y) / STEP_SECONDS
    ddy = np.diff(dy) / STEP_SECONDS
    # берём максимум по модулю
    return float(np.nanmax(np.abs(ddy)))

# --------------- 3) Оценка по компоненту ---------------

def assess_component(
    name: str,
    history_with_pred: List[float],
) -> Dict[str, Any]:
    """
    history_with_pred: история + (опционально) прогноз, последняя точка считается текущей для оценки
    """
    x_max = LIMITS[name]
    curr = float(history_with_pred[-1])

    d1, d5 = _get_deltas(history_with_pred)
    lvl_score = _level_score_0to3(curr, x_max)
    spd_score = _speed_score_0to2(d1, d5, x_max)
    total_score = lvl_score + spd_score  # 0..5

    # Зона риска 1..4 по суммарным баллам (простая и понятная шкала)
    # 0-1 -> зона 1 (норма)
    # 2-3 -> зона 2 (предупреждение)
    # 4   -> зона 3 (предавария)
    # 5   -> зона 4 (авария/резкий выход)
    if total_score <= 1:
        risk_zone = 1
    elif total_score <= 3:
        risk_zone = 2
    elif total_score == 4:
        risk_zone = 3
    else:
        risk_zone = 4

    # Характер движения (коротко и программно)
    osc = _oscillation_index(history_with_pred)
    acc = _acceleration_index(history_with_pred)  # г/с^3 (условно), без нормировки

    movement = "stable"
    if spd_score == 2 and d1 > 0:
        movement = "spike_up"
    elif spd_score == 2 and d1 < 0:
        movement = "spike_down"
    elif spd_score == 1 and d5 > 0:
        movement = "drift_up"
    elif spd_score == 1 and d5 < 0:
        movement = "drift_down"

    if osc >= 0.45:
        movement = "unstable_oscillations"  # важно для статуса "резкие колебания"

    return {
        "name": name,
        "value": curr,
        "max": x_max,
        "d1_3s": float(d1),
        "d5_15s": float(d5),
        "level_score": int(lvl_score),
        "speed_score": int(spd_score),
        "total_score": int(total_score),
        "risk_zone": int(risk_zone),
        "movement": movement,
        "osc_index_0to1": float(osc),
        "acc_abs": float(acc),
    }

# --------------- 4) Таблица статусов (id=0..∞) ---------------

STATUS_TABLE: List[Dict[str, Any]] = [
    {
        "id": 0,
        "name": "Норма",
        "triggers": [
            "Все компоненты risk_zone=1",
            "Нет колебаний: osc_index < 0.45 у всех компонентов",
        ],
        "description": "Выбросы в норме, опасных трендов нет.",
        "actions": [
            "Режим не менять.",
            "Стандартный мониторинг.",
        ],
    },
    {
        "id": 1,
        "name": "Предупреждение: умеренное ухудшение",
        "triggers": [
            "Есть компоненты risk_zone=2, но нет risk_zone=4",
            "ИЛИ медленный дрейф вверх (movement=drift_up) по любому компоненту",
        ],
        "description": "Есть рост/дрейф выбросов, риск выхода в предаварию при сохранении тренда.",
        "actions": [
            "Усилить наблюдение 5–10 минут.",
            "Проверить равномерность подачи воздуха/газа, работу тяги.",
            "Подготовить корректировку режима по характеру горения (см. ниже).",
        ],
    },
    {
        "id": 2,
        "name": "Недожёг / недостаток воздуха (CO↑ при NO↓)",
        "triggers": [
            "CO risk_zone>=2 и CO d5_15s > 0",
            "NO risk_zone=1 или NO d5_15s < 0",
        ],
        "description": "Признаки недостатка воздуха/смесеобразования: CO растёт, NO не растёт (или падает).",
        "actions": [
            "Увеличить подачу воздуха (ориентир +2…10%).",
            "Проверить смесеобразование/распределение воздуха по простенкам.",
            "Если CO близко к порогу: снизить нагрузку (подачу газа).",
        ],
    },
    {
        "id": 3,
        "name": "Избыток воздуха / жёсткое горение (NOx↑ при CO↓)",
        "triggers": [
            "NO или NO2 risk_zone>=2 и (NO d5_15s > 0 или NO2 d5_15s > 0)",
            "CO risk_zone=1 или CO d5_15s < 0",
        ],
        "description": "Признаки избытка воздуха/высокой температуры: растут NO/NO2 при низком/падающем CO.",
        "actions": [
            "Уменьшить подачу воздуха (ориентир -2…10%).",
            "Перераспределить газ/воздух для выравнивания температур.",
            "Следить, чтобы CO не ушёл в предупреждение (зона 2).",
        ],
    },
    {
        "id": 4,
        "name": "Высокий SO2 (состав топлива/просос сырого газа/герметичность)",
        "triggers": [
            "SO2 risk_zone>=2 и SO2 d5_15s > 0",
            "ИЛИ SO2 movement=spike_up",
        ],
        "description": "Рост SO2: возможен рост сернистости топлива/газа или попадание/просос сырого коксового газа.",
        "actions": [
            "Проверить качество газа/топлива (сернистость), режим очистки.",
            "Проверить герметичность, исключить прососы сырого коксового газа.",
            "При быстром росте: снизить нагрузку и стабилизировать горение.",
        ],
    },
    {
        "id": 5,
        "name": "Нестабильность / резкие колебания",
        "triggers": [
            "По любому компоненту movement=unstable_oscillations",
            "ИЛИ одновременные скачки (speed_score=2) у 2+ компонентов",
        ],
        "description": "Режим нестабилен: вероятны сбои регулирования/датчиков/исполнительных механизмов.",
        "actions": [
            "Проверить датчики, качество сигнала, исполнительные механизмы.",
            "Проверить стабильность подачи газа/воздуха (нет ли рывков).",
            "Перевести регулирование в более стабильный режим (сглаживание/ограничение изменений).",
        ],
    },
    {
        "id": 6,
        "name": "Авария: превышение/резкий выход к пределу",
        "triggers": [
            "Любой компонент risk_zone=4",
            "ИЛИ 2+ компонентов risk_zone=3 одновременно",
        ],
        "description": "Аварийная ситуация по выбросам: превышение порогов или быстрый выход к ним.",
        "actions": [
            "Немедленно снизить нагрузку (подачу газа).",
            "Корректировать воздух по характеру: если CO↑ -> воздуха больше; если NOx↑ -> воздуха меньше.",
            "Проверить тягу/герметичность/газоочистку.",
            "Действовать по аварийному регламенту предприятия.",
        ],
    },
]

def choose_status(component_infos: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Возвращает выбранный статус (словарь из STATUS_TABLE) + пояснение почему.
    Порядок важен: авария -> колебания -> спец-ситуации -> предупреждение -> норма
    """
    zones = {k: v["risk_zone"] for k, v in component_infos.items()}
    max_zone = max(zones.values())
    count_zone3 = sum(1 for z in zones.values() if z == 3)

    # 6) авария
    if max_zone == 4 or count_zone3 >= 2:
        st = next(s for s in STATUS_TABLE if s["id"] == 6)
        return {**st, "why": {"max_zone": max_zone, "count_zone3": count_zone3, "zones": zones}}

    # 5) нестабильность/колебания
    if any(v["movement"] == "unstable_oscillations" for v in component_infos.values()):
        st = next(s for s in STATUS_TABLE if s["id"] == 5)
        return {**st, "why": {"zones": zones, "osc": {k: v["osc_index_0to1"] for k, v in component_infos.items()}}}

    # 4) SO2
    if component_infos["SO2"]["risk_zone"] >= 2 and (component_infos["SO2"]["d5_15s"] > 0 or component_infos["SO2"]["movement"] == "spike_up"):
        st = next(s for s in STATUS_TABLE if s["id"] == 4)
        return {**st, "why": {"SO2": component_infos["SO2"], "zones": zones}}

    # 2) недостаток воздуха (CO↑, NO↓)
    if (component_infos["CO"]["risk_zone"] >= 2 and component_infos["CO"]["d5_15s"] > 0) and \
       (component_infos["NO"]["risk_zone"] == 1 or component_infos["NO"]["d5_15s"] < 0):
        st = next(s for s in STATUS_TABLE if s["id"] == 2)
        return {**st, "why": {"CO": component_infos["CO"], "NO": component_infos["NO"], "zones": zones}}

    # 3) избыток воздуха (NOx↑, CO↓)
    if (component_infos["NO"]["risk_zone"] >= 2 or component_infos["NO2"]["risk_zone"] >= 2) and \
       ((component_infos["NO"]["d5_15s"] > 0) or (component_infos["NO2"]["d5_15s"] > 0)) and \
       (component_infos["CO"]["risk_zone"] == 1 or component_infos["CO"]["d5_15s"] < 0):
        st = next(s for s in STATUS_TABLE if s["id"] == 3)
        return {**st, "why": {"NO": component_infos["NO"], "NO2": component_infos["NO2"], "CO": component_infos["CO"], "zones": zones}}

    # 1) предупреждение
    if max_zone == 2 or any(v["movement"] in ("drift_up", "spike_up") for v in component_infos.values()):
        st = next(s for s in STATUS_TABLE if s["id"] == 1)
        return {**st, "why": {"zones": zones, "movements": {k: v["movement"] for k, v in component_infos.items()}}}

    # 0) норма
    st = next(s for s in STATUS_TABLE if s["id"] == 0)
    return {**st, "why": {"zones": zones}}

# --------------- 5) Интеграция: собрать итог для API ---------------

def build_situation_assessment(
    history_dict: Dict[str, List[float]],
    pred_dict: Dict[str, float],
) -> Dict[str, Any]:
    """
    history_dict: {"CO":[...], "SO2":[...], "NO":[...], "NO2":[...]}
    pred_dict: {"CO":pred, "SO2":pred, "NO":pred, "NO2":pred}
    """

    vals = {}

    component_infos = {}
    for name in ["CO", "SO2", "NO", "NO2"]:
        hist = list(history_dict[name])
        hist_with_pred = hist + [float(pred_dict[name])]
        component_infos[name] = assess_component(name, hist_with_pred)
        vals[name] = component_infos[name]["value"]

    status = choose_status(component_infos)

    # удобные поля для фронта
    risk_zone = {k: v["risk_zone"] for k, v in component_infos.items()}
    return {
        "risk_zone": risk_zone,                # 1..4
        "component_details": component_infos,  # всё для дебага/объяснимости
        "overall_status": status,              # таблица + why
        "overall_status_id": int(status["id"]),
        "vals": vals
    }



def load_data(): #имитация
    # try:
    #     load_data.count += 1
    # except:
    #     load_data.count = 0

    # try:
    #     df1 = load_data.df1
    # except: 
    #     load_data.df1 = pd.read_csv(DATA_PATH_1)
    #     df1 = load_data.df1

    # # df1 = pd.read_csv(DATA_PATH_1)
    # # df2 = pd.read_csv(DATA_PATH_2)
    # # df3 = pd.read_csv(DATA_PATH_3)

    # # print((100*load_data.count), (100*(load_data.count+1)))

    # # return df1[:100]
    return get_data_save()

def load_model(config):
    try:
        return load_model.models
    except:
        models = {}
        for col, path in zip(config["target_cols"], [MODEL_PATH_1, MODEL_PATH_2, MODEL_PATH_3, MODEL_PATH_4]):
            models[col] = joblib.load(path)
        load_model.models = models

        return models

def load_predict_config(path: str = PREDICT_CONFIG_PATH) -> dict:
    """Загрузка конфигурации предсказаний (n_s_pred, features_list) из txt (JSON)."""
    try:
        return load_predict_config.config
    except:
        if not os.path.exists(path):
            config = {"n_s_pred": 30, "features_list": None}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            return config

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
def make_features_for_predict(last_rows_raw: pd.DataFrame, target_col: str, features_list) -> pd.DataFrame:
    """Подготовка одной строки фич для предикта на основе сырых данных.

    last_rows_raw: DataFrame с последними замерами, минимум ['datetime', target_col]
    """
    if 'datetime' not in last_rows_raw.columns:
        raise ValueError("last_rows_raw должен содержать колонку 'datetime'")
    if target_col not in last_rows_raw.columns:
        raise ValueError(f"last_rows_raw должен содержать колонку '{target_col}'")
    if len(last_rows_raw) < 100:
        raise ValueError("Нужно минимум 100 строк для лагов до lag_100")

    dt = pd.to_datetime(last_rows_raw['datetime'].iloc[-1])

    current_state = {
        'lag_0': last_rows_raw[target_col].iloc[-1],
        'lag_5': last_rows_raw[target_col].iloc[-5],
        'lag_20': last_rows_raw[target_col].iloc[-20],
        'lag_100': last_rows_raw[target_col].iloc[-100],
        'hour': dt.hour,
        'rolling_mean': last_rows_raw[target_col].tail(20).mean(),
    }

    return pd.DataFrame([current_state]).reindex(columns=features_list)


def predict_n_seconds_ahead(
    model,
    last_rows_raw: pd.DataFrame,
    target_col: str,
    n_s_pred: int,
    features_list=None,
):
    """Универсальный предикт на n_s_pred секунд вперёд.

    model: обученная модель
    last_rows_raw: сырые последние данные с колонками ['datetime', target_col]
    n_s_pred: горизонт прогноза в секундах
    features_list: список фич в том же порядке, что при обучении.
                   Если None и у модели есть feature_names_in_ — берём их.
    """
    if features_list is None:
        if hasattr(model, "feature_names_in_"):
            features_list = list(model.feature_names_in_)
        else:
            raise ValueError("Передай features_list (например, X_train.columns.tolist())")

    X_input = make_features_for_predict(last_rows_raw, target_col, features_list)
    pred = model.predict(X_input)[0]

    dt = pd.to_datetime(last_rows_raw['datetime'].iloc[-1])
    prediction_time = dt + pd.Timedelta(seconds=n_s_pred)

    return pred, prediction_time
    
def predict_targets(config, models):
    try:
        predict_targets.count += 1
    except:
        predict_targets.count = 0

    maxx = {"CO": 79.5,
            "SO2": 34.58,
            "NO": 6.772,
            "NO2": 19.1388}
    
    res = {
        "metrics": 
        {
            "NO2": None,
            "NO": None,
            "CO": None,
            "SO2": None
        },

        "risk_zone": 
        {
            "NO2": 0,
            "NO": 0,
            "CO": 0,
            "SO2": 0
        },

        "history":
        {
            "NO2": None,
            "NO": None,
            "CO": None,
            "SO2": None
        },
        "for_admin":{
            "time": (str(dt.now()).split()[1]).split(".")[0],
            "candle": 101 + predict_targets.count

        }
    }

    history = load_data()

    if len(history) < 100:
        raise ValueError(f"История содержит только {len(history)} строк, нужно минимум 100")

    pred_dict = {}
    history_dict = {}

    for col, col_name in zip(config["target_cols"], config["target_cols"]):
        pred_val, pred_time = predict_n_seconds_ahead(model= models[col], 
                         last_rows_raw= history, 
                         target_col= col,
                         features_list= config["features_list"],
                         n_s_pred=int(config["n_s_pred"]))

        
        pred_dict[col_name] = float(pred_val)

        h = list(history[col])  # как у тебя
        history_dict[col_name] = h

        res["metrics"][col_name] = float(pred_val) # , str(pred_time))
        res["history"][col_name] = h

    assessment = build_situation_assessment(history_dict, pred_dict)

    res["risk_zone"] = assessment["risk_zone"]  # 1..4
    res["vals"] = assessment["vals"]
    res["overall_status_id"] = assessment["overall_status_id"]
    res["overall_status"] = assessment["overall_status"]
    res["component_details"] = assessment["component_details"]  # опционально, но полезно

    return res

    # for col, col_name in zip(config["target_cols"], config["target_cols"]):
    #     resolt, t_time = list(predict_n_seconds_ahead(model= models[col], 
    #                     last_rows_raw= history, 
    #                     target_col= col,
    #                     features_list= config["features_list"],
    #                     n_s_pred=int(config["n_s_pred"])))
        
    #     # Метрики
    #     res["metrics"][col_name] = float(resolt), str(t_time)

    #     # Зона риска
    #     col_maxx = maxx[col_name]
    #     if resolt >= col_maxx:
    #         res["risk_zone"][col_name] = 3
    #     elif col_maxx*0.9 <= resolt < col_maxx:
    #         res["risk_zone"][col_name] = 2
    #     elif col_maxx*0.7 <= resolt < col_maxx*0.9:
    #         res["risk_zone"][col_name] = 1
    #     else:
    #         res["risk_zone"][col_name] = 0

        
    #     # История
    #     h = list(history[col])
    #     res["history"][col_name] = h

    #     # d_temp_dt = np.diff(h) / np.diff(pd.to_datetime(history["datetime"]))
    #     threshold = 0.01  # Порог в г/с²
    #     derivative = np.gradient(resolt, 3)
    #     print(derivative)
    #     # significant_changes = np.where(np.abs(derivative) > threshold)[0]

        
    #     # for idx in [94, 95]:
    #     #     print(f"Индекс {idx}: производная = {derivative[idx]:.6f} г/с²")
        

    # return res
        
def main():
    config = load_predict_config(PREDICT_CONFIG_PATH)

    for i in range(10):
        s = time.time()
        print(predict_targets(config, load_model(config)))
        end = time.time()
        prod = end - s
        print(f"Worked time: {prod}")
        print("-----")
        time.sleep(3)

# main()