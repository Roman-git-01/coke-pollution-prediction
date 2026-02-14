from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, Any
import logging
import pandas as pd
from datetime import datetime as dt
from fastapi.middleware.cors import CORSMiddleware

from data_save import get_data_save, set_data_save
from working_handler import load_predict_config, load_model, predict_targets

# Глобальная переменная для хранения результата
result = {
    "metrics": None,
    "log": [None],
    "vals": None,
    "history": None,
    "overall_status": {
        "description": None,
        "actions": None
    }
}


class GasData(BaseModel):
    datetime: str
    NO: float
    NO2: float
    SO2: float
    CO: float

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загружаем конфигурацию при старте приложения
try:
    config = load_predict_config()
    logger.info("   Конфигурация успешно загружена +")
except Exception as e:
    logger.error(f"Ошибка загрузки конфигурации: {e}")
    raise

# Загружаем модели при старте приложения (один раз!)
try:
    models = load_model(config)
    logger.info("   Модели успешно загружены +")
except Exception as e:
    logger.error(f"Ошибка загрузки моделей: {e}")
    raise

try:
    result = predict_targets(config=config, models=models)
    logger.info("   Первичный resolt вычислен +")
except Exception as e:
    logger.error(f"Ошибка вычисления resolt: {e}")
    raise

app = FastAPI(docs_url=None, redoc_url=None)

origins = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://localhost:5000",
    "http://localhost:3000",  # Фронтенд на другом порту
    "null",  # Для файлов, открытых напрямую из файловой системы
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "Accept",
        "Origin",
        "User-Agent",
        "DNT",
        "Cache-Control",
        "X-Mx-ReqToken",
        "Keep-Alive",
        "If-Modified-Since",
        "Pragma"
    ],
    expose_headers=["*"],
    max_age=600,  # Кэширование preflight-запросов на 10 минут
)


def update_data(new_df):

    df_main = get_data_save()
    df_main = pd.concat([df_main, new_df])
    df_main = df_main.iloc[1:]  # Удаляем первую строку
    return df_main

def update_prediction():
    """Функция для обновления глобального результата"""
    global result
    try:
        # s = time.time()
        result = predict_targets(config=config, models=models)
        # end = time.time()
        # prod = end - s
        # print(f"Worked time: {prod}")
        logger.info("Результат успешно обновлён после добавления новых данных!")
    except Exception as e:
        logger.error(f"Ошибка при обновлении предсказания: {e}")
        raise

@app.post("/data")
async def post_data(data: dict):
    """
    POST‑метод для добавления одной строки в DataFrame
    с последующим обновлением результата предсказания
    """
    global result
    try:
        logging.info(f"Получены данные: {data}")

        # Преобразуем данные в DataFrame и добавляем к существующему
        new_df = pd.DataFrame([data])
        updated_df = update_data(new_df)
        set_data_save(updated_df)  # Сохраняем обновлённый DataFrame

        # Обновляем глобальный результат — вычисляем только один раз
        update_prediction()

        return {
            "status": "success",
            "datetime": dt.now(),
            "message": "Данные успешно добавлены и обработаны",
            "prediction_result": result,
        }
    except Exception as e:
        logging.error(f"Ошибка при обработке данных: {str(e)} {data.dict()}")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обработке данных: {str(e)}"
        )

# GET‑эндпоинты — просто возвращают уже вычисленный результат
@app.get('/api')
async def api():
    """Возвращает полный результат предсказания"""
    return result

@app.get("/api/pred_val")
async def get_pred_val():
    """Возвращает метрики предсказания"""
    if result["metrics"] is None:
        raise HTTPException(status_code=404, detail="Метрики ещё не рассчитаны")
    return result["metrics"]

@app.get("/api/current")
async def get_current():
    """Возвращает текущие значения"""
    if result["vals"] is None:
        raise HTTPException(status_code=404, detail="Текущие значения ещё не рассчитаны")
    return result["vals"]

@app.get("/api/history")
async def get_history():
    """Возвращает историю"""
    if result["history"] is None:
        raise HTTPException(status_code=404, detail="История ещё не рассчитана")
    return result["history"]

@app.get("/api/description")
async def get_description():
    """Возвращает описание статуса"""
    if result["overall_status"]["description"] is None:
        raise HTTPException(status_code=404, detail="Описание ещё не рассчитано")
    return result["overall_status"]["description"]

@app.get("/api/actions")
async def get_actions():
    """Возвращает рекомендации"""
    if result["overall_status"]["actions"] is None:
        raise HTTPException(status_code=404, detail="Рекомендации ещё не рассчитаны")
    return result["overall_status"]["actions"]

@app.get("/api/risk")
async def get_risk_endpoint():
    """Возвращает рекомендации"""
    if result["risk_zone"] is None:
        raise HTTPException(status_code=404, detail="Риск ещё не рассчитан")
    return result["risk_zone"]

@app.get("/api/adm")
async def get_adm_endpoint():
    """Возвращает рекомендации"""
    if result["for_admin"] is None:
        raise HTTPException(status_code=404, detail="Рекомендации ещё не рассчитаны")
    return result["for_admin"]

# @app.get("/docs")
# async def some_endpoint(request: Request):
#     client_ip = request.client.host
#     return {"client_ip": client_ip}


@app.get("/api/status")
async def status():
    uptime = int(1)
    
    return {
        "status": "ok",
        "timestamp": dt.now(),
        "version": "1.0.0",
        "uptime": uptime,
        "message": "Service is running normally"
    }


def get_client_ip(request: Request) -> str:
    """Получает реальный IP клиента с учётом прокси"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Берём первый IP в списке (оригинальный клиент)
        return forwarded.split(",")[0].strip()
    return request.client.host

# @app.middleware("http")
# async def block_ips_middleware(request: Request, call_next):
#     client_ip = get_client_ip(request)
#     if client_ip in BLOCKED_IPS:
#         return JSONResponse(
#             status_code=403,
#             content={"detail": "Your IP is blocked"}
#         )
#     response = await call_next(request)
#     return response


@app.get("/main")
async def read_root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(html)


class Message(BaseModel):
    text: str

# # Эндпоинт для получения сообщения
# @app.post("/send")
# async def receive_message(message: Message):
#     print(f"Получено сообщение: {message.text}")
#     # Обрабатываем сообщение
#     if message.text == "hi world":
#         response_text = "Hello back!"
#     else:
#         response_text = f"Received: {message.text}"
#     return {"response": response_text, "status": "success"}


# @app.post("/gas-data/")
# async def receive_gas_data(gas_data: list[GasData]):
#     print(f"\nПолучено {len(gas_data)} записей газа:")
#     for i, item in enumerate(gas_data):
#         print(f"{i+1}. {item.datetime} | NO: {item.NO} | NO2: {item.NO2} | SO2: {item.SO2} | CO: {item.CO}")
#     return {
#         "status": "success",
#         "received_count": len(gas_data),
#         "first_datetime": gas_data[0].datetime if gas_data else None
#     }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
