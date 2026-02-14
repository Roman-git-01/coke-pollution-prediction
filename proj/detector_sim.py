import requests
import pandas as pd
import datetime as dt
import time

DATA_PATH_1 = "DATA_csv/tags_1.csv"
DATA_PATH_2 = "DATA_csv/tags_2.csv"
DATA_PATH_3 = "DATA_csv/tags_3.csv"
URL = "http://46.173.26.164:8000/data"
# URL = "http://localhost:8000/data"

N = 10
S = 3       #переод (секунд/раз)


def load_data():
    try:
        return load_data.df1
    except:
        df_1 = pd.read_csv(DATA_PATH_1)
        df_2 = pd.read_csv(DATA_PATH_2)
        df1 = pd.concat([df_1, df_2]).sort_values('datetime').reset_index(drop=True)
        df1 = df1.drop('Unnamed: 0', axis=1)
        load_data.df1 = df1
        return df1

def post_data():
    try:
        post_data.count += 1
    except:
        post_data.count = 1

    post_data_post(load_data().iloc[100 + post_data.count])
    # print(f"Data = {load_data().iloc[100 + post_data.count].to_json()}")

def post_data_post(df):
    try: 
        post_data_post.count += 1
    except: 
        post_data_post.count = 0
    gas_data_list = df.to_dict()
    response = requests.post(URL, json=gas_data_list)
    # json_data = df.to_json()
    # response = requests.post(URL, json=json_data)  
    print(f"{dt.datetime.now()} | JSON Data Response: {response} | {post_data_post.count}") 

    print(f"{dt.datetime.now()} | Статус: {response.status_code}")
    if response.status_code != 200:
        print(f"Ошибка: {response.text}")
    else:
        print("Успешно отправлено")


def post_data_incorrect():
    try: 
        post_data_incorrect.count += 1
    except: 
        post_data_incorrect.count = 0

    response = requests.post(URL, json={"datetime": "2026-01-12 00:10:00.60",
                                        "NO": 90.0,
                                        "NO2": 90.0,
                                        "SO2": 90.0,
                                        "CO": 90.0})
    # json_data = df.to_json()
    # response = requests.post(URL, json=json_data)  
    print(f"{dt.datetime.now()} | JSON Data Response: {response} | {post_data_incorrect.count}") 

def main():
    for i in range(1000):
        post_data()
        time.sleep(S)

# main()
# post_data_incorrect()