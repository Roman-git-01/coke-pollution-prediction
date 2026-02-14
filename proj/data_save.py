import pandas as pd
import logging

START_DATA_PATH = "DATA_csv/start_data.csv"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_start_data():
    try:
        return load_start_data.df1
    except: 
        load_start_data.df1 = pd.read_csv(START_DATA_PATH )
        load_start_data.df1 = load_start_data.df1[["datetime", "NO", "NO2", "SO2", "CO"]]
        return load_start_data.df1
    
df_save = load_start_data()
    
def get_data_save():
    global df_save
    logger.info("      Данные отправлены!")

    return df_save

def set_data_save(df):
    global df_save
    logger.info("      Данные хранилища обновлены!")
    df_save = df
    