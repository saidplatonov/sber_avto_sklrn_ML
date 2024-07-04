import json

import dill
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, field_validator, Field
from typing import Optional
import random
from datetime import datetime
import pandas as pd

app = FastAPI()


@app.get('/status')
def status():
    return "I'm OK"


class SessionDataForm(BaseModel):
    session_id: Optional[str] = None
    client_id: Optional[str] = None
    visit_date: Optional[int] = None
    visit_time: Optional[str] = None
    visit_number: Optional[int] = None
    utm_source: Optional[str] = None
    utm_medium: Optional[str] = None
    utm_campaign: Optional[str] = None
    utm_adcontent: Optional[str] = None
    utm_keyword: Optional[str] = None
    device_category: Optional[str] = None
    device_os: Optional[str] = None
    device_brand: Optional[str] = None
    device_model: Optional[str] = None
    device_screen_resolution: Optional[str] = None
    device_browser: Optional[str] = None
    geo_country: Optional[str] = None
    geo_city: Optional[str] = None


class InputCR(BaseModel):
    cr: int = Field(default=1, description="A value that must be 0 or 1")

    @field_validator('cr')
    def check_value(cls, v):
        if v not in [0, 1]:
            raise ValueError('Value must be 0 or 1')
        return v


@app.get('/get_test_json', response_model=SessionDataForm)
def get_json_for_api_test(cr: int = Query(default=1, description="A value that must be 0 or 1")):
    """
    Получаем случайный JSON файл с соответствующим CR
    :param cr: 0 or 1, ConversionRate
    :return: json
    """
    try:
        # Валидация параметра
        input_data = InputCR(cr=cr)

        # Загрузка данных
        test_df_file = "tmp/data_to_test_api.pkl"
        test_df = pd.read_pickle(test_df_file)

        # Фильтрация и выбор случайной строки
        rnd_json = test_df[test_df['conversion_rate'] == cr].sample(n=1).drop(columns=['conversion_rate']).to_json(
            orient='records')
        return json.loads(rnd_json)[0]
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Data file not found. Please check the file path.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred: " + str(e))


file_model = "model/model_pipe.pkl"
with open(file_model, 'rb') as pkl_pipe:
    model_pipe = dill.load(pkl_pipe)
print(model_pipe['metadata'])


class Prediction(BaseModel):
   Session_ID: object
   Conversion_Rate_Prediction: int


@app.post('/predict', response_model=Prediction)
def get_predict_by_session_json(session_data: SessionDataForm):
    df = pd.DataFrame.from_dict([session_data.dict()])
    m_predict = model_pipe['model'].predict(df)[0]

    return {
        "Session_ID": df['session_id'].to_string(index=False),
        "Conversion_Rate_Prediction": m_predict
    }