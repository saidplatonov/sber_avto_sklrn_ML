import json

import dill
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, field_validator, Field
from typing import Optional

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


@app.get('/all_feature_name', response_model=list)
def get_all_feature_in():
    features_list = model_pipe['model'].named_steps['preprocessor'].named_steps['columns_transform'].feature_names_in_
    print(features_list)
    return features_list


@app.get('/get_feature_imp', response_model= list)
def get_feature_importance(f: str = Query(default=None, description="Must match the name of the Fitch on the input "
                                                                 "(you can see the list of feature_in by GET request "
                                                                 "\"/all_feature_name\")")):
    # Извлечение обученного классификатора из пайплайна
    classifier = model_pipe['model'].named_steps['classifier']

    # Извлечение преобразователя признаков из пайплайна
    preprocessor = model_pipe['model'].named_steps['preprocessor']

    # Извлечение всех шагов из ColumnTransformer
    numerical_features = preprocessor.named_steps['columns_transform'].transformers_[0][2]
    categorical_features = preprocessor.named_steps['columns_transform'].transformers_[1][1].named_steps[
        'ohe_encoder'].get_feature_names_out()

    # Объединение имен признаков
    all_features = list(numerical_features) + list(categorical_features)

    # Получение коэффициентов модели
    coefficients = classifier.coef_[0]

    # Создание DataFrame для удобства просмотра
    feature_importance = pd.DataFrame({'Feature': all_features, 'Coefficient': coefficients})
    feature_importance.sort_values(by='Coefficient', ascending=False, inplace=True)


    # Создаем словарь для сопоставления закодированных имен с исходными
    features_In_list = model_pipe['model'].named_steps['preprocessor'].named_steps[
        'columns_transform'].feature_names_in_

    # Функция для расшифровки названия фичи
    def decode_feature_name(feature_name):
        # Разделяем закодированное имя фичи на индекс и закодированное имя
        parts = feature_name.split('_', 1)
        if len(parts) == 2 and parts[0].startswith('x'):
            index = int(parts[0][1:])
            original_name = features_In_list[index]
            return original_name
        return feature_name

    # Применяем функцию к колонке 'Feature' в вашем DataFrame
    feature_importance.insert(0, 'Feature_in', feature_importance['Feature'].apply(decode_feature_name))

    if f is None:
        result_feat = feature_importance.head(25).to_json(orient='records')
    else:
        result_feat = feature_importance.loc[feature_importance['Feature_in'] == f].head(25).to_json(orient='records')

    return json.loads(result_feat)